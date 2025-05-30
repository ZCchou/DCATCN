import math
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import savgol_filter
from scipy.stats import genextreme
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tqdm.keras import TqdmCallback
from Flight93.model.DCATCN import build_model

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

TOTAL_DATA_POINTS = 16000
TRAIN_END_IDX = 12800
D = 20  # Sliding window size
dataset_prefix = "data/o_thor_flight93"
train_file = f"{dataset_prefix}_selected.csv"
vis_dir = os.path.join('vis', 'CrossGEVvis')
os.makedirs(vis_dir, exist_ok=True)

experiment_data_types = ['bias', 'drift']  # Added: types of test datasets

# --------------- Random Seed ---------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# --------------- datapreprocess ---------------
def smooth_dataframe(df, window_length=51, polyorder=3):
    df_sm = df.copy()
    for c in df.columns:
        if c == 'time': continue
        wl = window_length
        if len(df[c]) < wl:
            wl = len(df[c]) if len(df[c])%2 else len(df[c]) - 1
        if wl < polyorder+2:
            wl = (polyorder+2) if (polyorder+2)%2 else (polyorder+3)
        df_sm[c] = savgol_filter(df[c].values, wl, polyorder)
    return df_sm

def fit_normalization_params(df):
    params = {}
    for c in df.columns:
        if c=='time': continue
        params[c] = {'mean': df[c].mean(), 'std': df[c].std()}
    return params

def apply_normalization(df, norm_params):
    df_n = df.copy()
    for c in df_n.columns:
        if c in norm_params:
            p = norm_params[c]
            df_n[c] = (df_n[c] - p['mean'])/(p['std']+1e-8)
    return df_n

def create_sequences(X, y, window_size):
    xs, ys = [], []
    for i in range(len(X)-window_size):
        xs.append(X[i:i+window_size])
        ys.append(y[i+window_size])
    return np.array(xs), np.array(ys)

# --------------- threshold calculate（MSE/MAD/GEV） ---------------
def calculate_threshold(residuals, method='gev', alpha=0.90):
    """
    method: 'mse' | 'mad' | 'gev'
    return：threshold, params
    """
    abs_res = np.abs(residuals)
    if method == 'mse':
        sq = abs_res**2
        thr = np.mean(sq)
        return thr, None
    if method == 'mad':
        med = np.median(abs_res)
        mad = 1.4826 * np.median(np.abs(abs_res-med))
        thr = med + 3*mad
        return thr, (med, mad)
    # GEV
    N = len(abs_res)
    k = math.ceil((1 - alpha) * N)
    residuals_tail = np.sort(abs_res)[-k:]
    try:

        params = genextreme.fit(residuals_tail)
    except:
        params = genextreme.fit(residuals_tail, floc=0)
    thr = genextreme.ppf(alpha, *params)
    return thr, params
# --------------- residual ---------------
def process_residuals(model, X, y_true):
    y_pred = model.predict(X, verbose=0).flatten()
    res = np.abs(y_true - y_pred)
    return savgol_filter(res, window_length=161, polyorder=3)

# --------------- main---------------
if __name__ == '__main__':
    set_seed()

    # 1. train data
    df = pd.read_csv(train_file)[:TRAIN_END_IDX]
    df_sm = smooth_dataframe(df)
    norm_params = fit_normalization_params(df_sm)
    df_n = apply_normalization(df_sm, norm_params)

    features = df_n.columns.drop(['time','navvd'])
    X_train, y_train = create_sequences(df_n[features].values,
                                        df_n['navvd'].values, D)

    # 2. train
    model = build_model((D, X_train.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train,
              epochs=100, batch_size=128,
              validation_split=0.2,
              callbacks=[TqdmCallback(), early_stop],
              verbose=0)

    # 3. train residual
    train_res = process_residuals(model, X_train, y_train)


    # 5. compare on test data
    for exp_type in experiment_data_types:
        test_file = f"{dataset_prefix}_{exp_type}.csv"
        df_t = pd.read_csv(test_file)
        df_t_sm = smooth_dataframe(df_t)
        df_t_n = apply_normalization(df_t_sm, norm_params)

        X_test, y_test = create_sequences(
            df_t_n[features].values,
            df_t_n['navvd'].values, D
        )
        # real label
        true_labels = df_t['anomaly'].values[D:]

        # test residual
        test_res = process_residuals(model, X_test, y_test)

        print(f"\n---- {exp_type.upper()} Anomaly Detection ----")
        for method in ['mse','mad','gev']:
            thr, _ = calculate_threshold(train_res, method=method)
            # MSE 方法使用残差平方，其余使用原残差
            scores = test_res**2 if method=='mse' else test_res
            pred_labels = (scores > thr).astype(int)

            TP = np.sum((pred_labels==1)&(true_labels==1))
            TN = np.sum((pred_labels==0)&(true_labels==0))
            FP = np.sum((pred_labels==1)&(true_labels==0))
            FN = np.sum((pred_labels==0)&(true_labels==1))

            TPR = TP/(TP+FN)*100 if TP+FN>0 else 0
            FPR = FP/(TN+FP)*100 if TN+FP>0 else 0
            ACC = (TP+TN)/(TP+TN+FP+FN)*100
            precision = TP/(TP+FP) if TP+FP>0 else 0
            recall = TP/(TP+FN) if TP+FN>0 else 0
            f1 = 2*(precision*recall)/(precision+recall) if precision+recall>0 else 0

            print(f"[{method.upper()}] Threshold={thr:.3f} | TPR={TPR:.2f}%  FPR={FPR:.2f}%  "
                  f"ACC={ACC:.2f}%  Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")
