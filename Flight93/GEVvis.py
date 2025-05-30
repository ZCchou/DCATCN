import os
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import savgol_filter
from scipy.stats import genextreme
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tqdm.keras import TqdmCallback
from Flight93.model.CrossAttentionBiTCN import build_model


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

TOTAL_DATA_POINTS = 16000
TRAIN_END_IDX = 12800
D = 20  # 滑动窗口大小
dataset_prefix = "data/o_thor_flight93"
train_file = f"{dataset_prefix}_selected.csv"
vis_dir = os.path.join('vis', 'CrossGEVvis')
os.makedirs(vis_dir, exist_ok=True)

# --------------- random seed ---------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# --------------- data process ---------------
def smooth_dataframe(df, window_length=151, polyorder=3):
    df_sm = df.copy()
    for c in df.columns:
        if c == 'time': continue
        wl = window_length
        if len(df[c]) < wl:
            wl = len(df[c]) if len(df[c]) % 2 else len(df[c]) - 1
        if wl < polyorder + 2:
            wl = (polyorder + 2) if (polyorder + 2) % 2 else (polyorder + 3)
        df_sm[c] = savgol_filter(df[c].values, wl, polyorder)
    return df_sm

def fit_normalization_params(df):
    params = {}
    for c in df.columns:
        if c == 'time': continue
        params[c] = {'mean': df[c].mean(), 'std': df[c].std()}
    return params

def apply_normalization(df, norm_params):
    df_n = df.copy()
    for c in df_n.columns:
        if c in norm_params:
            p = norm_params[c]
            df_n[c] = (df_n[c] - p['mean'])/(p['std'] + 1e-8)
    return df_n

def create_sequences(X, y, window_size):
    xs, ys = [], []
    for i in range(len(X) - window_size):
        xs.append(X[i:i+window_size])
        ys.append(y[i+window_size])
    return np.array(xs), np.array(ys)

# --------------- GEV calculate ---------------
def calculate_threshold(residuals, alpha=0.90):

    N = len(residuals)
    k = math.ceil((1 - alpha) * N)
    residuals_tail = np.sort(residuals)[-k:]
    try:
        params = genextreme.fit(residuals_tail)
    except:
        params = genextreme.fit(residuals_tail, floc=0)
    threshold = genextreme.ppf(alpha, *params)
    return threshold, params

def plot_gev_fit(residuals, params, threshold, save_path, alpha=0.90):
    """plot GEV and threshold(tail) """
    N = len(residuals)
    k = math.ceil((1 - alpha) * N)
    residuals_tail = np.sort(residuals)[-k:]
    shape, loc, scale = params

    plt.figure(figsize=(10,6))
    counts, bins, _ = plt.hist(
        residuals_tail, bins=30, density=True,
        alpha=0.6, color='gray', label='Tail Empirical'
    )
    x = np.linspace(bins.min(), bins.max(), 200)
    pdf = genextreme.pdf(x, shape, loc=loc, scale=scale)
    plt.plot(x, pdf, 'r-', lw=2, label='Fitted GEV PDF')
    plt.axvline(
        threshold, linestyle='--', lw=2, color='b',
        label=f'Threshold = {threshold:.3f}'
    )
    plt.title('GEV&threshold')
    plt.xlabel('Residual')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# --------------- residual ---------------
def process_residuals(model, X, y_true):
    y_pred = model.predict(X, verbose=0).flatten()
    res = np.abs(y_true - y_pred)
    return savgol_filter(res, window_length=161, polyorder=3)

# --------------- Main Process ---------------
if __name__ == '__main__':
    set_seed()
    # 1. Load and preprocess training data
    df = pd.read_csv(train_file)[:TRAIN_END_IDX]
    df_sm = smooth_dataframe(df)
    norm_params = fit_normalization_params(df_sm)
    df_n = apply_normalization(df_sm, norm_params)

    # 2. Build sequence inputs
    features = df_n.columns.drop(['time','navvd'])
    X_train, y_train = create_sequences(
        df_n[features].values, df_n['navvd'].values, D
    )

    # 3. Build and train model
    model = build_model((D, X_train.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=100, batch_size=128,
        validation_split=0.2,
        callbacks=[TqdmCallback(), early_stop],
        verbose=0
    )

    # 4. Calculate training residuals and fit GEV threshold
    train_res = process_residuals(model, X_train, y_train)
    threshold, gev_params = calculate_threshold(train_res, alpha=0.90)



    plot_gev_fit(
        residuals=train_res,
        params=gev_params,
        threshold=threshold,
        save_path=os.path.join(vis_dir, 'gev_threshold_visualization.png'),
        alpha=0.90
    )

    print(f"GEV threshold (α=0.90): {threshold:.3f}，save at {vis_dir}/gev_threshold_visualization.png")
