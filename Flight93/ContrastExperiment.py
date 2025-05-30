import time

from scipy.signal import savgol_filter
import os
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISM'] = '1'
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# Enable TF deterministic operations
tf.config.experimental.enable_op_determinism()
from scipy.stats import genextreme
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tqdm.keras import TqdmCallback
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import importlib
# ------------------------------
# Global index variables (total 16000 data points)
# ------------------------------
TOTAL_DATA_POINTS = 16000
TRAIN_END_IDX = 12800  # Training set: first 12800 samples
TEST_BEGIN_IDX = 12800  # Test set start index
TEST_END_IDX = 16000  # Test set end index
model_names = [
    "LSTM",
    "CNNBiGRU",
    "DCATCN",
    "CNNBiLSTMAttention",
    "ConvLSTM",

]
# ------------------------------
# Parameter settings
# ------------------------------
dataset_prefix = "data/o_thor_flight93"  # Modify according to actual dataset name
experiment_data_types = ['bias', 'drift']
train_file = f"{dataset_prefix}_selected.csv"
D = 15  # Sliding window size


# Fix random seed
def set_seed(seed=42):
    # Python hash & random
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # TensorFlow
    tf.random.set_seed(seed)

#  Data smoothing function: apply Savitzky–Golay filter to all features except "time"
# ------------------------------
def smooth_dataframe(df, window_length=51, polyorder=3):
    df_smoothed = df.copy()
    for col in df.columns:
        if col != 'time':
            wl = window_length
            if len(df[col]) < window_length:
                wl = len(df[col]) if len(df[col]) % 2 == 1 else len(df[col]) - 1
            if wl < polyorder + 2:
                wl = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
            df_smoothed[col] = savgol_filter(df[col].values, window_length=wl, polyorder=polyorder)
    return df_smoothed


# ------------------------------
#  Normalization-related functions (using training data mean and std)
# ------------------------------
def fit_normalization_params(df):
    params = {}
    for col in df.columns:
        if col != 'time':
            params[col] = {"mean": df[col].mean(), "std": df[col].std()}
    return params


def apply_normalization(df, norm_params):
    df_normalized = df.copy()
    for col in df_normalized.columns:
        if col != 'time' and col in norm_params:
            df_normalized[col] = (df_normalized[col] - norm_params[col]["mean"]) / (norm_params[col]["std"] + 1e-8)
    return df_normalized


# ------------------------------
# Data preprocessing
# ------------------------------
def create_sequences(inputs, targets, window_size):
    X, y = [], []
    for i in range(len(inputs) - window_size):
        X.append(inputs[i:i + window_size])
        y.append(targets[i + window_size])
    return np.array(X), np.array(y)

# Threshold calculation function
def calculate_threshold(residuals, alpha):
    residuals = np.abs(residuals)
    import math
    N = len(residuals)
    k = math.ceil((1 - alpha) * N)
    residuals_tail = np.sort(residuals)[-k:]
    try:
        params = genextreme.fit(residuals_tail)
    except:
        params = genextreme.fit(residuals_tail, floc=0)
    threshold = genextreme.ppf(alpha, *params)
    return threshold, params

# ------------------------------
#  Visualization function: plot original vs smoothed data (upper and lower subplots)
# ------------------------------
def plot_time_comparison(time, original, smoothed, title, save_path):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time, original, label='origin', color='blue', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title + "：origin")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(time, smoothed, label='smooth', color='red', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title + "：smooth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
# ------------------------------
# Visualization function: plot prediction vs ground truth
# ------------------------------
def plot_prediction_vs_true(true_vals, pred_vals, title, save_path, ano_vals=None):
    plt.figure(figsize=(12, 6))
    plt.ylim(-7, 7)
    if ano_vals is not None:
        plt.plot(np.arange(len(ano_vals)), ano_vals, label='anomaly', color='red', linestyle='--', linewidth=2)
    plt.plot(np.arange(len(true_vals)), true_vals, label='true', color='green', linewidth=2)
    plt.plot(np.arange(len(pred_vals)), pred_vals, label='prediction', color='blue', linestyle='--', linewidth=2)
    plt.title(title)
    plt.xlabel('Sample index')
    plt.ylabel('value')
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ------------------------------
# Visualization function: plot anomaly detection results
# ------------------------------
def plot_anomaly(anomaly_scores, threshold, pred_labels, title, save_path):
    plt.figure(figsize=(10, 4))
    plt.plot(anomaly_scores, label='anomaly score')
    plt.axhline(threshold, color='r', linestyle='--', label='threshold')
    plt.scatter(np.where(pred_labels == 1)[0], anomaly_scores[pred_labels == 1],
                c='red', s=10, label='detected anomaly')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ------------------------------
# Visualization function: plot recovery results
# ------------------------------
def plot_recovery(true_vals, recovered_vals, title, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(true_vals)), true_vals, label='original data', color='blue', linewidth=2)
    plt.plot(np.arange(len(recovered_vals)), recovered_vals, label='recovered data', color='green', linestyle='--',
             linewidth=2)
    plt.ylim(-10, 10)
    plt.title(title)
    plt.xlabel('Sample index')
    plt.ylabel('value')
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ------------------------------
# Custom callback to record loss
# ------------------------------
class LossHistory(Callback):
    def on_train_begin(self, logs=None):
        self.losses = {'batch': [], 'epoch': []}
        self.val_losses = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs=None):
        self.losses['batch'].append(logs.get('loss'))
        self.val_losses['batch'].append(logs.get('val_loss'))

    def on_epoch_end(self, epoch, logs=None):
        self.losses['epoch'].append(logs.get('loss'))
        self.val_losses['epoch'].append(logs.get('val_loss'))

    def plot_loss(self, save_path):
        plt.figure(figsize=(12, 6))
        plt.plot(self.losses['epoch'], label='Training Loss')
        plt.plot(self.val_losses['epoch'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(save_path, dpi=300)
        plt.close()


# ------------------------------
# Residual calculation (with smoothing)
# ------------------------------
def process_residuals(model, X, y_true):
    y_pred = model.predict(X, verbose=0).flatten()
    residual = np.abs(y_true - y_pred)
    smoothed = savgol_filter(residual, window_length=161, polyorder=3)
    return smoothed


# ------------------------------
# Test set experiment process
# ------------------------------
def run_experiment(test_file, exp_type, window_size, model, threshold, feature_list, norm_params,vis_dir,timing_file):
    print(f"\n---- Running experiment: {exp_type} ----")
    test_datahavey = pd.read_csv(test_file)
    test_data = smooth_dataframe(test_datahavey)
    test_data = apply_normalization(test_data, norm_params)

    test_begin_idx = TEST_BEGIN_IDX
    test_end_idx = TEST_END_IDX


    # Test input features (excluding navvd) and target values (navvd)
    test_inputs = test_data.iloc[test_begin_idx:test_end_idx][feature_list].values
    test_target = test_data.iloc[test_begin_idx:test_end_idx]['navvd'].values
    X_test, y_test = create_sequences(test_inputs, test_target, window_size)

    # —— Inference Timing ——
    start_inf = time.time()
    _ = model.predict(X_test, verbose=0).flatten()
    end_inf = time.time()
    infer_time = end_inf - start_inf
    print(f"{exp_type} infer time: {infer_time:.2f} seconds")
    try:
        with open(timing_file, 'a') as f:
            f.write(f"{CURRENT_MODEL},infer,{exp_type},{infer_time:.2f}\n")
    except NameError:
        pass

    or_data = pd.read_csv(train_file)
    or_data = smooth_dataframe(or_data)
    nor_data = apply_normalization(or_data, norm_params)
    normal_oinputs = nor_data.iloc[test_begin_idx:test_end_idx][feature_list].values
    normal_otarget = nor_data.iloc[test_begin_idx:test_end_idx]['navvd'].values
    _, y_normaltest = create_sequences(normal_oinputs, normal_otarget, window_size)

    # Plot test set prediction vs true values (without anomalies)
    y_pred = model.predict(X_test, verbose=0).flatten()
    plot_prediction_vs_true(y_normaltest, y_pred,
                            f"{exp_type} Test Data: Prediction vs Actual (No Anomaly)",
                            os.path.join(vis_dir, f'{exp_type}_prediction_vs_actual.png'), y_test)

    test_smoothed = process_residuals(model, X_test, y_test)
    anomaly_scores = test_smoothed
    pred_labels = (anomaly_scores > threshold).astype(int)
    plot_anomaly(anomaly_scores, threshold, pred_labels,
                 f"{exp_type} Test Data Anomaly Detection",
                 os.path.join(vis_dir, f'{exp_type}_anomaly_analysis.png'))

    if 'anomaly' in test_datahavey.columns:
        true_labels = test_datahavey.iloc[test_begin_idx + window_size:test_end_idx]['anomaly'].values
        TP = np.sum((pred_labels == 1) & (true_labels == True))
        TN = np.sum((pred_labels == 0) & (true_labels == False))
        FP = np.sum((pred_labels == 1) & (true_labels == False))
        FN = np.sum((pred_labels == 0) & (true_labels == True))
        metrics = {
            "TPR": TP / (TP + FN) * 100 if (TP + FN) > 0 else 0,
            "FPR": FP / (TN + FP) * 100 if (TN + FP) > 0 else 0,
            "ACC": (TP + TN) / (TP + TN + FP + FN) * 100
        }
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"Evaluation Results for {exp_type}:")
        for k, v in metrics.items():
            print(f"{k}: {v:.2f}%")
        print(f"f1_score: {f1_score:.4f}")
        print(f"TP: {TP:.4f}")
        print(f"TN: {TN:.4f}")
        print(f"FP: {FP:.4f}")
        print(f"FN: {FN:.4f}")
        results_path = os.path.join(vis_dir, f"{exp_type}_metrics.txt")
        #
        with open(results_path, 'w') as f:
            f.write(f"Evaluation Results for {exp_type}:\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.2f}%\n")
            f.write(f"f1_score: {f1_score:.4f}\n")
            f.write(f"TP: {TP:.4f}\n")
            f.write(f"TN: {TN:.4f}\n")
            f.write(f"FP: {FP:.4f}\n")
            f.write(f"FN: {FN:.4f}\n\n")

        y_pred_test = model.predict(X_test, verbose=0).flatten()
        y_recovered = np.where(pred_labels == 1, y_pred_test,y_normaltest)

        origin_data = pd.read_csv(train_file)
        origin_data_smoothed = smooth_dataframe(origin_data)
        origin_data_smoothed = apply_normalization(origin_data_smoothed, norm_params)
        features_origin = origin_data_smoothed.columns.drop(['time', 'navvd'])
        test_begin_origin = TRAIN_END_IDX
        test_end_origin = TOTAL_DATA_POINTS
        origin_inputs = origin_data_smoothed.iloc[test_begin_origin:test_end_origin][features_origin].values
        origin_target = origin_data_smoothed.iloc[test_begin_origin:test_end_origin]['navvd'].values
        _, y_test_origin = create_sequences(origin_inputs, origin_target, window_size)
        mae_val = np.mean(np.abs(y_recovered - y_test_origin))
        mse_val = np.mean((y_recovered - y_test_origin) ** 2)
        print(f"Data Recovery Results for {exp_type}: MAE = {mae_val:.4f}, MSE = {mse_val:.4f}")
        with open(results_path, 'a') as f:
            f.write(f"Data Recovery Results for {exp_type}:\n")
            f.write(f"  MAE = {mae_val:.4f}\n")
            f.write(f"  MSE = {mse_val:.4f}\n")

        plot_recovery(y_test_origin, y_recovered,
                      f"{exp_type} data-recovery:origin vs recovery",
                      os.path.join(vis_dir, f'{exp_type}_original_vs_recovered.png'))

        plt.figure(figsize=(12, 4))
        time_indices = test_data.iloc[test_begin_idx + window_size:test_end_idx]['time'].values
        plt.plot(time_indices, y_test, label='testdata', linewidth=1.5)

        anomaly = pred_labels.astype(bool)
        starts = np.where((~anomaly[:-1]) & (anomaly[1:]))[0] + 1
        ends = np.where((anomaly[:-1]) & (~anomaly[1:]))[0] + 1
        if anomaly[0]:
            starts = np.insert(starts, 0, 0)
        if anomaly[-1]:
            ends = np.append(ends, len(anomaly))
        for s, e in zip(starts, ends):
            plt.axvspan(time_indices[s], time_indices[e - 1], color='red', alpha=0.3)

        plt.title(f'{exp_type} Test Data and Anomaly Interval')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'{exp_type}_anomaly_over_time.png'), dpi=300)
        plt.close()
    else:
        plt.figure(figsize=(12, 6))
        plt.plot(anomaly_scores, label='Anomaly Score')
        plt.axhline(threshold, color='r', linestyle='--', label='Threshold')
        plt.title(f'{exp_type} Anomaly Detection Result')
        plt.ylabel('Smoothed Residual')
        plt.xlabel('Sample Index')
        plt.legend()
        plt.savefig(os.path.join(vis_dir, f'{exp_type}_anomaly_detection.png'), dpi=300)
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Test Data')
        plt.title(f'{exp_type} Test Data Visualization')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(os.path.join(vis_dir, f'{exp_type}_test_data.png'), dpi=300)
        plt.close()


# ------------------------------
# Main process: training, threshold calculation, visualization of training/test predictions
# ------------------------------
if __name__ == '__main__':
    set_seed()
    train_end_idx = TRAIN_END_IDX
    origin_train = pd.read_csv(train_file)[:train_end_idx]
    origin_train_smoothed = smooth_dataframe(origin_train)

    norm_params = fit_normalization_params(origin_train_smoothed)
    origin_train_smoothed = apply_normalization(origin_train_smoothed, norm_params)

    # Training input: all features except 'time' and 'navvd'
    features_train = origin_train_smoothed.columns.drop(['time', 'navvd'])
    train_inputs = origin_train_smoothed[:train_end_idx][features_train].values
    train_target = origin_train_smoothed[:train_end_idx]['navvd'].values

    X_train, y_train = create_sequences(train_inputs, train_target,D)

    # For each model: train, calculate threshold, test experiment
    for model_name in model_names:
        print(f"\n=== Running model: {model_name} ===")
        # create dir
        vis_dir = os.path.join('vis', model_name)
        result_dir = os.path.join('timecost', model_name)
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        timing_file = os.path.join(result_dir, "timing.csv")
        if not os.path.exists(timing_file):
            with open(timing_file, 'w') as f:
               f.write("model,phase,exp_type,time_s\n")
        CURRENT_MODEL = model_name

        if 'time' in origin_train.columns and 'navvd' in origin_train.columns:
            plot_time_comparison(origin_train['time'], origin_train['navvd'],
                                 origin_train_smoothed['navvd'],
                                 "train data: navvd ",
                                 os.path.join(vis_dir, 'training_navvd_comparison.png'))

        # dynamic import
        module = importlib.import_module(f"Flight93.model.{model_name}")
        build_model = getattr(module, "build_model")
        model = build_model((D, X_train.shape[2]))
        model.summary()


        split = int(0.8 * len(X_train))
        history_cb = LossHistory()
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        start_train = time.time()
        model.fit(
            X_train[:split], y_train[:split],
            validation_data=(X_train[split:], y_train[split:]),
            epochs=100, batch_size=128,
            callbacks=[TqdmCallback(verbose=0), es, history_cb],
            verbose=0
        )
        end_train = time.time()
        train_time = end_train - start_train
        print(f"{model_name} train time : {train_time:.2f} 秒")

        with open(timing_file, 'a') as f:
            f.write(f"{model_name},train,,{train_time:.2f}\n")

        history_cb.plot_loss(os.path.join(vis_dir, 'loss.png'))

        # calculate residual and thre
        train_sm = process_residuals(model, X_train, y_train)
        threshold, _ = calculate_threshold(train_sm, alpha=0.90)

        # test data exp
        feature_list = list(features_train)
        for dt in experiment_data_types:
            test_file_dt = f"{dataset_prefix}_{dt}.csv"
            run_experiment(
                test_file_dt, dt, D,
                model, threshold,
                feature_list, norm_params,
                vis_dir,
                timing_file
            )