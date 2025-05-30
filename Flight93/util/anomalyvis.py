import pandas as pd
import matplotlib.pyplot as plt
import os

# Create output directory
os.makedirs('../anomalyvis', exist_ok=True)

def plot_comparison(normal_data, anomaly_data, title, save_path, y_label='Value'):
    """Plot comparison between normal and anomalous data"""
    plt.figure(figsize=(14, 4))
    plt.plot(anomaly_data, label='Anomaly', color='red', linestyle='--')
    plt.plot(normal_data, label='Normal', color='blue')
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel(y_label)
    plt.legend()
    # Remove grid
    plt.grid(False)
    # Set y-axis range
    plt.ylim(-13, 13)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def process_anomaly(anomaly_type, normal_df, anomaly_df):
    """Handle comparison plots for a single type of anomaly"""
    normal_navvd = normal_df['navvd']
    anomaly_navvd = anomaly_df['navvd']

    # Full data comparison (16000 samples)
    plot_comparison(
        normal_navvd,
        anomaly_navvd,
        f'{anomaly_type.title()} Anomaly Data (16000 points)',
        f'../anomalyvis/{anomaly_type}16000.png'
    )

    # Comparison of the last 3200 samples (index 12800 to 16000)
    end_idx = len(normal_navvd)
    start_idx = end_idx - 3200
    plot_comparison(
        normal_navvd[start_idx:end_idx],
        anomaly_navvd[start_idx:end_idx],
        f'{anomaly_type.title()} Anomaly Last 3200 Points',
        f'../anomalyvis/{anomaly_type}3200.png'
    )

# Load data
normal_df = pd.read_csv('../data/o_thor_flight93_selected.csv')
drift_df = pd.read_csv('../data/o_thor_flight93_drift.csv')
bias_df = pd.read_csv('../data/o_thor_flight93_bias.csv')

# Process drift anomaly
process_anomaly('drift', normal_df, drift_df)

# Process bias anomaly
process_anomaly('bias', normal_df, bias_df)

print("Visualization files have been saved to the 'anomalyvis' directory!")
