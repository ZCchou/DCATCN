import pandas as pd
from Flight93.util import Normalize as nm, anomalyinjection as aj, MIC

# Extract and preprocess dataset

# 1. Load raw data and extract the desired range
data = pd.read_csv('thor_flight93_confidence_metrics1_2013_11_05.csv')
data = data[7000:23000]
data.to_csv('thor_flight93_16000.csv', index=False)

# Normalization
ndata = nm.min_max_normalize_skip_first_column(data)
ndata.to_csv('thor_flight93_normalized.csv', index=False)
time = ndata.iloc[:, 0]
print(time)
time.to_csv('time.csv', index=False)
data = ndata.iloc[:, 1:]

# MIC feature selection
target_column = 'navvd'  # Target column name
threshold = 0.7  # Empirical threshold
related_params = MIC.select_significant_parameters(data, target_column, threshold)
print("Parameters significantly related to the target column:", related_params)

# Compute and output MIC correlation scores as anomaly scores, sorted in descending order
mic_scores = {}
for col in related_params:
    score = MIC.compute_mic(data[col].values, data[target_column].values)
    mic_scores[col] = score

# Sort by descending score
sorted_scores = sorted(mic_scores.items(), key=lambda x: x[1], reverse=True)
print("MIC anomaly scores for each feature (descending order):")
for feature, score in sorted_scores:
    print(f"{feature}: {score:.4f}")

# Extract relevant parameter columns from the original data
related_params.append('navvd')
data = data[related_params]
merged_data = pd.concat([time, data], axis=1)
merged_data.to_csv('thor_flight93_selected.csv', index=False)

# Load the selected data
data = pd.read_csv('thor_flight93_selected.csv')

# Define anomaly injection range
startindex = 14800
endindex = 16000

# Inject anomalies
data_bias = aj.inject_bias_anomaly(data, 'navvd', startindex, endindex, 1)
data_drift = aj.inject_drift_anomaly(data, 'navvd', startindex, endindex)

# Mark anomalies
data_bias = aj.mark_anomalies(data_bias, startindex, endindex)
data_drift = aj.mark_anomalies(data_drift, startindex, endindex)

# Save processed datasets
data_bias.to_csv('thor_flight93_bias.csv', index=False)
data_drift.to_csv('thor_flight93_drift.csv', index=False)

