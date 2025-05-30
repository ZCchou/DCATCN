import pandas as pd
from Flight93.util import anomalyinjection as aj, MIC

# Load and extract target portion from the original file
data = pd.read_csv('thor_flight93_confidence_metrics1_2013_11_05.csv')
data = data[8500:24500]
data.to_csv('thor_flight93_16000.csv', index=False)
data = pd.read_csv('thor_flight93_16000.csv')

# Specify anomaly injection range
startindex = 14800
endindex = 16000

# Inject anomalies
data_bias = aj.inject_bias_anomaly(data, 'navvd', startindex, endindex, 2)
data_drift = aj.inject_drift_anomaly(data, 'navvd', startindex, endindex)

# Extract timestamp column
time = data.iloc[:, 0]
print(time)
time.to_csv('time.csv', index=False)

# Remove time column for feature selection
data = data.iloc[:, 1:]
data_bias = data_bias.iloc[:, 1:]
data_drift = data_drift.iloc[:, 1:]

# MIC feature selection
# Define target column and empirical threshold
target_column = 'navvd'
threshold = 0.5
related_params = MIC.select_significant_parameters(data, target_column, threshold)
print("Parameters significantly correlated with the target column:", related_params)

# Ensure 'navvd' is included in selected parameters
related_params.append('navvd')

# Extract selected parameters from original data
odata = pd.read_csv('thor_flight93_16000.csv')
odata = odata[related_params]
data_drift = data_drift[related_params]
data_bias = data_bias[related_params]

# Merge timestamp with selected features
print(data)
odata = pd.concat([time, odata], axis=1)
data_drift = pd.concat([time, data_drift], axis=1)
data_bias = pd.concat([time, data_bias], axis=1)

# Save as new dataset files
odata.to_csv("o_thor_flight93_selected.csv", index=False)

# Mark anomaly segments
data_bias = aj.mark_anomalies(data_bias, startindex, endindex)
data_drift = aj.mark_anomalies(data_drift, startindex, endindex)

# Save anomaly-injected files
data_bias.to_csv('o_thor_flight93_bias.csv', index=False)
data_drift.to_csv('o_thor_flight93_drift.csv', index=False)
