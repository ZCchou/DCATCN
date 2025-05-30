import numpy as np
import pandas as pd


def inject_bias_anomaly(data: pd.DataFrame, column: str, start_index: int, end_index: int, bias: float) -> pd.DataFrame:
    """
   Inject fixed bias anomaly.

Args:
    data (pd.DataFrame): Original data
    column (str): Target column name
    start_index (int): Start index of the anomaly (inclusive)
    end_index (int): End index of the anomaly (inclusive)
    bias (float): Fixed bias value to inject

Returns:
    pd.DataFrame: A copy of the data with injected anomalies

    """
    # 1. Data validation
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")

    if column not in data.columns:
        raise ValueError(f"Column {column} does not exist in the data")

    if start_index < 0 or end_index > len(data) or start_index >= end_index:
        raise ValueError("Invalid start or end index")

    # 2. Create a copy of the data
    data_copy = data.copy()

    # 3. Inject fixed bias
    data_copy.loc[start_index:end_index, column] += bias

    return data_copy


def inject_drift_anomaly(data: pd.DataFrame, column: str, start_index: int, end_index: int) -> pd.DataFrame:
    """
    Inject a linear drift anomaly into a time series.

    Args:
        data (pd.DataFrame): Original dataset
        column (str): Target column name
        start_index (int): Start index of the anomaly (inclusive)
        end_index (int): End index of the anomaly (inclusive)

    Returns:
        pd.DataFrame: A copy of the dataset with the drift anomaly injected
    """
    # 1. Data validation
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")

    if column not in data.columns:
        raise ValueError(f"Column {column} does not exist in the dataset")

    if start_index < 0 or end_index > len(data) or start_index >= end_index:
        raise ValueError("Invalid start or end index")

    # 2. Create a copy of the data
    data_copy = data.copy()

    # 3. Generate 1200 linearly spaced points from 2 to 3
    drift = np.linspace(2, 3, 1200)

    # 4. Get the length of the injection range
    num_points = end_index - start_index

    if num_points > len(drift):
        raise ValueError("Injection range exceeds available drift points (1200)")

    # 5. Select the first num_points values from the drift sequence
    drift_segment = drift[:num_points]

    # 6. Inject drift into the specified range
    data_copy.loc[start_index:end_index - 1, column] += drift_segment

    return data_copy


def mark_anomalies(data, start_index, end_index):
    """
    Mark a specified segment in the data as anomalous.

    Args:
        data (pd.DataFrame): Target dataset containing at least one numeric column.
        start_index (int): Start index of the anomaly segment (inclusive).
        end_index (int): End index of the anomaly segment (exclusive).

    Returns:
        pd.DataFrame: DataFrame with the anomaly segment marked.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")

    if start_index < 0 or end_index > len(data) or start_index >= end_index:
        raise ValueError("Invalid start or end index")

    # Create a new 'anomaly' column with default value False
    data['anomaly'] = False

    # Mark the specified range as True
    data.loc[start_index:end_index, 'anomaly'] = True

    return data
