# Import required libraries
import numpy as np
import pandas as pd
from minepy import MINE

def compute_mic(x, y, alpha=0.6, c=15):
    """
    Compute the Maximal Information Coefficient (MIC) between two variables.

    Parameters:
        x (array-like): Data for the first variable.
        y (array-like): Data for the second variable.
        alpha (float): Alpha parameter for MINE, controls threshold sensitivity.
        c (int): Complexity parameter for MINE, controls grid resolution.

    Returns:
        float: MIC value between x and y.
    """
    mine = MINE(alpha=alpha, c=c)
    mine.compute_score(x, y)
    return mine.mic()

def select_significant_parameters(data, target_column, threshold):
    """
    Select features significantly correlated with the target column
    based on Maximal Information Coefficient (MIC).

    Parameters:
        data (pd.DataFrame): Input dataset (excluding timestamp).
        target_column (str): Name of the target column.
        threshold (float): Empirical threshold to select significant features.

    Returns:
        list: List of parameters with MIC >= threshold.
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column {target_column} not found in the dataset.")

    y = data[target_column].values

    # Exclude target column and convert remaining data to matrix
    X = data.drop(columns=[target_column])
    X_mat = X.values
    columns = X.columns

    # Define a helper function to compute MIC between each feature and the target
    def compute_mic_for_column(x):
        mine = MINE(alpha=0.6, c=15)
        mine.compute_score(x, y)
        return mine.mic()

    # Compute MIC values for all features using vectorized computation
    mic_values = np.apply_along_axis(compute_mic_for_column, axis=0, arr=X_mat)
    mic_df = pd.DataFrame({'Parameter': columns, 'MIC': mic_values})

    # Filter parameters based on the threshold
    significant_parameters = mic_df[mic_df['MIC'] >= threshold]['Parameter'].tolist()

    return significant_parameters
