def min_max_normalize(df):
    """
    Perform min-max normalization on all numeric columns in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Normalized DataFrame.
    """
    # Create a copy to avoid modifying the original data
    normalized_df = df.copy()

    # Normalize each numeric column
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:  # Normalize only numeric columns
            x_min = df[column].min()
            x_max = df[column].max()
            if x_max != x_min:  # Avoid division by zero
                normalized_df[column] = (df[column] - x_min) / (x_max - x_min)
            else:
                normalized_df[column] = 0  # If all values are the same, set to 0
    return normalized_df


def min_max_normalize_skip_first_column(df):
    """
    Perform min-max normalization on all numeric columns except the first one
    (e.g., time column).

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Normalized DataFrame with the first column unchanged.
    """
    # Create a copy to avoid modifying the original data
    normalized_df = df.copy()

    # Skip the first column (assumed to be time)
    for column in df.columns[1:]:  # Start from the second column
        if df[column].dtype in ['float64', 'int64']:  # Normalize only numeric columns
            x_min = df[column].min()
            x_max = df[column].max()
            if x_max != x_min:  # Avoid division by zero
                normalized_df[column] = (df[column] - x_min) / (x_max - x_min)
            else:
                normalized_df[column] = 0  # If all values are the same, set to 0
    return normalized_df
