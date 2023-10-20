import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def detect_outliers_using_moving_avg(data, features, window_size, threshold):
    """
    Detect outliers in a DataFrame using moving average across multiple features.

    Parameters:
    - data: DataFrame containing the data
    - features: List of columns (features) to analyze
    - window_size: Size of the moving average window
    - threshold: Multiplier for standard deviation to determine bounds

    Returns:
    - DataFrame with a single column "is_outlier_ma" indicating if a row is an outlier.
    """
    
    
    # Create a placeholder column for outliers, initially setting everything to False (not an outlier)
    data["is_outlier_ma"] = False

    for feature in features:
        moving_avg = data[feature].rolling(window=window_size).mean()
        moving_std = data[feature].rolling(window=window_size).std()

        upper_bound = moving_avg + threshold * moving_std
        lower_bound = moving_avg - threshold * moving_std

        is_outlier = (data[feature] > upper_bound) | (data[feature] < lower_bound)

        # Update the outlier placeholder column with OR operation to keep track of outliers from any feature
        data["is_outlier_ma"] = data["is_outlier_ma"] | is_outlier

    return data["is_outlier_ma"]

