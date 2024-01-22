import pandas as pd
import numpy as np

def process_data(df: pd.DataFrame, config):
    """
    Function to proces data

    Args:
        df: a dataframe of the dataset

    Returns:
        X: independent variable
        y: target variable
    """
    features = config.get_features_name()
    target = config.get_target_name()

    X, y = df[features], y[target]
    
    return X,y

