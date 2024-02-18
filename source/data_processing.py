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



def melt_df(df, id_vars, var_name, value_name):
    """
    Function to melted dataframe
    Args:
        df: dataset in form of a dataframe
        id_vars: retained columns
        var_name: column name for all the melted columns
        value_name: column name for all the melted values

    Returns:
        df_melted: df melted
    
    """
    df_melted = pd.melt(df, id_vars=id_vars, var_name=var_name, value_name=value_name)

    return df_melted
