import numpy as np
import pandas as pd

def load_dataset(path: str) -> pd.DataFrame:
    """
    Function to open dataset
    Args:
        path: path to dataset
    Returns:
        df : dataset in form of DataFrame
    """
    df = pd.read_excel(path)
    return df
