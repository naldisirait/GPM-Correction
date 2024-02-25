import pandas as pd
import xarray as xr
import pickle

def read_pickle(filename):
    with open(filename, 'rb') as file:
        # Use pickle.load() to deserialize and load the data
        loaded_data = pickle.load(file)
    return loaded_data

def load_dataset(config_class):
    """
    Function to open dataset
    Args:
        config: Configuration of the experiment
    returns:
        ann_max_gpm: annual maximum precipitation of gpm
        ann_max_stas: annual maximum precipitation of station observation
        df_stas: detail information about the station
    """
    path_df = config_class.get_path_df()
    path_ann_max_gpm = config_class.get_path_ann_max_gpm()
    path_ann_max_stas = config_class.get_path_ann_max_stas()

    df = pd.read_excel(path_df)
    max_gpm = xr.open_dataset(path_ann_max_gpm)
    max_stas = read_pickle(path_ann_max_stas)

    return max_gpm, max_stas, df
