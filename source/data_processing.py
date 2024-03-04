import pandas as pd
import numpy as np
from scipy import stats
import xarray as xr
import torch

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

def get_index_coord(lats, lons, lat, lon):
    lats1 = abs(lats - lat)
    lons1 = abs(lons - lon)
    
    idx_lat = np.argmin(lats1)
    idx_lon = np.argmin(lons1)

    min_lats = lats[idx_lat]
    min_lons = lons[idx_lon]
    
    return idx_lat,idx_lon, min_lats, min_lons

def filter_stasiun(max_stas):
    stas_chosen = {}
    for key,val in max_stas.items():
        if np.isnan(np.max(val)) or len(val) < 20 or (0.0 in val):
            continue
        else:
            stas_chosen[key] = val
    return stas_chosen

def return_period(data: np.ndarray ,T : int) -> float:
    """
    Function to calculate return periode
    Args:
        Data: vector of annual max precipitation
        T   : Return period number
    """
    fit_distribution = stats.gumbel_r.fit(data)
    return_period_value = stats.gumbel_r.ppf([1-(1/T)], *fit_distribution)
    
    return  return_period_value

def fitted_parameters_gumbel(data):
    """
    Function to calculate fitted parameters gumbel
    Args:
        data: annual max precipitation data
    
    Returns:
        fitted_distribution: a parameter from gumbel distribution

    """
    fit_distribution = stats.gumbel_r.fit(data)
    return fit_distribution

def cal_pu_gpm(data,T):
    output_pu = []
    for i in range(len(data)):
        pu = []
        for t in T:
            pu.append(return_period(data[i], t))
        output_pu.append(pu)
    return output_pu

def get_ann_max_gpm_at_station(df_stations, pu_station, ann_max_gpm, lats, lons):
    coord_gpm_at_station = {}
    ann_max_gpm_at_stasiun = {}
    for station in pu_station:
        latlon = df_stations[df_stations['Nama Stasiun'] == station][["Lintang","Bujur"]].values
        lat,lon = latlon[0][0], latlon[0][1]
        idx_lat, idx_lon, min_lat, min_lon = get_index_coord(lats, lons, lat, lon)
        coord_gpm_at_station[station] = [min_lat, min_lon]
        ann_max_gpm_at_stasiun[station] = ann_max_gpm[:,idx_lon, idx_lat]
    return ann_max_gpm_at_stasiun,coord_gpm_at_station

def seperate_input_output(dataset):
    X,y = [],[]
    for key,val in dataset.items():
        X.append(val[0])
        y.append(val[1])
    return np.array(X),np.array(y)

def pre_process_dataset1(dataset,validation_stasiun_name):
    val_dataset = {}
    train_dataset = {}
    for key, val in dataset.items():
        if key in validation_stasiun_name:
            val_dataset[key] = dataset[key]
        else:
            train_dataset[key] = dataset[key]
    X_train, y_train = seperate_input_output(train_dataset)
    X_val, y_val = seperate_input_output(val_dataset)
    return X_train, y_train, X_val, y_val

def get_ann_max_grid_gpm_at_station(df_stations, pu_station, ann_max_gpm, lats, lons, number_of_grid):
    coord_gpm_at_station = {}
    ann_max_gpm_at_stasiun = {}
    for station in pu_station:
        latlon = df_stations[df_stations['Nama Stasiun'] == station][["Lintang","Bujur"]].values
        lat,lon = latlon[0][0], latlon[0][1]
        idx_lat, idx_lon, min_lat, min_lon = get_index_coord(lats, lons, lat, lon)
        coord_gpm_at_station[station] = [min_lat, min_lon]
        idx_lon_start, idx_lon_end = idx_lon - (int(number_of_grid/2)), idx_lon + (int(number_of_grid/2)+1)
        idx_lat_start, idx_lat_end = idx_lat - (int(number_of_grid/2)), idx_lat + (int(number_of_grid/2)+1)
        ann_max_gpm_at_stasiun[station] = ann_max_gpm[:,idx_lon_start:idx_lon_end, idx_lat_start:idx_lat_end]
    return ann_max_gpm_at_stasiun,coord_gpm_at_station

def process_data1(max_gpm, max_stas, df_stas, config_class):
    """
    Function to process raw data using approach 1
    Args:
        max_gpm : xarray dataset contains annual max precipitation GPM and its entity
        max_stas: dictionary of annual max precipitation each station, key is the station name
        df_stas: dataframe contains detail information at each station
    Returns:
        X_train, y_train, X_val, y_val
    """
    lats = max_gpm['latitude'].values
    lons = max_gpm['longitude'].values
    ann_max_values = max_gpm['__xarray_dataarray_variable__'].values
    arr_ann_max_gpm = ann_max_values[1:-1,:,:]

    max_stas = filter_stasiun(max_stas)

    #calculate return period on each station
    pu_stas = {}
    T = config_class.get_T()
    for key,val in max_stas.items():
        pu = []
        for t in T:
            pu.append(return_period(data= val, T = t))
        pu_stas[key] = pu

    ann_max_gpm_at_stasiun,coord_gpm_at_station = get_ann_max_gpm_at_station(df_stas, pu_stas, arr_ann_max_gpm, lats, lons)

    dataset = {}
    for stasiun in pu_stas:
        dataset[stasiun] = (ann_max_gpm_at_stasiun[stasiun], pu_stas[stasiun])

    #get station name for validation data
    path_skema_test = config_class.get_path_skema_test()
    skema_test = config_class.get_skema_test()
    data1 = pd.read_excel(f"{path_skema_test}/Skema Testing {skema_test}.xlsx")
    stasiun_test = list(data1['Nama Stasiun'].unique())

    X_train, y_train, X_val, y_val = pre_process_dataset1(dataset= dataset, validation_stasiun_name = stasiun_test)

    return X_train, y_train, X_val, y_val
    
def process_data2(max_gpm, max_stas, df_stas, config_class):
    """
    Function to process raw data using approach 1
    Args:
        max_gpm : xarray dataset contains annual max precipitation GPM and its entity
        max_stas: dictionary of annual max precipitation each station, key is the station name
        df_stas: dataframe contains detail information at each station
    Returns:
        X_train, y_train, X_val, y_val
    """

    lats = max_gpm['latitude'].values
    lons = max_gpm['longitude'].values
    ann_max_values = max_gpm['__xarray_dataarray_variable__'].values
    arr_ann_max_gpm = ann_max_values[1:-1,:,:]

    #filter data
    max_stas = filter_stasiun(max_stas)

    #calculate return period and fitted distribution on each station
    pu_stas = {}
    fitted_parameters_stas = {}
    T = config_class.get_T()
    for key,val in max_stas.items():
        pu = []
        for t in T:
            pu.append(return_period(data= val, T = t))
        pu_stas[key] = pu
        fitted_parameters_stas[key] = fitted_parameters_gumbel(val)

    ann_max_gpm_at_stasiun,coord_gpm_at_station = get_ann_max_grid_gpm_at_station(df_stas, pu_stas, arr_ann_max_gpm, lats, lons, 3)
    
    #get station name for validation data
    path_skema_test = config_class.get_path_skema_test()
    skema_test = config_class.get_skema_test()
    data1 = pd.read_excel(f"{path_skema_test}/Skema Testing {skema_test}.xlsx")
    stasiun_test = list(data1['Nama Stasiun'].unique())

    #seperate train and val data
    output_name = config_class.get_output_name()
    X_train, y_train, X_val, y_val = [], [], [], []
    for station in pu_stas:
        if station in stasiun_test:
            X_val.append(ann_max_gpm_at_stasiun[station])
            if output_name == "Periode Ulang":
                y_val.append(pu_stas[station])
            elif output_name == "Fitted Distribution Gumbel":
                y_val.append(fitted_parameters_stas[station])
        else:
            X_train.append(ann_max_gpm_at_stasiun[station])
            if output_name == "Periode Ulang":
                y_train.append(pu_stas[station])
            elif output_name == "Fitted Distribution Gumbel":
                y_train.append(fitted_parameters_stas[station])
                
    return X_train, y_train, X_val, y_val

def process_data(max_gpm, max_stas, df_stas, config_class):
    approach = config_class.get_approach()
    if approach == 1:
        #do the processing_data1
        X_train, y_train, X_Val, y_val = process_data1(max_gpm, max_stas, df_stas, config_class)
    
    elif approach ==2:
        #do the processing_data2
        X_train, y_train, X_Val, y_val = process_data2(max_gpm, max_stas, df_stas, config_class)

    return X_train, y_train, X_Val, y_val
    