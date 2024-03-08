import numpy as np
import pandas as pd
import torch

from sklearn.metrics import mean_squared_error

def inference_dl(model,X):
    y_pred = model(X)
    return y_pred

def eval_model(model, X, y_true, criterion):
    model.eval()
    with torch.no_grad():
        y_pred_tensor = inference_dl(model,X)
        y_true = y_true.view(len(y_true),-1)
        #print(f"pred shape: {y_pred_tensor.shape}, True shape: {y_true.shape}")
        mse = criterion(y_pred_tensor, y_true)
    model.train()
    return mse,y_pred_tensor

def cal_mse(data1, data2):
    mse = mean_squared_error(data1, data2)
    return mse

def eval_experiment(pu_stas, pu_gpm, pu_predicted, config_class):
    """
    Function to evaluate experiment

    Args:
        pu_stas: return period of stations
        pu_gpm: return period of gpm
        pu_predicted: return period of predicted value
    Returns:
        mse_stas_gpm: mse of each return period between station and raw gpm
        mse_stas_predicted: mse of each return periode between station and predicted
    
    """
    pu_stas = np.array(pu_stas)
    s1,s2,s3 = pu_stas.shape
    pu_stas = np.reshape(pu_stas, (s1,s2))
    
    pu_gpm = np.array(pu_gpm)
    s1,s2,s3 = pu_gpm.shape
    pu_gpm = np.reshape(pu_gpm, (s1,s2))
    
    pu_predicted = pu_predicted.detach().numpy()

    #print(pu_stas.shape, pu_gpm.shape, pu_predicted.shape)
    output_size = config_class.get_output_size()
    mse_stas_gpm  = []
    mse_stas_predicted = []
    for i in range(output_size):
        mse_stas_gpm.append(cal_mse(data1=pu_stas[:,i], data2=pu_gpm[:,i]))
        mse_stas_predicted.append(cal_mse(data1=pu_stas[:,i], data2=pu_predicted[:,i]))
    return mse_stas_gpm, mse_stas_predicted
    
class Evaluation:
    def __init__(self, model):
        self.model = model

    def calculate_mse(self, y_pred, y_true):
        mse = mean_squared_error(y_pred=y_pred, y_true= y_true)
        return mse

    def calculate_rmse(self, y_pred, y_true):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        return rmse
    
    def calculate_nmse_mean(self, y_pred, y_true):
        rmse = self.calculate_rmse(y_pred=y_pred , y_true=y_true)
        nmse_mean = rmse/np.mean(y_true)
        return nmse_mean
    
    def calculate_nmse_std(self, y_pred, y_true):
        rmse = self.calculate_rmse(y_pred=y_pred , y_true=y_true)
        nmse_std = rmse/np.std(y_true)
        return nmse_std
    
    def calculate_nmse_min_max(self, y_pred, y_true):
        rmse = self.calculate_rmse(y_pred=y_pred , y_true=y_true)
        nmse_min_max = rmse/(np.max(y_true) - np.min(y_true))
        return nmse_min_max

    