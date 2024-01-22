import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

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

    