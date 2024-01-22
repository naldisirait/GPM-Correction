import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import sklearn

class MLModel:
    def __init__(self,config, model = None):
        self.config = config
        self.model = model

    def set_model(self, model):
        self.model = model
    
    def get_model(self):
        #get model name from configuration
        model_name = self.config['model_name']
        
        if model_name == "xgboost":
            # Define the XGBoost model
            model = xgb.XGBRegressor(
                objective='reg:squarederror',  # 'reg:squarederror' for regression tasks
                n_estimators=100,               # Number of boosting rounds
                max_depth=3                     # Maximum depth of each tree
            )
        else:
            model = None
        self.set_model(model)

        return self.model

