import numpy as np
import pandas as pd
import torch

from utils.utils import read_json_file
from source.data_ingest import load_dataset
from source.configuration import Configuration
from source.data_processing import process_data
from source.data_processing import cal_pu_gpm
from source.building_data_loader import create_data_loader
from source.model import create_model
from source.train_model import train_model
from source.evaluation import eval_experiment
from source.evaluation import inference_dl

def main():
    #Read configuration
    config = read_json_file("config.json")
    config_class = Configuration(config)
    T = config_class.get_T()

    gpu_idx = config_class.get_gpu_index()
    device = torch.device("cpu")
    print(f"0. Using device: {device}")

    batch_size = config_class.get_batch_size()
    # Open dataset
    ann_max_gpm, ann_max_stas, df_stasiun = load_dataset(config_class)
    print("Loading Dataset DONE!")

    #Processing Dataset
    X_train, y_train, X_val, y_val = process_data(max_gpm=ann_max_gpm, 
                                                  max_stas=ann_max_stas,
                                                  df_stas=df_stasiun,
                                                  config_class=config_class)
    
    pu_gpm = cal_pu_gpm(X_val,T)
     
    X_train = torch.tensor(X_train,dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train,dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val,dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val,dtype=torch.float32).to(device)

    print("Processing Dataset DONE!")

    #Create Data Loader
    train_loader = create_data_loader(X=X_train, y=y_train, batch_size=batch_size, shuffle= True)
    val_loader = create_data_loader(X=X_val, y=y_val, batch_size=batch_size, shuffle= False)
    
    #Create Model
    model = create_model(config_class=config_class)

    print("Creating Model.. DONE!")

    #Train Model
    model, train_loss, eval_loss = train_model(config_class= config_class, model=model, train_loader=train_loader,
                                               X_val=X_val, y_val=y_val, device=device)
    
    print("Training model.. DONE")

    pu_predicted = inference_dl(model=model, X=X_val)

    #Eval experiment
    mse_stas_gpm, mse_stas_predicted = eval_experiment(pu_gpm=pu_gpm, pu_predicted=pu_predicted, 
                                                       pu_stas=y_val, config_class=config_class)
    
    print("Evaluating Experiment.. DONE!")
    
if __name__ == "__main__":
    main()