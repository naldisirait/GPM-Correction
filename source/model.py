import torch
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        # Add input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.SiLU())
        # Add hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.Dropout(0.2))
        # Add output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
def create_dl_model(config_class):
    input_size = config_class.get_input_size()
    hidden_layer = config_class.get_hidden_layer()
    output_size = config_class.get_output_size()
    model = MLP(input_size=input_size, hidden_sizes=hidden_layer, output_size=output_size)
    return model

def create_ml_model(config_class):
    pass

def create_model(config_class):
    model_name = config_class.get_model_name()
    if model_name == "Machine Learning ":
        model = create_ml_model(config_class=config_class)
    elif model_name == "Deep Learning":
        model = create_dl_model(config_class=config_class)
    return model
