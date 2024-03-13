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

class RegCNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegCNN, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=16, kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=0)
        # Define the max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Define the fully connected layers
        self.fc1 = nn.Linear(32 * 1 * 1, 128)  # Adjust input size based on your input shape
        self.fc2 = nn.Linear(128, 64)  # Assuming 10 output classes
        self.fc3 = nn.Linear(64,output_size)

    def forward(self, x):
        # Forward pass through the convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        #print(x.shape)
        # Reshape the tensor for the fully connected layers
        x = x.view(-1, 32 * 1 * 1)  # Adjust the product based on your input shape
        # Forward pass through the fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        #print(x.shape)
        return x
    
def create_dl_model(config_class):
    model_dl_type = config_class.get_model_dl_type()

    input_size = config_class.get_input_size()
    hidden_layer = config_class.get_hidden_layer()
    output_size = config_class.get_output_size()
    
    if model_dl_type == "MLP":
        model = MLP(input_size=input_size, hidden_sizes=hidden_layer, output_size=output_size)
    elif model_dl_type == "CNN":
        model = RegCNN(input_size=input_size, output_size=output_size)
    else:
        print("Unknown Model, please specify specific Model you want to build.")
        return None
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
