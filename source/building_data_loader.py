import torch
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X_sample = self.X[idx]
        y_sample = self.y[idx]
        
        return X_sample, y_sample
    
def create_data_loader(X: torch.tensor,
                       y: torch.tensor,
                       batch_size: int,
                       shuffle: bool):
    """
    Function to create data laoder
    Args:
        X: an array of the tropical cyclone
        y: tropical cyclone class
        batch_size: batch size 
        shuffle: suffle the data while training, make it false when creating data loader for validation
    Returns:
        data_loader: data loader 
    
    """
    custom_dataset = CustomDataset(X, y)    
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader