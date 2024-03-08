import torch
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from .evaluation import eval_model

def train_model(model, config_class, train_loader, X_val, y_val, device):
    # Define loss function and optimize
    #model.to(device)d
    train_loss, eval_loss = [],[]
    learning_rate = config_class.get_learning_rate()
    epochs = config_class.get_epochs()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            #inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.view(len(targets),-1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # # L2 regularization
            # l2_reg = torch.tensor(0.)
            # for param in model.parameters():
            #     l2_reg += torch.norm(param)
            # loss += lambda_ * l2_reg
            loss.backward()
            optimizer.step()
        mse_eval,y_pred = eval_model(model=model, X=X_val, y_true=y_val, criterion=criterion)
        train_loss.append(loss.item())
        eval_loss.append(mse_eval)
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.4f}, Eval Loss: {mse_eval}')
    return model, train_loss, eval_loss