#Filename:	synthetic_data.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sen 04 Apr 2022 10:00:32 

import copy
import sys
import torch
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

def epoch_train(model, iterator, optimizer, criterion, device):

    epoch_loss = 0
    model.train()
    prediction = []
    label = []

    for batch_idx, (data, target) in enumerate(iterator):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        _, preds = torch.max(output, 1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(target)
        label.extend(target.tolist())
        prediction.extend(preds.reshape(-1).tolist())

    acc = accuracy_score(label, prediction)

    return epoch_loss / len(iterator.dataset), acc

def epoch_val(model, iterator, criterion, device):

    epoch_loss = 0
    model.eval()
    prediction = []
    label = []

    with torch.no_grad():
        for  batch_idx, (data, target) in enumerate(iterator):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, preds = torch.max(output, 1)
            loss = criterion(output, target)
            epoch_loss += loss.item() * len(target)
            label.extend(target.tolist())
            prediction.extend(preds.reshape(-1).tolist())

    acc = accuracy_score(label, prediction)
    return epoch_loss / len(iterator.dataset), acc

if __name__ == "__main__":

    dataset = pd.read_csv("../../data/synthetic/synthetic_data_simple.csv")
    dataset = dataset.to_numpy().astype(np.float32)
    lens = len(dataset)
    train, test = dataset[0:int(lens * 0.7), ], dataset[int(lens*0.7):, ]
    train_x, train_y = train[:, 0:4], train[:, 4]
    test_x, test_y = test[:, 0:4], test[:, 4]
    train_y = train_y.astype(np.int64)
    test_y = test_y.astype(np.int64)

    DTrain = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    DTest = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    train_loader = DataLoader(DTrain, batch_size = 32)
    test_loader = DataLoader(DTest, batch_size = 32)
    
    model = nn.Sequential(
            nn.Linear(4, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
            )
    '''
    model = nn.Sequential(
            nn.Linear(4, 25),
            nn.ReLU(),
            nn.Linear(25, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
            )
    '''    
    optimizer = optim.Adam(model.parameters(), lr = 0.02)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    model = model.to(device)
    criterion = criterion.to(device)

    best_model = None
    best_acc = -float('inf')
    

    for epoch in range(100):
        train_loss, train_acc = epoch_train(model, train_loader, optimizer, criterion, device)

        if epoch % 5 == 0:
            print(f"Epoch : {epoch} | Train Loss: {train_loss:.3f} | Train acc: {train_acc:.3f}")

    test_loss, test_acc = epoch_val(model, test_loader, criterion, device)
    print(f"Test. Loss: {test_loss:.3f} | Test acc: {test_acc:.3f}")
    #torch.save(model, "synthetic_model_simple_v1.pt")
    torch.save(model, "synthetic_model_simple.pt")

