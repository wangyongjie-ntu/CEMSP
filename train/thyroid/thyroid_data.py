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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import shuffle
import torch.optim as optim
from torchsampler import ImbalancedDatasetSampler
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data.sampler as sampler

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
    f1 = f1_score(label, prediction)

    return epoch_loss / len(iterator.dataset), acc, f1

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
    f1 = f1_score(label, prediction)

    return epoch_loss / len(iterator.dataset), acc, f1

if __name__ == "__main__":

    dataset = pd.read_csv("../../data/thyroid/thyroid_dataset.csv")
    dataset = dataset.to_numpy().astype(np.float32)
    lens = len(dataset)
    train, test = dataset[0:int(lens * 0.7), ], dataset[int(lens*0.7):, ]
    train_x, train_y = train[:, 0:4], train[:, 4]
    test_x, test_y = test[:, 0:4], test[:, 4]

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    train_y = train_y.astype(np.int64)
    test_y = test_y.astype(np.int64)

    DTrain = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    DTest = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    
    num0 = (dataset[:, -1] == 0).sum()
    num1 = (dataset[:, -1] == 1).sum()
    weight = 1 / np.array([num0, num1])
    samples_weight = np.array([weight[int(i)] for i in train_y])
    mysampler = sampler.WeightedRandomSampler(samples_weight, num_samples = len(train), replacement  = True)
    train_loader = DataLoader(DTrain, sampler = mysampler, shuffle = False, batch_size = 32)
    test_loader = DataLoader(DTest, batch_size = 32)

    # Old the model
    '''
    model = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            )

    # new model
    '''
    model = nn.Sequential(
            nn.Linear(4, 20),
            nn.ReLU(),
            nn.Linear(20, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            )
    optimizer = optim.Adam(model.parameters(), lr = 0.003)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(200):
        train_loss, train_acc, train_f1 = epoch_train(model, train_loader, optimizer, criterion, device)
        if epoch % 5 == 0:
            print(f"Epoch : {epoch} | Train Loss: {train_loss:.3f} | Train acc: {train_acc:.3f} | Train F1: {train_f1:.3f}")

    test_loss, test_acc, test_f1 = epoch_val(model, test_loader, criterion, device)
    print(f"Test. Loss: {test_loss:.3f} | Test acc: {test_acc:.3f} | Test F1: {test_f1:.3f}")
    torch.save(model, "thyroid_data_new.pt")
    #torch.save(model, "thyroid_data.pt")

