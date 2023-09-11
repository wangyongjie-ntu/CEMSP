#Filename:	train.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Kam 26 Mei 2022 01:26:11 

import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import torch

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
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

def confusion(y_test,y_test_pred,X):
    names=['Non Hepatitis','Hepatitis']
    cm=confusion_matrix(y_test,y_test_pred)
    f,ax=plt.subplots(figsize=(10,10))
    sns.heatmap(cm,annot=True,linewidth=.5,linecolor="r",fmt=".0f",ax=ax)
    plt.title(X, size = 25)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()

    return

if __name__ == "__main__":

    data = pd.read_csv('../../data/Hepatitis/HepatitisC_dataset_processed.csv')
    standard_sc = preprocessing.StandardScaler() 

    X = data.drop(['Category'],axis=1)
    y = data["Category"]
    train_x, test_x, train_y, test_y = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

    train_x=standard_sc.fit_transform(train_x).astype(np.float32)
    test_x=standard_sc.transform(test_x).astype(np.float32)
    
    train_y = train_y.to_numpy().astype(np.int64)
    test_y = test_y.to_numpy().astype(np.int64)

    DTrain = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    DTest = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    num0 = (train_y == 0).sum()
    num1 = (train_y == 1).sum()
    weight = 1 / np.array([num0, num1])
    samples_weight = np.array([weight[int(i)] for i in train_y])
    mysampler = sampler.WeightedRandomSampler(samples_weight, num_samples = len(train_x), replacement  = True)

    train_loader = DataLoader(DTrain, sampler = mysampler, shuffle = False, batch_size = 32)
    test_loader = DataLoader(DTest, batch_size = 32)
    '''
    model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
            )
    ''' 
    model = nn.Sequential(
            nn.Linear(10, 30),
            nn.ReLU(),
            nn.Linear(30, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
            )


    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(100):
        train_loss, train_acc, train_f1 = epoch_train(model, train_loader, optimizer, criterion, device)
        if epoch % 5 == 0:
            print(f"Epoch : {epoch} | Train Loss: {train_loss:.3f} | Train acc: {train_acc:.3f} | Train F1: {train_f1:.3f}")

    test_loss, test_acc, test_f1 = epoch_val(model, test_loader, criterion, device)
    print(f"Test. Loss: {test_loss:.3f} | Test acc: {test_acc:.3f}, Test F1: {test_f1}")
    torch.save(model, "Hepatitis_model_simple_v1.pt")


