#Filename:	cemsp.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sen 04 Apr 2022 09:14:42 

import os
import torch
import numpy as np
import pandas as pd
from z3 import *
from sklearn.preprocessing import StandardScaler

class BruteForceSolver(object):

    def __init__(self, n, model_path, to_replace, desired_pred):
        self.n = n
        self.model = torch.load(model_path)
        self.model = self.model.cpu() # disable GPU
        self.model.eval() # work on eval mode
        self.to_replace = to_replace
        self.desired_pred = desired_pred

    def verify(self, input_x):
        pow_n = (2**self.n - 1)
        for i in range(1, pow_n):
            binary = f'{i:04b}'
            binary = list(binary)
            tmp = [int(j) for j in binary]
            mask = np.array(tmp).astype(np.float32)
            _input = input_x * (1 - mask) + self.to_replace * mask
            input_tensor = torch.from_numpy(_input)
            pred = self.model(input_tensor).squeeze()
            output = torch.round(pred)
            if output.item() == self.desired_pred:
                print(mask)
                #print(input_x, _input)

#Local Test on Synthetic Dataset
if __name__ == "__main__":

    dataset = pd.read_csv("../data/synthetic/synthetic_data_simple.csv")
    dataset = dataset.to_numpy().astype(np.float32)
    lens = len(dataset)
    train, val, test = dataset[0:int(lens * 0.5), ], dataset[int(lens * 0.5):int(lens * 0.75), ], dataset[int(lens*0.75):, ]
    train_x, train_y = train[:, 0:4], train[:, 4:5]
    val_x, val_y = val[:, 0:4], val[:, 4:5]
    test_x, test_y = test[:, 0:4], test[:, 4:5]
    idx = np.where(test_y == 0)[0]
    abnormal_test = test_x[idx]

    model_path = "../train/synthetic/synthetic_model_simple.pt"
    #to_replace = np.array([[1., 2., 2, 2.]]).astype(np.float32)
    to_replace = np.array([[-0.3, -0.2, -0.2, 0]]).astype(np.float32)
    desired_pred = 1
    n = 4
    bfsolver = BruteForceSolver(4, model_path, to_replace, desired_pred)
    for i in range(10,14):
        print(str(i) * 30)
        bfsolver.verify(abnormal_test[i:i+1])
