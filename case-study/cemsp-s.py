#Filename:	../test/cemsp.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Kam 28 Apr 2022 10:05:00 

import pandas as pd
import numpy as np
from numpyencoder import NumpyEncoder
import torch
import init
import json
from util.dt_model import DTModel
from util.nn_model import NNModel
from cemsp import *
from util.evaluator import *
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":

    dataset = pd.read_csv("../data/synthetic/synthetic_data_simple.csv")
    dataset = dataset.to_numpy().astype(np.float32)
    lens = len(dataset)
    train, test = dataset[0:int(lens * 0.7), ], dataset[int(lens*0.7):, ]
    train_x, train_y = train[:, 0:4], train[:, 4]
    test_x, test_y = test[:, 0:4], test[:, 4]
    train_y = train_y.astype(np.int64)
    test_y = test_y.astype(np.int64)

    #model = DTModel('../train/synthetic/synthetic.pickle')
    model = NNModel('../train/synthetic/synthetic_model_simple.pt')

    # obtain true negative set of test set
    idx = np.where(test_y == 0)[0]
    pred_y = model.predict(test_x)
    idx1 = np.where(pred_y == 0)[0]
    tn_idx = set(idx).intersection(idx1)
    abnormal_test = test_x[list(tn_idx)]

    # obtain true positive set of train set
    idx2 = np.where(train_y == 1)[0]
    pred_ty = model.predict(train_x)
    idx3 = np.where(pred_ty == 1)[0]
    tp_idx = set(idx2).intersection(idx3)
    normal_train = train_x[list(tp_idx)]
    
    # set the value to replace
    normal_range = np.array([[0.5, 0.4, 0., 0.5]]).astype(np.float32)
    # initialize the evaluator
    evaluator = Evaluator(train_x, normal_train)

    desired_pred = 1
    n = 4
    i = 1
    input_x = abnormal_test[i:i+1]
    to_replace = np.where(input_x < normal_range, normal_range, input_x)
    mapsolver = MapSolver(n)
    cfsolver = CFSolver(n, model, input_x, to_replace, desired_pred = 1)
    num_of_cf = 0
    cf_list = []
    for text, cf, mask in FindCF(cfsolver, mapsolver):
        tmp_result = {}
        num_of_cf += 1
        tmp_result['input'] = input_x
        tmp_result['cf'] = cf
        tmp_result['mask'] = mask
        cf_list.append(tmp_result)

    print(cf_list)
