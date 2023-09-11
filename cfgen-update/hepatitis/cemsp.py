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
from util.nn_model import NNModel
from cf.cemsp import *
from util.evaluator import *
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    data = pd.read_csv("../../data/Hepatitis/HepatitisC_dataset_processed.csv")
    standard_sc = preprocessing.StandardScaler()

    X = data.drop(['Category'],axis=1)
    y = data["Category"]
    train_x, test_x, train_y, test_y = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

    train_x=standard_sc.fit_transform(train_x).astype(np.float32)
    test_x=standard_sc.transform(test_x).astype(np.float32)

    train_y = train_y.to_numpy().astype(np.int64)
    test_y = test_y.to_numpy().astype(np.int64)

    model = NNModel("../../train/hepatitis/Hepatitis_model_simple.pt")
    model1 = NNModel("../../train/hepatitis/Hepatitis_model_simple_v1.pt")

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
    normal_test = train_x[list(tp_idx)]

    # set the normal range
    normal_range = np.array([[35.6, 30, 10, 10, 0, 5.32, 5.368, 59, 10, 66],
        [46, 120, 35, 35, 21, 12.92, 5.368, 84, 42, 87]])
    normal_range = standard_sc.transform(normal_range).astype(np.float32)
    normal_range = normal_range * 0.3
    
    # initialize the evaluator
    evaluator = Evaluator(train_x, normal_test)

    desired_pred = 1
    n = 10
    data_lists = []
    num_lists = []
    cf_lists = []
    cf2_lists = []
    diversity_lists = []
    diversity2_lists = []


    for i in range(len(abnormal_test)):
        print(str(i) * 2)
        input_x = abnormal_test[i:i+1]
        to_replace = np.where(input_x < normal_range[0, :], normal_range[0,:], input_x)
        to_replace = np.where(to_replace > normal_range[1, :], normal_range[1,:], to_replace)
        mapsolver = MapSolver(n)
        cfsolver = CFSolver(n, model, input_x, to_replace, desired_pred = 1)
        num_of_cf = 0
        cf_list = []
        for text, cf, mask in FindCF(cfsolver, mapsolver):
            tmp_result = {}
            num_of_cf += 1
            tmp_result['cf'] = cf
            tmp_result['mask'] = mask
            tmp_result['sparsity'] = evaluator.sparsity(input_x, cf)
            tmp_result['aps'] = evaluator.average_percentile_shift(input_x, cf)
            tmp_result['proximity'] = evaluator.proximity(cf)
            cf_list.append(tmp_result)

        # 修改此处，增加diversity的计算
        cfs = [_tmp_result['cf'] for _tmp_result in cf_list]
        cfs = np.reshape(cfs, (-1, input_x.shape[1]))
        print('cfs', cfs.shape)
        diversity = evaluator.diversity(cfs)

        # random select a cf as target to move
        mapsolver1 = MapSolver(n)
        cfsolver1 = CFSolver(n, model1, input_x, to_replace, desired_pred = 1)
        cf2_list = []
        for text, cf, mask in FindCF(cfsolver1, mapsolver1):
            cf2_list.extend(cf)

        _cfs = np.reshape(cf2_list, (-1, input_x.shape[1]))
        print('_cfs', _cfs.shape)
        diversity2 = evaluator.diversity(_cfs)

        data_lists.append(input_x)
        num_lists.append(num_of_cf)
        cf_lists.append(cf_list)
        cf2_lists.append(cf2_list)
        diversity_lists.append(diversity)
        diversity2_lists.append(diversity2)

    results = {}
    results["data"] = data_lists
    results["num"] = num_lists
    results["cf"] = cf_lists
    results["cf2"] = cf2_lists
    results['diversity'] = diversity_lists
    results['diversity2'] = diversity2_lists
    
    with open("Hepatitis_cemsp.json", "w") as f:
        json.dump(results, f, cls = NumpyEncoder)

