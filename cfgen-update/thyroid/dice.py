#Filename:	dice.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sel 19 Apr 2022 06:34:13 

import init
import json
import torch
import pandas as pd
import numpy as np
from numpyencoder import NumpyEncoder
from util.nn_model import NNModel
from cf.dice import DiCE
from util.evaluator import *
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":

    dataset = pd.read_csv("../../data/thyroid/thyroid_dataset.csv")
    dataset = dataset.to_numpy().astype(np.float32)
    lens = len(dataset)
    train, test = dataset[0:int(lens * 0.7), ], dataset[int(lens*0.7):, ]
    train_x, train_y = train[:, 0:4], train[:, 4:5]
    test_x, test_y = test[:, 0:4], test[:, 4:5]

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    model = NNModel('../../train/thyroid/thyroid_data.pt')
    model1 = NNModel('../../train/thyroid/thyroid_data_new.pt')

    # obtain true negative set of test set
    idx = np.where(test_y == 0)[0]
    pred_y = model.predict(test_x)
    pred_y_ = model1.predict(test_x)
    idx1 = np.where(pred_y == 0)[0]
    idx1_ = np.where(pred_y_ == 0)[0]
    tn_idx = set(idx).intersection(idx1)
    tn_idx = tn_idx.intersection(idx1_)
    abnormal_test = test_x[list(tn_idx)]

    # obtain true positive set of train set
    idx2 = np.where(train_y == 1)[0]
    pred_ty = model.predict(train_x)
    idx3 = np.where(pred_ty == 1)[0]
    tp_idx = set(idx2).intersection(idx3)
    normal_train = train_x[list(tp_idx)]

    inital_points = normal_train.mean(0)[np.newaxis, :]

    target = 0.5
    dice = DiCE(target, model)
    dice1 = DiCE(target, model1)
    # initialize the evaluator
    evaluator = Evaluator(train_x, normal_train)

    with open("thyroid_cemsp.json", "r") as f:
        cfmss_results = json.load(f)
    
    num_lists = cfmss_results["num"]
    data_lists = []
    cf_lists = []
    diversity_lists = []
    diversity2_lists = []

    for i in range(len(abnormal_test)):
        print(str(i) * 2)
        input_x = abnormal_test[i:i+1] 
        num_of_cf = num_lists[i]
        cfs = dice.generate_counterfactuals(input_x, inital_points, num_of_cf)
        cfs_ = dice1.generate_counterfactuals(input_x, inital_points, num_of_cf)

        cf_list = []
        for j in range(num_of_cf):
            tmp_result = {}
            cf = cfs[j:j+1]
            tmp_result['cf'] = cf
            tmp_result['cf2'] = cfs_[j:j+1]
            tmp_result['sparsity'] = evaluator.sparsity(input_x, cf)
            tmp_result['aps'] = evaluator.average_percentile_shift(input_x, cf)
            tmp_result['proximity'] = evaluator.proximity(cf)
            cf_list.append(tmp_result)
        
        data_lists.append(input_x)
        cf_lists.append(cf_list)
        # 增加了diversity
        diversity = evaluator.diversity(cfs)
        diversity_lists.append(diversity)
        diversity2 = evaluator.diversity(cfs_)
        diversity2_lists.append(diversity2)

    results = {}
    results["data"] = data_lists
    results["num"] = num_lists
    results["cf"] = cf_lists
    results['diversity'] = diversity_lists
    results['diversity2'] = diversity2_lists
    
    with open("thyroid_dice.json", "w") as f:
        json.dump(results, f, cls = NumpyEncoder)

