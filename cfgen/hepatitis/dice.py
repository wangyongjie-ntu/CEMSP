# Filename:	dice.py
# Author:	Wang Yongjie
# Email:		yongjie.wang@ntu.edu.sg
# Date:		Sel 19 Apr 2022 06:34:13

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
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    data = pd.read_csv("../../data/Hepatitis/HepatitisC_dataset_processed.csv")
    standard_sc = preprocessing.StandardScaler()

    X = data.drop(['Category'], axis=1)
    y = data["Category"]
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    train_x = standard_sc.fit_transform(train_x).astype(np.float32)
    test_x = standard_sc.transform(test_x).astype(np.float32)

    train_y = train_y.to_numpy().astype(np.int64)
    test_y = test_y.to_numpy().astype(np.int64)

    model = NNModel("../../train/hepatitis/Hepatitis_model_simple.pt")

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

    inital_points = normal_train.mean(0)[np.newaxis, :]

    target = 0.5
    dice = DiCE(target, model)
    # initialize the evaluator
    evaluator = Evaluator(train_x, normal_train)

    with open("Hepatitis_cemsp_sigma.json", "r") as f:
        cfmss_results = json.load(f)

    num_lists = cfmss_results["num"]
    data_lists = []
    cf_lists = []
    cf2_lists = []

    for i in range(len(abnormal_test)):
        input_x = abnormal_test[i:i + 1]
        num_of_cf = num_lists[i]
        cfs = dice.generate_counterfactuals(input_x, inital_points, num_of_cf)
        print(str(i), str(model.predict(cfs)))
        cf_list = cfs

        from util.sigma_test_util import get_noised_cfs


        def generate_cf(j, input_x):
            num = cfmss_results['cf2'][i][j]['num_of_cf2']
            cfs_ = dice.generate_counterfactuals(input_x, inital_points, num)
            return cfs_


        cf2_list = get_noised_cfs(input_x, cf_list, [0.0001, 0.001, 0.01, 0.1], evaluator, generate_cf)
        data_lists.append(input_x)
        cf_lists.append(cf_list)
        cf2_lists.append(cf2_list)

    results = {}
    results["data"] = data_lists
    results["num"] = num_lists
    results["cf"] = cf_lists
    results['cf2'] = cf2_lists

    with open("Hepatitis_dice_sigma.json", "w") as f:
        json.dump(results, f, cls=NumpyEncoder)

