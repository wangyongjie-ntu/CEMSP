#Filename:	plaincf.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Jum 27 Mei 2022 02:05:28 

import init
import json
import pandas as pd
import numpy as np
from numpyencoder import NumpyEncoder
from util.nn_model import NNModel
from cf.plaincf import PlainCF
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
    
    # initialize the evaluator
    evaluator = Evaluator(train_x, normal_test)

    target = 0.5
    pcf = PlainCF(target, model)
    pcf1 = PlainCF(target, model1)
    _lambda = 5
    optimizer = 'adam'
    lr = 0.001

    with open("Hepatitis_cemsp.json", "r") as f:
        cfmss_results = json.load(f)
    
    num_lists = cfmss_results["num"]

    data_lists = []
    cf_lists = []
    diversity_lists = []
    diversity2_lists = []

    for i in range(len(abnormal_test)):
        input_x = abnormal_test[i:i+1]
        num_of_cf = num_lists[i]
        cf_list = []
        for j in range(num_of_cf):
            tmp_result = {}
            cf = pcf.generate_counterfactuals(input_x, _lambda, optimizer, lr)
            while model.predict(cf) != 1:
                cf = pcf.generate_counterfactuals(input_x, _lambda, optimizer, lr)
            print(str(i))
            cf_ = pcf1.generate_counterfactuals(input_x, _lambda, optimizer, lr)
            while model1.predict(cf_) != 1:
                cf_ = pcf1.generate_counterfactuals(input_x, _lambda, optimizer, lr)

            tmp_result['cf'] = cf
            tmp_result['cf2'] = cf_
            tmp_result['sparsity'] = evaluator.sparsity(input_x, cf)
            tmp_result['aps'] = evaluator.average_percentile_shift(input_x, cf)
            tmp_result['proximity'] = evaluator.proximity(cf)
            cf_list.append(tmp_result)

        cfs = [_tmp_result['cf'] for _tmp_result in cf_list]
        cfs = np.reshape(cfs, (-1, input_x.shape[1]))
        diversity = evaluator.diversity(cfs)

        _cfs = [_tmp_result['cf2'] for _tmp_result in cf_list]
        _cfs = np.reshape(_cfs, (-1, input_x.shape[1]))
        diversity2 = evaluator.diversity(_cfs)

        data_lists.append(input_x)
        cf_lists.append(cf_list)
        diversity_lists.append(diversity)
        diversity2_lists.append(diversity2)
        
    results = {}
    results["data"] = data_lists
    results["num"] = num_lists
    results["cf"] = cf_lists
    results['diversity'] = diversity_lists
    results['diversity2'] = diversity2_lists
    
    with open("Hepatitis_plaincf.json", "w") as f:
        json.dump(results, f, cls = NumpyEncoder)

