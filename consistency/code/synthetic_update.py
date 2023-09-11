#Filename:	example.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sen 08 Agu 2022 03:49:23 

import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import json
from numpyencoder import NumpyEncoder
import sys
sys.path.insert(0, '../../')
from util.evaluator import *

from consistency import IterativeSearch
from consistency import PGDsL2
from consistency import StableNeighborSearch

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def create_model_from_torch1(model_path):
    torch_model = torch.load(model_path)
    torch_model.eval()
    model = Sequential()

    weight = torch_model[0].weight.detach().t().detach().numpy().astype(np.float32)
    bias = torch_model[0].bias.detach().numpy().astype(np.float32)
    layer1 = Dense(25, input_shape = (4,), activation = 'relu', weights = [weight, bias])

    weight = torch_model[2].weight.detach().t().detach().numpy().astype(np.float32)
    bias = torch_model[2].bias.detach().numpy().astype(np.float32)
    layer2 = Dense(20, input_shape = (25,), activation = 'relu', weights = [weight, bias])

    weight = torch_model[4].weight.detach().t().detach().numpy().astype(np.float32)
    bias = torch_model[4].bias.detach().numpy().astype(np.float32)
    layer3 = Dense(2, input_shape = (20,), activation = 'softmax', weights = [weight, bias])

    model.add(layer1)
    model.add(layer2)
    model.add(layer3)

    return model

def create_model_from_torch(model_path):
    torch_model = torch.load(model_path)
    torch_model.eval()
    model = Sequential()

    weight = torch_model[0].weight.detach().t().detach().numpy().astype(np.float32)
    bias = torch_model[0].bias.detach().numpy().astype(np.float32)
    layer1 = Dense(20, input_shape = (4,), activation = 'relu', weights = [weight, bias])

    weight = torch_model[2].weight.detach().t().detach().numpy().astype(np.float32)
    bias = torch_model[2].bias.detach().numpy().astype(np.float32)
    layer2 = Dense(20, input_shape = (20,), activation = 'relu', weights = [weight, bias])

    weight = torch_model[4].weight.detach().t().detach().numpy().astype(np.float32)
    bias = torch_model[4].bias.detach().numpy().astype(np.float32)
    layer3 = Dense(2, input_shape = (20,), activation = 'softmax', weights = [weight, bias])

    model.add(layer1)
    model.add(layer2)
    model.add(layer3)

    return model

if __name__ == "__main__":
    
    dataset = pd.read_csv("../../data/synthetic/synthetic_data_simple.csv")
    dataset = dataset.to_numpy().astype(np.float32)
    lens = len(dataset)
    train, test = dataset[0:int(lens * 0.7), ], dataset[int(lens*0.7):, ]
    train_x, train_y = train[:, 0:4], train[:, 4]
    test_x, test_y = test[:, 0:4], test[:, 4]
    train_y = train_y.astype(np.int64)
    test_y = test_y.astype(np.int64)

    model_path = '../../train/synthetic/synthetic_model_simple.pt'
    baseline_model = create_model_from_torch(model_path)
    model_path1 = '../../train/synthetic/synthetic_model_simple_v1.pt'
    baseline_model1 = create_model_from_torch1(model_path1)

    # obtain true positive set of test set
    idx = np.where(test_y == 0)[0]
    pred_y = np.argmax(baseline_model.predict(test_x), axis = 1)
    idx1 = np.where(pred_y == 0)[0]
    pred_y_ = np.argmax(baseline_model1.predict(test_x), axis = 1)
    idx1_ = np.where(pred_y_ == 0)[0]
    tn_idx = set(idx).intersection(idx1)
    tn_idx = set(tn_idx).intersection(idx1_)
    abnormal_test = test_x[list(tn_idx)]

    # obtain true negative set of train set
    idx2 = np.where(train_y == 1)[0]
    pred_ty = baseline_model.predict(train_x).round()
    pred_ty = np.argmax(baseline_model.predict(train_x), axis = 1)
    idx3 = np.where(pred_ty == 1)[0]
    tp_idx = set(idx2).intersection(idx3)
    normal_test = train_x[list(tp_idx)]

    # initialize the evaluator
    evaluator = Evaluator(train_x, normal_test)

    with open("../../cfgen/synthetic/synthetic_cemsp.json", "r") as f:
        cfmss_results = json.load(f)

    num_lists = cfmss_results["num"]

    data_lists = []
    cf_lists = []
    diversity_lists = []
    diversity2_lists = []

    sns_fn = StableNeighborSearch(baseline_model,
                     clamp=[train_x.min(), train_x.max()],
                     num_classes=2,
                     sns_eps=0.1,
                     sns_nb_iters=100,
                     sns_eps_iter=1.e-3,
                     n_interpolations=20)

    pgd_iter_search = PGDsL2(baseline_model,
                        clamp=[train_x.min(), train_x.max()],
                        num_classes=2,
                        eps=5,
                        nb_iters=100,
                        eps_iter=0.04,
                        sns_fn=sns_fn)

    sns_fn1 = StableNeighborSearch(baseline_model1,
                     clamp=[train_x.min(), train_x.max()],
                     num_classes=2,
                     sns_eps=0.1,
                     sns_nb_iters=100,
                     sns_eps_iter=1.e-3,
                     n_interpolations=20)

    pgd_iter_search1 = PGDsL2(baseline_model1,
                        clamp=[train_x.min(), train_x.max()],
                        num_classes=2,
                        eps=5,
                        nb_iters=100,
                        eps_iter=0.04,
                        sns_fn=sns_fn1)

    for i in range(len(abnormal_test)):
        if i > 100:
            break
        input_x = abnormal_test[i:i+1]
        num_of_cf = num_lists[i]
        cf_list = []
        for j in range(num_of_cf):
            tmp_result = {}
            flag = 0
            while True:
                try:
                    pgd_cf, pred_cf, is_valid = pgd_iter_search(input_x, num_interpolations=10, batch_size= 1)
                    if np.argmax(baseline_model.predict(pgd_cf)) == 1:
                        flag = 0
                        break
                except:
                    pass
                if flag > 3:
                    break
                flag += 1

            if flag > 0:
                break

            print(str(i))

            flag = 0
            while True:
                try:
                    pgd_cf_, pred_cf_, is_valid_ = pgd_iter_search1(input_x, num_interpolations=10, batch_size= 1)
                    if np.argmax(baseline_model1.predict(pgd_cf_)) == 1:
                        flag = 0
                        break
                except:
                    pass

                if flag > 3:
                    break
                flag += 1
            if flag > 0:
                break

            tmp_result['cf'] = pgd_cf
            tmp_result['cf2'] = pgd_cf_
            tmp_result['sparsity'] = evaluator.sparsity(input_x, pgd_cf)
            tmp_result['aps'] = evaluator.average_percentile_shift(input_x, pgd_cf_)
            tmp_result['proximity'] = evaluator.proximity(pgd_cf)
            cf_list.append(tmp_result)

        cfs = [_tmp_result['cf'] for _tmp_result in cf_list]
        cfs = np.reshape(cfs, (-1, input_x.shape[1]))
        diversity = evaluator.diversity(cfs)

        _cfs = [_tmp_result['cf2'] for _tmp_result in cf_list]
        _cfs = np.reshape(cfs, (-1, input_x.shape[1]))
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
    
    with open("synthetic_sns_update.json", "w") as f:
        json.dump(results, f, cls = NumpyEncoder)

