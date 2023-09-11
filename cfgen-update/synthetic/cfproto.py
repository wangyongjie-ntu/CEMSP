#Filename:	cfproto.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sel 14 Jun 2022 01:48:36 

import init
import json
import pandas as pd
import numpy as np
from numpyencoder import NumpyEncoder
from util.nn_model import NNModel
from cf.cfproto import CFProto
from util.evaluator import *

if __name__ == "__main__":

    dataset = pd.read_csv("../../data/synthetic/synthetic_data_simple.csv")
    dataset = dataset.to_numpy().astype(np.float32)
    lens = len(dataset)
    train, test = dataset[0:int(lens * 0.7), ], dataset[int(lens*0.7):, ]
    train_x, train_y = train[:, 0:4], train[:, 4]
    test_x, test_y = test[:, 0:4], test[:, 4]
    train_y = train_y.astype(np.int64)
    test_y = test_y.astype(np.int64)

    model = NNModel('../../train/synthetic/synthetic_model_simple.pt')
    model1 = NNModel('../../train/synthetic/synthetic_model_simple_v1.pt')

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

    # prototype is the same as the replaced vector in cfmss.
    prototype = np.array([[0.5, 0.4, 0., 0.5]]).astype(np.float32)
    # initialize the evaluator
    evaluator = Evaluator(train_x, normal_train)

    target = 0.5
    cfproto = CFProto(target,model)
    cfproto1 = CFProto(target, model1)

    with open("synthetic_cemsp.json", "r") as f:
        cfmss_results = json.load(f)
    
    num_lists = cfmss_results["num"]

    data_lists = []
    cf_lists = []
    diversity_lists = []
    diversity2_lists = []

    for i in range(len(abnormal_test)):
        if i > 100:
            break
        input_x = abnormal_test[i:i+1]
        prototype = np.where(input_x < prototype, prototype, input_x)
        num_of_cf = num_lists[i]
        cf_list = []
        for j in range(num_of_cf):
            tmp_result = {}
            cf = cfproto.generate_counterfactuals(input_x, prototype)
            while model.predict(cf) != 1:
                cf = cfproto.generate_counterfactuals(input_x, prototype)
            print(str(i))
            cf_ = cfproto1.generate_counterfactuals(input_x, prototype)
            while model1.predict(cf_) != 1:
                cf_ = cfproto1.generate_counterfactuals(input_x, prototype)
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
    
    with open("synthetic_cfproto1.json", "w") as f:
        json.dump(results, f, cls = NumpyEncoder)

