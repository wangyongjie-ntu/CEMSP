# Filename:	cfproto.py
# Author:	Wang Yongjie
# Email:		yongjie.wang@ntu.edu.sg
# Date:		Sel 14 Jun 2022 01:48:36

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
    train, test = dataset[0:int(lens * 0.7), ], dataset[int(lens * 0.7):, ]
    train_x, train_y = train[:, 0:4], train[:, 4]
    test_x, test_y = test[:, 0:4], test[:, 4]
    train_y = train_y.astype(np.int64)
    test_y = test_y.astype(np.int64)

    model = NNModel('../../train/synthetic/synthetic_model_simple.pt')

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
    cfproto = CFProto(target, model)

    with open("synthetic_cemsp_sigma.json", "r") as f:
        cfmss_results = json.load(f)

    num_lists = cfmss_results["num"]

    data_lists = []
    cf_lists = []
    cf2_lists = []
    diversity_lists = []
    diversity2_lists = []

    for i in range(len(abnormal_test)):
        if i > 100:
            break
        input_x = abnormal_test[i:i + 1]
        prototype = np.where(input_x < prototype, prototype, input_x)
        num_of_cf = num_lists[i]
        cf_list = []
        for j in range(num_of_cf):
            # tmp_result = {}
            cf = cfproto.generate_counterfactuals(input_x, prototype)
            while model.predict(cf) != 1:
                cf = cfproto.generate_counterfactuals(input_x, prototype)
            print(str(i))
            cf_list.extend(cf)

        cf_list = np.reshape(cf_list, (-1, input_x.shape[1]))

        # 修改此处，改为添加高斯噪声
        from util.sigma_test_util import get_noised_cfs


        def generate_cfs_func(j, input_x):
            _cfs = []
            # result中cf2下第i个样本的第j次扰动产生的解释的num
            num = cfmss_results['cf2'][i][j]['num_of_cf2']
            print()
            for _ in range(num):
                cf_ = cfproto.generate_counterfactuals(input_x, prototype)
                while model.predict(cf_) != 1:
                    cf_ = cfproto.generate_counterfactuals(input_x, prototype)
                _cfs.extend(cf_)
            _cfs = np.reshape(_cfs, (-1, input_x.shape[1]))
            return _cfs


        cf2_list = get_noised_cfs(input_x, cf_list, [0.0001, 0.001, 0.01, 0.1], evaluator, generate_cfs_func)

        data_lists.append(input_x)
        cf_lists.append(cf_list)
        cf2_lists.append(cf2_list)

    results = {}
    results["data"] = data_lists
    results["num"] = num_lists
    results["cf"] = cf_lists
    results['cf2'] = cf2_lists

    with open("synthetic_cfproto_sigma.json", "w") as f:
        json.dump(results, f, cls=NumpyEncoder)

