# Filename:	growingsphere.py
# Author:	Wang Yongjie
# Email:		yongjie.wang@ntu.edu.sg
# Date:		Sen 18 Apr 2022 08:21:20

import init
import json
import pandas as pd
import numpy as np
from numpyencoder import NumpyEncoder
from util.nn_model import NNModel
from cf.growingsphere import GrowingSphere
from util.evaluator import *
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":

    dataset = pd.read_csv("../../data/thyroid/thyroid_dataset.csv")
    dataset = dataset.to_numpy().astype(np.float32)
    lens = len(dataset)
    train, test = dataset[0:int(lens * 0.7), ], dataset[int(lens * 0.7):, ]
    train_x, train_y = train[:, 0:4], train[:, 4]
    test_x, test_y = test[:, 0:4], test[:, 4]

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    train_y = train_y.astype(np.int64)
    test_y = test_y.astype(np.int64)

    model = NNModel('../../train/thyroid/thyroid_data.pt')

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

    # initialize the GrowingSphere algorithm
    target = 1
    eta = 5
    gs = GrowingSphere(target, model)

    # identify number of cfs
    with open("thyroid_cemsp_sigma.json", "r") as f:
        cfmss_results = json.load(f)

    num_lists = cfmss_results["num"]

    data_lists = []
    cf_lists = []
    cf2_lists = []

    for i in range(len(abnormal_test)):
        if i in [14, 20, 23, 26, 53]:
            continue
        print(str(i) * 2)
        input_x = abnormal_test[i:i + 1]
        num_of_cf = num_lists[i]
        cf_list = []

        for j in range(num_of_cf):
            tmp_result = {}
            cf = gs.generate_counterfactual(input_x, eta)
            if len(cf.shape) == 2:
                cf = cf[0:1]  # only select first one at a time.
            else:
                cf = cf[np.newaxis, :]
            cf_list.extend(cf)
        cf_list = np.reshape(cf_list, (-1, input_x.shape[1]))
        from util.sigma_test_util import get_noised_cfs

        def generate_cf(j, input_x):
            num = cfmss_results['cf2'][i][j]['num_of_cf2']
            _cfs = []
            for _ in range(num):
                cf_ = gs.generate_counterfactual(input_x, eta)
                if len(cf_.shape) == 2:
                    cf_ = cf_[0:1]  # only select first one at a time.
                else:
                    cf_ = cf_[np.newaxis, :]
                _cfs.extend(cf_)
            _cfs = np.reshape(_cfs, (-1, input_x.shape[1]))
            return _cfs

        cf2_list = get_noised_cfs(input_x, cf_list, [0.0001, 0.001, 0.01, 0.1], evaluator, generate_cf)

        data_lists.append(input_x)
        cf_lists.append(cf_list)
        cf2_lists.append(cf2_list)

    results = {}
    results["data"] = data_lists
    results["num"] = num_lists
    results["cf"] = cf_lists
    results['cf2'] = cf2_lists

    with open("thyroid_growingsphere_sigma.json", "w") as f:
        json.dump(results, f, cls=NumpyEncoder)

