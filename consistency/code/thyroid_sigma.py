# Filename:	example.py
# Author:	Wang Yongjie
# Email:		yongjie.wang@ntu.edu.sg
# Date:		Sen 08 Agu 2022 03:49:23

import json
import sys

import pandas as pd
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from numpyencoder import NumpyEncoder
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '../../../')
from util.evaluator import *

from consistency import PGDsL2
from consistency import StableNeighborSearch

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def dnn(input_shape, n_classes=2):
    x = tf.keras.Input(input_shape)
    y = tf.keras.layers.Dense(128)(x)
    y = tf.keras.layers.Activation('relu')(y)
    y = tf.keras.layers.Dense(128)(y)
    y = tf.keras.layers.Activation('relu')(y)
    y = tf.keras.layers.Dense(n_classes)(y)
    y = tf.keras.layers.Activation('softmax')(y)
    return tf.keras.models.Model(x, y)


def train_dnn(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
    model.evaluate(X_test, y_test, batch_size=batch_size)
    return model


def create_model_from_torch(model_path):
    torch_model = torch.load(model_path)
    torch_model.eval()
    model = Sequential()

    weight = torch_model[0].weight.detach().t().detach().numpy().astype(np.float32)
    bias = torch_model[0].bias.detach().numpy().astype(np.float32)
    layer1 = Dense(32, input_shape=(4,), activation='relu', weights=[weight, bias])

    weight = torch_model[2].weight.detach().t().detach().numpy().astype(np.float32)
    bias = torch_model[2].bias.detach().numpy().astype(np.float32)
    layer2 = Dense(16, input_shape=(32,), activation='relu', weights=[weight, bias])

    weight = torch_model[4].weight.detach().t().detach().numpy().astype(np.float32)
    bias = torch_model[4].bias.detach().numpy().astype(np.float32)
    layer3 = Dense(2, input_shape=(16,), activation='softmax', weights=[weight, bias])

    model.add(layer1)
    model.add(layer2)
    model.add(layer3)

    return model


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

    model_path = '../../train/thyroid/thyroid_data.pt'
    baseline_model = create_model_from_torch(model_path)
    # baseline_model = dnn(train_x.shape[1:], n_classes= 2)
    # baseline_model = train_dnn(baseline_model, train_x, train_y, test_x, test_y, batch_size=32)

    # obtain true positive set of test set
    idx = np.where(test_y == 0)[0]
    pred_y = np.argmax(baseline_model.predict(test_x), axis=1)
    idx1 = np.where(pred_y == 0)[0]
    tn_idx = set(idx).intersection(idx1)
    abnormal_test = test_x[list(tn_idx)]

    # obtain true negative set of train set
    idx2 = np.where(train_y == 1)[0]
    pred_ty = baseline_model.predict(train_x).round()
    pred_ty = np.argmax(baseline_model.predict(train_x), axis=1)
    idx3 = np.where(pred_ty == 1)[0]
    tp_idx = set(idx2).intersection(idx3)
    normal_test = train_x[list(tp_idx)]

    # initialize the evaluator
    evaluator = Evaluator(train_x, normal_test)

    with open("../../cfgen/thyroid/thyroid_cemsp_sigma.json", "r") as f:
        cfmss_results = json.load(f)

    num_lists = cfmss_results["num"]

    data_lists = []
    cf_lists = []
    cf2_lists = []

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

    for i in range(len(abnormal_test)):
        input_x = abnormal_test[i:i + 1]
        num_of_cf = num_lists[i]
        cf_list = []
        for j in range(num_of_cf):
            tmp_result = {}
            flag = 0
            while True:
                try:
                    pgd_cf, pred_cf, is_valid = pgd_iter_search(input_x, num_interpolations=10, batch_size=1)
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
            cf_list.extend(pgd_cf)
        cf_list = np.reshape(cf_list, (-1, input_x.shape[1]))

        from util.sigma_test_util import get_noised_cfs

        def generate_cf(j, input_x):
            num = cfmss_results['cf2'][i][j]['num_of_cf2']
            _cfs = []
            for _ in range(num):
                flag = 0
                while True:
                    try:
                        pgd_cf_, pred_cf_, is_valid_ = pgd_iter_search(input_x, num_interpolations=10, batch_size=1)
                        if np.argmax(baseline_model.predict(pgd_cf_)) == 1:
                            flag = 0
                            break
                    except:
                        pass
                    if flag > 3:
                        break
                    flag += 1
                if flag > 0:
                    break
                _cfs.extend(pgd_cf_)
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

    with open("thyroid_sns_sigma.json", "w") as f:
        json.dump(results, f, cls=NumpyEncoder)

