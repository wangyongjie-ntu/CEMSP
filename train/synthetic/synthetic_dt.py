#Filename:	synthetic_data.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sen 04 Apr 2022 10:00:32 

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, confusion_matrix, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

if __name__ == "__main__":

    dataset = pd.read_csv("../../data/synthetic/synthetic_data_simple.csv")
    dataset = dataset.to_numpy().astype(np.float32)
    lens = len(dataset)
    train, test = dataset[0:int(lens * 0.7), ], dataset[int(lens*0.7):, ]
    train_x, train_y = train[:, 0:4], train[:, 4]
    test_x, test_y = test[:, 0:4], test[:, 4]

    #model = DecisionTreeClassifier(random_state = 0)
    model = RandomForestClassifier(n_estimators = 200)
    model.fit(train_x, train_y)
    y_pred = model.predict(test_x)
    
    print(confusion_matrix(y_pred, test_y))
    pr, rc, fs, sup = precision_recall_fscore_support(test_y, y_pred, average='macro')
    res = {"Accuracy": round(accuracy_score(test_y, y_pred), 4),
                              "Precision": round(pr, 4), "Recall":round(rc, 4), "FScore":round(fs, 4)}
    print(res)
    filename = "synthetic.pickle"
    pickle.dump(model, open(filename, 'wb'))
