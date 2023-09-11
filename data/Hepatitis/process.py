#Filename:	process.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sab 04 Jun 2022 06:23:06 

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

if __name__ == "__main__":

    data = pd.read_csv("./HepatitisCdata.csv")
    data = data.drop(["Unnamed: 0", "Sex", "Age"], axis=1)
    data['Category'].loc[data['Category'].isin(["1=Hepatitis","2=Fibrosis", "3=Cirrhosis"])] = 0
    data['Category'].loc[data['Category'].isin(["0=Blood Donor", "0s=suspect Blood Donor"])] = 1
    cols = data.columns.tolist()[1:]
    for i in cols:
        data[i] = data[i].fillna(data[i].mean())

    #data.to_csv("HepatitisC_dataset_processed.csv", index = False)

