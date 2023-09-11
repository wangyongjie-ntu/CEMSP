#Filename:	process.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sel 26 Apr 2022 12:58:33 

import numpy as np
import pandas as pd

if __name__ == "__main__":

    df = pd.read_csv("hypothyroid.csv")
    #cols = ['FTI','TSH','T3','TT4','T4U']
    cols = ['FTI','TSH','T3','TT4']
    df=df.replace({"?":np.NAN})
    for i in cols:
        df[i] = pd.to_numeric(df[i])

    cols.append('binaryClass')
    df = df[cols]
    df = df.dropna()
    df = df.reset_index(drop = True)
    '''
    for i in cols:
        df[i] = df[i].fillna(df[i].mean())
    '''
    # Negative: thyroid patients
    # df["binaryClass"]=df["binaryClass"].map({"P":0,"N":1})
    df["binaryClass"]=df["binaryClass"].map({"P":1,"N":0})
    print(df.head(10))
    
    #df.to_csv("thyroid_dataset.csv", index = False)

