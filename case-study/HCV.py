import json

with open('Hepatitis_cemsp.json') as f:
    results = json.load(f)


data = results['data']
num = results['num']
cfs = results['cf']

def same_direction(input_x, cf, i, j):
    def sgn(x):
        return 1 if x >=0 else -1
    if sgn(input_x[i] - cf[i]) * sgn(input_x[j] - cf[j]) > 0:
        return True

for i in range(len(data)):
    num_of_cfs = num[i]
    input_x = data[i]
    cfs_list = cfs[i]
    for cf in cfs_list:
        mask = cf['mask']
        a, b = 2, 3
        if mask[a] == mask[b] and mask[a] == 1:
            if same_direction(input_x[0], cf['cf'][0], a, b):
                print(f'{i}, {num_of_cfs}')


import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

_data = pd.read_csv("../data/Hepatitis/HepatitisC_dataset_processed.csv")
standard_sc = preprocessing.StandardScaler()

X = _data.drop(['Category'],axis=1)
y = _data["Category"]
train_x, test_x, train_y, test_y = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

train_x=standard_sc.fit(train_x)


print(standard_sc.inverse_transform(data[15])[0])
for item in cfs[15]:
    mask = item['mask']
    print(mask)
    print(standard_sc.inverse_transform(item['cf'])[0])
    print('-' * 50)
