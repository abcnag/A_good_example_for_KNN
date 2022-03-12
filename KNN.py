'''
In the name of GOD
'''
# imports
import pandas as pd
import numpy as np

# load data
df = pd.read_csv("C:/Users/M/Desktop/abbas/learning me/MakTab Khoone/Data Sets/heart.csv")
data = df[['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
       'exng', 'oldpeak', 'slp', 'caa', 'thall']]
####

# normalize data
from sklearn.preprocessing import StandardScaler as scl

data = scl().fit_transform(data)
####

# train test
from sklearn.model_selection import train_test_split

X_Train , X_Test , Y_Train , Y_Test = train_test_split(data , df['output'])
Y_Train = Y_Train.values
Y_Test = Y_Test.values

# KNN model
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=20 , weights="distance" , algorithm="auto")
'''
n_neighbors: Number of neighbors to use
Weight function used in prediction. Possible values: uniform(all poins are same) , distance(weight points by the inverse of their distance)
algorithm: auto(automatic find best algorithm) , balltree , KDtree 
'''
model.fit(X_Train , Y_Train)

# find loss
def loss(y_pre,y_test):
    correct_pre  = 0
    for i in range(len(y_pre)):
        if y_pre[i] == y_test[i]:
            correct_pre += 1
    print('model Accuracy ===>' , 100*correct_pre/len(y_pre) , "%")
####
Y_pre = model.predict(X_Test)
loss(Y_pre,Y_Test)

# BallTree model
from sklearn.neighbors import BallTree
X = X_Train
model2 = BallTree(X)
def check(inx):
    scores = []
    a = model2.query_radius(inx,r=3.2)[0]
    for i in a:
        scores.append(Y_Test[i:i+1])
    true = scores.count(1)
    false = scores.count(0)
    if true >= false:
        return 1
    else:
        return 0
Y_pre2 = []
for i in X_Test:
    Y_pre2.append(check([i]))
loss(Y_pre2,Y_Test)
