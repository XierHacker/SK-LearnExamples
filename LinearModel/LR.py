import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

#load data
iris=load_iris()
dataSet=iris.data
labels=iris.target

#print(dataSet.shape)
#print(labels.shape)

#
LR=LogisticRegression(n_jobs=-1)


