import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

boston=load_boston()
dataSet=boston.data
labels=boston.target

print(dataSet.shape)
print(labels.shape)

splited=train_test_split(dataSet,labels,test_size=0.3)
print("elements in splited:\n",len(splited))
print("\n")

print("dataSet split into:")
print(splited[0].shape)
print(splited[1].shape)

print("labels split into:")
print(splited[2].shape)
print(splited[3].shape)
