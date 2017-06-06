import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
from mlp import MLP



print("-----------------------Load Data--------------------------")
train_frame=pd.read_csv("train_merged.csv")
print("shape of train_frame:",train_frame.shape)

'''
positive_train_frame=train_frame[train_frame["label"]>0]
print("shape of positive_train_frame:",positive_train_frame.shape)

sub_train_frame1=train_frame.ix[:200000]
print("shape of sub_train_frame1:",sub_train_frame1.shape)

sub_train_frame=pd.concat(objs=[sub_train_frame1,positive_train_frame],axis=0)
print("shape of sub_train_frame:",sub_train_frame.shape)

'''

train_labels_frame=train_frame.pop(item="label")

trainSet=train_frame.values
train_labels=train_labels_frame.values

#load model
model=MLP(100,50)
model.fit(X=trainSet,y=train_labels)
