import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import log_loss
import data
import merge

#从data中得到train,test,和labels
train_frame,train_labels_frame,test_frame=data.loadFrame()

#test
merged_frame=merge.preprocess(dataSet_frame=train_frame)
#print(merged_frame.shape)
#print(merged_frame.head(10))

trainSet=merged_frame.values
train_labels=train_labels_frame.values

#split dataset
sp=ShuffleSplit(n_splits=5,test_size=0.2)
indices=sp.split(X=trainSet,y=train_labels)

#model
LR=LogisticRegression()
i=1
for train_index,test_index in indices:
    LR.fit(X=trainSet[train_index],y=train_labels[train_index])

    #predict prob
    prob=LR.predict_proba(X=trainSet[test_index])


    #log loss
    loss=log_loss(y_true=train_labels[test_index],y_pred=prob)
    print("-------in set ",i,": ------")
    print("log loss:",loss)