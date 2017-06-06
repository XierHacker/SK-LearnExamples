import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss




print("-----------------------Load Data--------------------------")
train_frame=pd.read_csv("train_merged.csv")
print("shape of train_frame:",train_frame.shape)

positive_train_frame=train_frame[train_frame["label"]>0]
print("shape of positive_train_frame:",positive_train_frame.shape)

sub_train_frame1=train_frame.ix[:200000]
print("shape of sub_train_frame1:",sub_train_frame1.shape)

sub_train_frame=pd.concat(objs=[sub_train_frame1,positive_train_frame],axis=0)
print("shape of sub_train_frame:",sub_train_frame.shape)

train_labels_frame=sub_train_frame.pop(item="label")

trainSet=sub_train_frame.values
train_labels=train_labels_frame.values


#split dataset
sp=ShuffleSplit(n_splits=5,test_size=0.2)
indices=sp.split(X=trainSet,y=train_labels)

print("-------------------------model selection----------------------")
#model selection
param_dict={
            "learning_rate":[0.05,0.1,0.15,0.2,0.25],
            "n_estimators":[60,80,100,120,140,160]
        }
gbdt=GradientBoostingClassifier()

grid=GridSearchCV(estimator=gbdt,param_grid=param_dict,
                  scoring="neg_log_loss",cv=5,n_jobs=4)


grid.fit(X=trainSet,y=train_labels)

print("best score:",grid.best_score_)
print("best index:",grid.best_index_)
print("best params:",grid.best_params_)


print("-------------------------prediction-----------------------------")
for train_index,test_index in indices:
    #predict prob
    prob=grid.predict_proba(X=trainSet[test_index])
    #log loss
    loss=log_loss(y_true=train_labels[test_index],y_pred=prob)
    print("-------In Set ",i,": ------")
    print("log loss:",loss)
    i+=1