import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
import data
import merge


print("-----------------------Load Data--------------------------")
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
best_classfier=grid.best_estimator_
for train_index,test_index in indices:
    best_classfier.fit(X=trainSet[train_index],y=train_labels[train_index])

    #predict prob
    prob=best_classfier.predict_proba(X=trainSet[test_index])


    #log loss
    loss=log_loss(y_true=train_labels[test_index],y_pred=prob)
    print("-------In Set ",i,": ------")
    print("log loss:",loss)
    i+=1