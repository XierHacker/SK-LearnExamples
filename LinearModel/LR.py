import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

#load data
iris=load_iris()
dataSet=iris.data
labels=iris.target

#print(dataSet.shape)
#print(labels.shape)

#split dataSet
train_dataSet,test_dataSet,train_labels,test_labels=train_test_split(dataSet,labels,test_size=50)
#train model
LR=LogisticRegression(n_jobs=-1)
LR.fit(X=train_dataSet,y=train_labels)


#predict
result=LR.predict(X=test_dataSet)
prob=LR.predict_proba(X=test_dataSet)

print("groud truth:\n",test_labels)
print("result:\n",result)
print("prob:\n",prob)

#accuracy
acc=accuracy_score(y_true=test_labels,y_pred=result)
print("Accuracy:",acc)

#log loss
LogLoss=log_loss(y_true=test_labels,y_pred=prob)
print("Log Loss:",LogLoss)






