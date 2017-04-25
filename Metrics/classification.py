import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#load data
iris=load_iris()
dataSet=iris.data
labels=iris.target
#print(dataSet.shape)
#print(labels.shape)

#split dataSet to train part and test part
train_dataSet,test_dataSet,train_labels,test_labels=train_test_split(dataSet,labels,test_size=50)
#print(train_dataSet.shape)
#print(test_dataSet.shape)
#print(train_labels.shape)
#print(test_labels.shape)

decisionTree=DecisionTreeClassifier()
decisionTree.fit(X=train_dataSet,y=train_labels)

predict=decisionTree.predict(X=test_dataSet)
#print(predict)
#print("\n\n\n")
#print(test_labels)

#accuracy
accuracy=accuracy_score(y_true=test_labels,y_pred=predict)
print(accuracy)