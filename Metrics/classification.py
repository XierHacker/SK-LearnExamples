import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


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
prob=decisionTree.predict_proba(X=test_dataSet)
#print(prob)
#print(predict)
#print("\n\n\n")
#print(test_labels)

#confusion matrix
ConfuMat=confusion_matrix(y_true=test_labels,y_pred=predict)
print("Confusion Matrix:\n",ConfuMat)

#accuracy
accuracy=accuracy_score(y_true=test_labels,y_pred=predict)
print("Accuracy:\n",accuracy)

#precision
precision=precision_score(y_true=test_labels,y_pred=predict,average=None)
print("precision:\n",precision)

#recall
recall=recall_score(y_true=test_labels,y_pred=predict,average=None)
print("Recall:\n",recall)

#F1




#log loss
LogLoss=log_loss(y_true=test_labels,y_pred=prob)
print(LogLoss)