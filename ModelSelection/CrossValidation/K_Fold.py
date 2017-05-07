from sklearn.model_selection import KFold
from sklearn.datasets import load_boston

#load data
boston=load_boston()
dataSet=boston.data
labels=boston.target

print(dataSet.shape)
print(labels.shape)


#example1
kf=KFold(n_splits=4,shuffle=False)
n=kf.get_n_splits()
print("number of splitting iterations in the cross-validatior:",n)


#example2
kf2=KFold(n_splits=10,shuffle=False)
indices=kf2.split(X=dataSet,y=labels)
cv=0
for train_index,test_index in indices:
    print("cv:",cv)
    print("train_index:\n",train_index)
    print("test_index:\n",test_index)
    cv+=1