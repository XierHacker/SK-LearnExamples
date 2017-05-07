from sklearn.model_selection import ShuffleSplit
from sklearn.datasets import load_boston

#load data
boston=load_boston()
dataSet=boston.data
labels=boston.target

print(dataSet.shape)
print(labels.shape)


#example1
sp=ShuffleSplit(n_splits=10,test_size=0.2)
print(sp.get_n_splits())

#example2
sp2=ShuffleSplit(n_splits=10,test_size=0.2)
indices=sp2.split(X=dataSet,y=labels)
for train_index,test_index in indices:
    print("train_index:\n",train_index)
    print("test_index:\n",test_index)
