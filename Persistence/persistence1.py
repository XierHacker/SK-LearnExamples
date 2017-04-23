import numpy as np
from sklearn.linear_model import Ridge
from sklearn.externals import joblib
from sklearn.datasets import load_boston

#load data
boston=load_boston()
#print(boston.data.shape)
#print(boston.target.shape)

#split testSet and trainSet
trainSet=boston.data[:400]
trainLabels=boston.target[:400]
#print(trainSet.shape)
#print(trainLabels.shape)

testSet=boston.data[400:]
testLabels=boston.target[400:]


#use model
ridge=Ridge()
#train
ridge.fit(X=trainSet,y=trainLabels)
#ridge.fit(X=boston.data[,y=boston.target)

#predict
result=ridge.predict(X=testSet)
print("result:\n",result)
print("\n\n\n")
print("testLabels:\n",testLabels)

#save model to disk
joblib.dump(value=ridge,filename="ridgeModel.gz",compress=True)
print("model has saved!!")

#load model from disk
model=joblib.load(filename="ridgeModel.gz")
print(type(model))
result2=model.predict(testSet)
print(result2)



