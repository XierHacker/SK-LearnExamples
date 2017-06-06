import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
from mlp import MLP
import zipfile



print("-----------------------Load Data--------------------------")
train_frame=pd.read_csv("train_merged.csv")
print("shape of train_frame:",train_frame.shape)

test_frame=pd.read_csv("test_merged.csv")
print("shape of test_frame:",test_frame.shape)


train_labels_frame=train_frame.pop(item="label")
test_frame.pop(item="label")

train_frame.pop(item="instanceID")
ID_frame=test_frame.pop(item="instanceID")


print("shape of train_frame:",train_frame.shape)
print("shape of test_frame:",test_frame.shape)


trainSet=train_frame.values
train_labels=train_labels_frame.values
testSet=test_frame.values

#load model
model=MLP(300,150)
model.fit(X=trainSet,y=train_labels,print_log=True)

prob=model.predict_prob(testSet)


df = pd.DataFrame({'instanceID': ID_frame.values, 'prob': prob})
df.sort_values('instanceID', inplace=True)
df.to_csv('submission.csv', index=False)
with zipfile.ZipFile('submission.zip', 'w') as fw:
    fw.write('submission.csv', compress_type=zipfile.ZIP_DEFLATED)