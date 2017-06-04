import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import data
import merge

#从data中得到train,test,和labels
train_frame,train_labels_frame,test_frame=data.loadFrame()

#test
merged_frame=merge.merge(dataSet_frame=train_frame)
print(merged_frame.shape)

