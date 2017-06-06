import numpy as np
import pandas as pd
import data
import merge

print("-----------------------Load Data--------------------------")

#从data中得到trainSet
train_frame=data.loadTrainFrame()

#merged and one-hot
merged_frame=merge.preprocess(dataSet_frame=train_frame)
#print(merged_frame.shape)
#print(merged_frame.head(10))
print(merged_frame.head(5))
print(merged_frame.shape)
merged_frame.to_csv(path_or_buf="train_merged.csv",index=False)



