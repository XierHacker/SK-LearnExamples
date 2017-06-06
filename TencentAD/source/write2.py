import numpy as np
import pandas as pd
import data
import merge2

print("-----------------------Load Data--------------------------")

#从data中得到testSet
test_frame=data.loadTestFrame()

#merged and one-hot
merged_frame=merge2.preprocess(dataSet_frame=test_frame)
#print(merged_frame.shape)
#print(merged_frame.head(10))
print(merged_frame.head(5))
print(merged_frame.shape)
merged_frame.to_csv(path_or_buf="test_merged.csv",index=False)