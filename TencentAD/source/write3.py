import pandas as pd
import numpy as np
import data

#load data from files
train_frame=data.loadTrainFrame()
test_frame=data.loadTestFrame()

print(train_frame.shape)
print(test_frame.shape)

train_frame["instanceID"]=0
train_frame.pop("conversionTime")
print(train_frame.shape)

#concat
concated_frame=pd.concat([train_frame,test_frame],axis=0)

#user.csv
user_frame=data.loadUserFrame()
#ad.csv
ad_frame=data.loadAdFrame()

#position.csv
position_frame=data.loadPositionFrame()

def preprocess(dataSet_frame):

    #---------------------------------merge-------------------------------*
    # merge dataSet_frame and user_frame
    merged_frame = pd.merge(left=dataSet_frame, right=user_frame, on="userID", how="inner")

    # merge dataSet_frame and ad_frame
    merged_frame = pd.merge(left=merged_frame, right=ad_frame, on="creativeID", how="inner")

    # merge dataSet_frame and position_frame
    merged_frame = pd.merge(left=merged_frame, right=position_frame, on="positionID", how="inner")

    #drop "clickTime","conversionTime","creativeID","userID",
    # "posionnID","hometown","residence","adID","camgaignID",
    #"appID",

    merged_frame=merged_frame.drop(
        labels=[
                    "clickTime","creativeID","userID",
                    "positionID","hometown","residence","adID","camgaignID","appID"
        ],
        axis=1
    )

    #----------------------------------one hot-------------------------------------
    merged_frame["age"] = merged_frame["age"] // 20

    # trans format to str
    merged_frame["connectionType"] = merged_frame["connectionType"].astype(str)
    merged_frame["telecomsOperator"] = merged_frame["telecomsOperator"].astype(str)
    merged_frame["age"] = merged_frame["age"].astype(str)
    merged_frame["gender"] = merged_frame["gender"].astype(str)
    merged_frame["education"] = merged_frame["education"].astype(str)
    merged_frame["marriageStatus"] = merged_frame["marriageStatus"].astype(str)
    merged_frame["haveBaby"] = merged_frame["haveBaby"].astype(str)
    merged_frame["advertiserID"] = merged_frame["advertiserID"].astype(str)
    merged_frame["appPlatform"] = merged_frame["appPlatform"].astype(str)
    merged_frame["sitesetID"] = merged_frame["sitesetID"].astype(str)
    merged_frame["positionType"] = merged_frame["positionType"].astype(str)

    # one hot encoding
    merged_frame_dummy = pd.get_dummies(data=merged_frame)

    return merged_frame_dummy


merged_frame_dummy=preprocess(dataSet_frame=concated_frame)
print(merged_frame_dummy.shape)

train_frame_dummy=merged_frame_dummy[merged_frame_dummy["label"]>-1]
print(train_frame_dummy.shape)
test_frame_dummy=merged_frame_dummy[merged_frame_dummy["label"]<0]
print(test_frame_dummy.shape)


train_frame_dummy.to_csv(path_or_buf="train_merged.csv",index=False)
test_frame_dummy.to_csv(path_or_buf="test_merged.csv",index=False)

