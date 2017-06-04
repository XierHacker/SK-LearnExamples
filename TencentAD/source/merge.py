import pandas as pd
import numpy as np
import data

#从data中得到train,test,和labels
train_frame,train_labels_frame,test_frame=data.loadFrame()
#user.csv
user_frame=data.loadUserFrame()
#ad.csv
ad_frame=data.loadAdFrame()

#position.csv
position_frame=data.loadPositionFrame()


def merge(dataSet_frame,isTrain=True):
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
                    "clickTime","conversionTime","creativeID","userID",
                    "positionID","hometown","residence","adID","camgaignID","appID"
        ],
        axis=1
    )
    return merged_frame



