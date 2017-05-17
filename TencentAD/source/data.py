import pandas as pd
import numpy as np

def loadFrame():
    train_frame = pd.read_csv("../data/pre/train.csv")
    train_labels_frame=train_frame.pop("label")
    test_frame = pd.read_csv("../data/pre/test.csv")
    return train_frame,train_labels_frame,test_frame



def loadExtraFrame():
    ad_frame = pd.read_csv("../data/pre/ad.csv")
    app_categories_frame = pd.read_csv("../data/pre/app_categories.csv")
    position_frame = pd.read_csv("../data/pre/position.csv")
    user_frame = pd.read_csv("../data/pre/user.csv")
    user_app_actions_frame = pd.read_csv("../data/pre/user_app_actions.csv")
    user_installedapps_frame = pd.read_csv("../data/pre/user_installedapps.csv")
    return ad_frame, app_categories_frame, position_frame, \
           user_frame, user_app_actions_frame, user_installedapps_frame


