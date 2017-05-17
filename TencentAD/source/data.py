import pandas as pd
import numpy as np

def loadFrame():
    train_frame = pd.read_csv("../data/pre/train.csv")
    train_labels_frame=train_frame.pop("label")
    test_frame = pd.read_csv("../data/pre/test.csv",index_col="instanceID")
    test_frame.pop("label")
    return train_frame,train_labels_frame,test_frame


def loadUserFrame():
    user_frame = pd.read_csv("../data/pre/user.csv")
    return user_frame

def loadAdFrame():
    ad_frame = pd.read_csv("../data/pre/ad.csv")
    return ad_frame

def loadPositionFrame():
    position_frame = pd.read_csv("../data/pre/position.csv")
    return position_frame


def loadExtraFrame():
    app_categories_frame = pd.read_csv("../data/pre/app_categories.csv")
    user_app_actions_frame = pd.read_csv("../data/pre/user_app_actions.csv")
    user_installedapps_frame = pd.read_csv("../data/pre/user_installedapps.csv")
    return app_categories_frame,user_app_actions_frame, user_installedapps_frame


