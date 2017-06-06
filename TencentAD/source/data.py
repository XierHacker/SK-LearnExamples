import pandas as pd
import numpy as np

def loadTrainFrame():
    train_frame = pd.read_csv("../data/pre/train.csv")
    return train_frame

def loadTestFrame():
    test_frame = pd.read_csv("../data/pre/test.csv")
    return test_frame


def loadUserFrame():
    user_frame = pd.read_csv("../data/pre/user.csv")
    return user_frame

def loadAdFrame():
    ad_frame = pd.read_csv("../data/pre/ad.csv")
    return ad_frame

def loadPositionFrame():
    position_frame = pd.read_csv("../data/pre/position.csv")
    return position_frame

def loadAppCategoriesFrame():
    app_categories_frame = pd.read_csv("../data/pre/app_categories.csv")
    return app_categories_frame


def loadUserAppActionsFrame():
    user_app_actions_frame = pd.read_csv("../data/pre/user_app_actions.csv")
    return user_app_actions_frame

def loadUserInstalledAppsFrame():
    user_installedapps_frame = pd.read_csv("../data/pre/user_installedapps.csv")
    return user_installedapps_frame


