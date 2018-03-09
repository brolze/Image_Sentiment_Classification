#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 17:13:28 2018

@author: xujq
"""

''' get train/test data '''
import sys,os
sys.path.append(os.path.dirname(__file__))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def splitXy(train):
    """ train as sample """
    X_train = train['feature']
    X_train = X_train.apply(lambda x: np.array(x.split(' ')).astype(float))
    X_train = X_train.apply(lambda x:np.reshape(x,(48,48,1)))
    y_train = train['label']
    from keras.utils import to_categorical
    y_train = to_categorical(y_train)
    X_train = X_train.apply(lambda x:x/255)
    X_train = np.stack(X_train)
    return X_train,y_train

def get_train_test(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
    return X_train,X_test,y_train,y_test

if __name__ == "__main__":
    sample = pd.read_csv("data/sample.csv")
    train = pd.read_csv("data/train.csv")
    X,y = splitXy(train)
    X_train,X_test,y_train,y_test = get_train_test(X,y)

