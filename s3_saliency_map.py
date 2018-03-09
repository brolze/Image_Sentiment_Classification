#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 17:11:41 2018

@author: xujq
"""
import pandas as pd

''' load data '''
from utils import splitXy,get_train_test
sample = pd.read_csv("data/sample.csv")
train = pd.read_csv("data/train.csv")
X,y = splitXy(train)
X_train,X_test,y_train,y_test = get_train_test(X,y)

''' load model '''
from keras.models import load_model
model = load_model("model/my_model(3-8 16:44).h5")





