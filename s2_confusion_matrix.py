#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:23:53 2018

@author: xujq
"""


''' load model '''
from keras.models import load_model
model = load_model("model/my_model(3-8 16:44).h5")


''' get test data '''
import sys,os
sys.path.append(os.path.dirname(__file__))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
sample = pd.read_csv("data/sample.csv")
train = pd.read_csv("data/train.csv")

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
X,y = splitXy(train)
del train
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
del X,y



''' pick wrong prediction '''
#from keras.utils import to_categorical

def show_image(tmp):
    plt.figure(figsize=(1.5,1.5))  
    plt.imshow(tmp.reshape(48,48),cmap='gray')
    plt.show()

def label_meaning(label):
    meaning_dict = {
            0: u'生氣',
            1: u'厭惡',
            2: u'恐懼',
            3: u'高興',
            4: u'難過',
            5: u'驚訝',
            6: u'中立',         
            }
    return meaning_dict[label]

y_pred = model.predict_classes(X_test)
y_test = np.argmax(y_test,axis=1)
incorrects = np.nonzero(y_pred != y_test)[0]

limit = 20
start = 20
end = start + limit
while True:
    start +=1
    num = incorrects[start]
    show_image(X_test[num])
    pred = label_meaning(y_pred[num])
    true = label_meaning(y_test[num])
    print("pred %s , true label %s"%(pred,true))
    if start>end:
        break

''' confusion matrix '''
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib as mpl
ZHFONT = mpl.font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
cnf = confusion_matrix(y_test,y_pred)
def plot_cnf_matrix(cnf,cmap=plt.cm.Blues):
    plt.figure(figsize=(10,10))
    plt.imshow(cnf,cmap=cmap,interpolation="nearest")
    plt.ylabel("True label")
    plt.xlabel("Pred label")
    tick_marks = range(cnf.shape[0])
    name_marks = [label_meaning(x) for x in tick_marks]
    plt.xticks(tick_marks,name_marks,fontproperties=ZHFONT)
    plt.yticks(tick_marks,name_marks,fontproperties=ZHFONT)

#    cnf = cnf.astype('float') / cnf.sum(axis=1)[:, np.newaxis]
    thresh = cnf.max() / 2.
    for i,j in itertools.product(range(cnf.shape[0]),range(cnf.shape[1])):
        plt.text(i,j,format(cnf[i,j],'.2f'),
                 horizontalalignment="center",
                 color="white" if cnf[i, j] > thresh else "black")

plot_cnf_matrix(cnf)


''' try to find difference between fear(2) and hardfeeling(4)'''
analysis_y = pd.DataFrame([y_test,y_pred]).T
analysis_y.columns = ['true','pred']
analysis_y['correct'] = analysis_y['true'] == analysis_y['pred']
true_fear_but_hardfeeling = analysis_y[(analysis_y['true'] == 2)&(analysis_y['pred'] == 4)] 
true_hardfeeling_but_fear = analysis_y[(analysis_y['true'] == 4)&(analysis_y['pred'] == 2)]
true_fear = analysis_y[(analysis_y['true'] == 2)&(analysis_y['pred'] == 2)]
true_hardfeeling = analysis_y[(analysis_y['true'] == 4)&(analysis_y['pred'] == 4)]

def show_image_ax(tmp):
    plt.figure(figsize=(1.5,1.5))  
    plt.imshow(tmp.reshape(48,48),cmap='gray')
    plt.show()

pic_num = 0
type_ = true_fear_but_hardfeeling
fig,ax = plt.subplots(8,8,figsize=(12,12))
for i,j in itertools.product(range(8),range(8)):
    pic = X_test[type_.index[pic_num]]
    ax_tmp = ax[i,j]
    ax_tmp.imshow(pic.reshape(48,48),cmap='gray')
    pic_num+=1
    





























