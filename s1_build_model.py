#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 22:44:12 2018

@author: xujq


label

0 生氣
1 厭惡
2 恐懼
3 高興
4 難過
5 驚訝
6 中立

"""

'''
settings
'''
import keras
from keras import optimizers
conv_activation = keras.activations.relu
learning_rate = 0.01
#sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
sgd = optimizers.adagrad(lr=learning_rate)
epochs = 40


import sys,os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sample = pd.read_csv("data/sample.csv")
train = pd.read_csv("data/train.csv")
#test = pd.read_csv("data/test.csv")


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

#for i in range(30):
#    tmp = X_train[i]
#    plt.imshow(tmp.reshape(48,48),cmap='gray')
#    plt.show()
#    print("label is %s"%str(y_train[i]))

X,y = splitXy(train)
#X_test,y_test = splitXy(test)
del train
#del test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
del X,y





import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
## 指定第2块GPU可用 
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

config = tf.ConfigProto()  
#config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
config.gpu_options.per_process_gpu_memory_fraction = 0.5 #进行配置，使用70%的GPU
sess = tf.Session(config=config)
KTF.set_session(sess)


from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Flatten,Dense


model = Sequential(name = "model")
model.add(Conv2D(filters=8,
                 kernel_size=(3,3),
                 input_shape=(48,48,1),
                 activation = conv_activation,
                 data_format='channels_last')) # 48*48*1 -> 46*46*8
model.add(Conv2D(filters=16,
                 kernel_size=(3,3),
                 input_shape=(46,46,8),
                 activation = conv_activation,
                 data_format='channels_last')) # 46*46*8 -> 44*44*16
model.add(MaxPool2D(pool_size=(2,2),
                    padding="valid")) # 44*44*16 -> 22*22*16

model.add(Conv2D(filters=32,
                 kernel_size=(3,3),
                 input_shape=(22,22,16),
                 activation = conv_activation,
                 data_format='channels_last')) # 22*22*16 -> 20*20*32
model.add(Conv2D(filters=32,
                 kernel_size=(3,3),
                 input_shape=(20,20,32),
                 activation = conv_activation,
                 data_format='channels_last')) # 20*20*32 -> 18*18*32
model.add(MaxPool2D(pool_size=(2,2),
                    padding="valid")) # 18*18*32 -> 9*9*32


model.add(Flatten()) # 9*9*32 = 2592
model.add(Dense(units=128,activation="sigmoid"))
model.add(Dense(units=7,activation="softmax"))


# add optimizer
model.compile(sgd,loss="categorical_crossentropy",metrics=['accuracy'])
history = model.fit(x=X_train,y=y_train,batch_size=128,epochs=epochs,
                    validation_data=(X_test, y_test))

fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
ax = axes[0]
ax.plot(history.history['acc'],'o-',label="train")
ax.plot(history.history['val_acc'],'o-',label="test")
ax.set_xlabel('epochs')
ax.set_ylabel('accuracy')
ax = axes[1]
ax.plot(history.history['loss'],'o-',label="train")
ax.plot(history.history['val_loss'],'o-',label="test")
ax.set_xlabel('epochs')
ax.set_ylabel('loss')
plt.show()

import time
model_info = ""
model.save("model/my_model(%i-%i %i:%i)(%s).h5"%(time.localtime().tm_mon,
                            time.localtime().tm_mday,
                            time.localtime().tm_hour,
                            time.localtime().tm_min),
                            model_info)
model.summary()



