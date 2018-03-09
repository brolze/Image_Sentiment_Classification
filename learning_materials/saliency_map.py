#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 11:46:03 2018

@author: xujq
"""

import os
import argparse
from keras.models import loadmodel
from termcolor import colored,cprint
import keras.backend as K
from utils import *
import numpy as np
import matplotlib.pyplot as plt

def deprocessimage(x):
    """
    Hint: Normalize and Clip
    """
    return x

basedir = os.path.dirname(os.path.dirname(os.path.realpath(file)))
imgdir = os.path.join(base_dir, 'image')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
cmap_dir = os.path.join(img_dir, 'cmap')
if not os.path.exists(cmap_dir):
    os.makedirs(cmap_dir)
partial_see_dir = os.path.join(img_dir,'partial_see')
if not os.path.exists(partial_see_dir):
    os.makedirs(partial_see_dir)
model_dir = os.path.join(base_dir, 'model')

def main():
    parser = argparse.ArgumentParser(prog='plot_saliency.py',
            description='ML-Assignment3 visualize attention heat map.')
    parser.add_argument('--epoch', type=int, metavar='<#epoch>', default=1)
    args = parser.parse_args()
    model_name = "model-%s.h5" %str(args.epoch)
    model_path = os.path.join(model_dir, model_name)
    emotion_classifier = load_model(model_path)
    print(colored("Loaded model from {}".format(model_name), 'yellow', attrs=['bold']))

    private_pixels = load_pickle('fer2013/test_with_ans_pixels.pkl')
    private_pixels = [ np.fromstring(private_pixels[i], dtype=float, sep=' ').reshape((1, 48, 48, 1)) 
                       for i in range(len(private_pixels)) ]
    input_img = emotion_classifier.input
    img_ids = [17]

    for idx in img_ids:
        val_proba = emotion_classifier.predict(private_pixels[idx])
        pred = val_proba.argmax(axis=-1)
        target = K.mean(emotion_classifier.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grads])

        """
        Implement your heatmap processing here!
        hint: Do some normalization or smoothening on grads
        """

        thres = 0.5
        see = private_pixels[idx].reshape(48, 48)
        # for i in range(48):
            # for j in range(48):
                # print heatmap[i][j]
        see[np.where(heatmap <= thres)] = np.mean(see)

        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        test_dir = os.path.join(cmap_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        fig.savefig(os.path.join(test_dir, '{}.png'.format(idx)), dpi=100)

        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        test_dir = os.path.join(partial_see_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        fig.savefig(os.path.join(test_dir, '{}.png'.format(idx)), dpi=100)

if __name == "__main":
    main()
    ""
    
    
    