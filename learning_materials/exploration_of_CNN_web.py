#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:03:23 2018

@author: xujq

https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
"""


img_width=128
img_height=128
step=1.0
layer_name = 'block5_conv1'
#filter_index : can be any integer from 0 to 511, as there are 512 filters in that layer
filter_index = 0

from keras import applications

# build the VGG16 network
model = applications.vgg16.VGG16(include_top=False,
                           weights='imagenet')
model.summary()

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])



from keras import backend as K

# this is the placeholder for the input images
input_img = model.input

# build a loss function that maximizes the activation
# of the nth filter of the layer considered
layer_out = layer_dict[layer_name].output
output = layer_out[:, :, :, filter_index]
loss = K.mean(output)

# compute the gradient of the input picture wrt this loss
grads = K.gradients(loss, input_img)[0]

# normalization trick: we normalize the gradient
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# this function returns the loss and grads given the input picture
iterate = K.function([input_img], [loss, grads,layer_out,output])


import numpy as np
# we start from a gray image with some noise
input_img_data = np.random.random((1, img_width, img_height, 3))
input_img_data = (input_img_data - 0.5) * 20 + 128
# run gradient ascent for 20 steps
for i in range(20):
    loss_value, grads_value,layer_out,output_ = iterate([input_img_data])
    print(output_)
    print(loss_value)
    input_img_data += grads_value * step
    
    

from scipy.misc import imsave
# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

img = input_img_data[0]
img = deprocess_image(img)
imsave('%s_filter_%d.png' % (layer_name, filter_index), img)