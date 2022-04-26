"""MobileFaceNetModel Architecture"""
import os
import sys
import math
import keras
import tensorflow as tf
import numpy as np

from keras.models import Model
import keras.backend as K
from tensorflow.keras.optimizers import Adam
from keras.layers import (Conv2D, 
                          BatchNormalization, 
                          ReLU, 
                          DepthwiseConv2D, 
                          Activation, 
                          Input, 
                          Add, 
                          Flatten, 
                          Dense, 
                          Lambda,
                          Softmax)

from parameters import *

"""Convolutional Layer: Conv2D, Batch Normalization, ReLU Activation"""
def conv2d_layer(inputs, filters, kernel, strides, is_use_bias=False, padding='same'):
    x = Conv2D(filters, kernel, strides= strides, padding=padding, use_bias=is_use_bias)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

"""Bottle Neck Layer"""
def bottleneck_layer(inputs, out_dim, strides, expansion_ratio, is_use_bais=False, shortcut=True):

    bottleneck_dim = K.int_shape(inputs)[-1] * expansion_ratio

    # Convolutional Layer
    x = conv2d_layer(inputs, bottleneck_dim, kernel=(1, 1), strides=(1, 1), is_use_bias=is_use_bais)

    # Depthwise Convolutional Layer
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=strides, depth_multiplier=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Convolutional Layer
    x = Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)


    if shortcut and strides == (1, 1):
        in_dim = K.int_shape(inputs)[-1]
        if in_dim != out_dim:
            ins = Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same')(inputs)
            x = Add()([x, ins])
        else:
            x = Add()([x, inputs])
    return x

def build_model(input_shape=(112, 112, 3), embedding_dim=128):
    image_inputs = Input(shape=input_shape)

    # Operator: conv3x3, c=64, n=1, s=2 --> Input 112x112x3 Output 56x56x64
    net = conv2d_layer(image_inputs, filters=64, kernel=(3, 3), strides=(2, 2), is_use_bias=False) # size/2 (56)

    # Operator: depthwise conv3x3, c=64, n=1, s=1 --> Input 56x56x64 Output 56x56x64
    net = bottleneck_layer(net, out_dim=64, strides=(1, 1), expansion_ratio=1, is_use_bais=False, shortcut=True)

    # Operator: bottleneck, c=64, n=1, s=2, t=2 --> Input 56x56x64 Output 28x28x64
    net = bottleneck_layer(net, out_dim=64, strides=(2, 2), expansion_ratio=2, is_use_bais=False, shortcut=True) # size/4 (28)

    # Operator: bottleneck, c=64, n=4, s=1, t=2 --> Input 28x28x64 Output 28x28x64
    net = bottleneck_layer(net, out_dim=64, strides=(1, 1), expansion_ratio=2, is_use_bais=False, shortcut=True)
    net = bottleneck_layer(net, out_dim=64, strides=(1, 1), expansion_ratio=2, is_use_bais=False, shortcut=True)
    net = bottleneck_layer(net, out_dim=64, strides=(1, 1), expansion_ratio=2, is_use_bais=False, shortcut=True)
    net = bottleneck_layer(net, out_dim=64, strides=(1, 1), expansion_ratio=2, is_use_bais=False, shortcut=True)

    # Operator: bottleneck, c=128, n=1, s=2, t=4 --> Input 28x28x64 Output 14x14x128
    net = bottleneck_layer(net, out_dim=128, strides=(2, 2), expansion_ratio=4, is_use_bais=False, shortcut=True) # size/8 (14)

    # Operator: bottleneck, c=128, n=6, s=1, t=2 --> Input 14x14x128 Output 14x14x128
    net = bottleneck_layer(net, out_dim=128, strides=(1, 1), expansion_ratio=2, is_use_bais=False, shortcut=True)
    net = bottleneck_layer(net, out_dim=128, strides=(1, 1), expansion_ratio=2, is_use_bais=False, shortcut=True)
    net = bottleneck_layer(net, out_dim=128, strides=(1, 1), expansion_ratio=2, is_use_bais=False, shortcut=True)
    net = bottleneck_layer(net, out_dim=128, strides=(1, 1), expansion_ratio=2, is_use_bais=False, shortcut=True)
    net = bottleneck_layer(net, out_dim=128, strides=(1, 1), expansion_ratio=2, is_use_bais=False, shortcut=True)
    net = bottleneck_layer(net, out_dim=128, strides=(1, 1), expansion_ratio=2, is_use_bais=False, shortcut=True)

    # Operator: bottleneck, c=128, n=1, s=2, t=4 --> Input 14x14x128 Output 7x7x128
    net = bottleneck_layer(net, out_dim=128, strides=(2, 2), expansion_ratio=4, is_use_bais=False, shortcut=True) # size/16 (7)

    # Operator: bottleneck, c=128, n=2, s=1, t=2 --> Input 7x7x128 Output 7x7x128
    net = bottleneck_layer(net, out_dim=128, strides=(1, 1), expansion_ratio=2, is_use_bais=False, shortcut=True)
    net = bottleneck_layer(net, out_dim=128, strides=(1, 1), expansion_ratio=2, is_use_bais=False, shortcut=True)
    
    # Operator: conv1x1, c=512, n=1, s=1 --> Input 7x7x128 Output 7x7x512
    net = conv2d_layer(net, 512, (1, 1), (1, 1), True, 'valid')
    
    # Operator: linear GDConv7x7, c=512, n=1, s=1 --> Input 7x7x512 Output 1x1x512
    net = DepthwiseConv2D((7, 7), strides=(1, 1), depth_multiplier=1, padding='valid')(net)
    
    # Operator: linear conv1x1, c=128, n=1, s=1 --> Input 1x1x512 Output 1x1x128
    net = conv2d_layer(net, embedding_dim, (1, 1), (1, 1), True, 'valid')

    # Flatten Layer for Embedding
    net = Flatten()(net)
    net = Dense(128, name='dense_layer')(net)
    net = Lambda(lambda x: K.l2_normalize(x, axis=-1))(net)

    # generate model
    model = Model(inputs=image_inputs, outputs=net, name='MobileFaceNetModel')

    return model

# Define Custom Triplet Loss Function
def triplet_loss(y_true, y_pred, alpha = ALPHA):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.maximum(basic_loss, 0.0)
    return loss

# Test Model
if __name__ == '__main__':
    model = build_model()
    optimizer = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss=triplet_loss)
    print('\n')
    print('summary')
    print(model.summary())