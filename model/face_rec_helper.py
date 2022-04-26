"""Facial Recognition Helper Functions: Weight Init and Embedding"""

import tensorflow as tf
import numpy as np
import os
import cv2
from numpy import genfromtxt
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
import h5py
import matplotlib.pyplot as plt

# ?? TODO: DOESNT WORK
WEIGHTS = [
  'conv2d', 'batch_normalization', 'conv2d_1', 'batch_normalization_1', 
  'depthwise_conv2d', 'batch_normalization_2', 'conv2d_2', 'batch_normalization_3',
  'conv2d_3', 'batch_normalization_4', 'depthwise_conv2d_1', 'batch_normalization_5',
  'conv2d_4', 'batch_normalization_6', 'conv2d_5', 'batch_normalization_7', 
  'depthwise_conv2d_2', 'batch_normalization_8', 'conv2d_6', 'batch_normalization_9',
  'conv2d_7', 'batch_normalization_10', 'depthwise_conv2d_3', 'batch_normalization_11',
  'conv2d_8', 'batch_normalization_12', 'conv2d_9', 'batch_normalization_13',
  'depthwise_conv2d_4', 'batch_normalization_14', 'conv2d_10', 'batch_normalization_15',
  'conv2d_11', 'batch_normalization_16', 'depthwise_conv2d_5', 'batch_normalization_17',
  'conv2d_12', 'batch_normalization_18', 'conv2d_13', 'batch_normalization_19', 
  'depthwise_conv2d_6', 'batch_normalization_20', 'conv2d_14', 'batch_normalization_21',
  'conv2d_15', 'batch_normalization_22', 'depthwise_conv2d_7', 'batch_normalization_23',
  'conv2d_16', 'batch_normalization_24', 'conv2d_17', 'batch_normalization_25', 
  'depthwise_conv2d_8', 'batch_normalization_26', 'conv2d_18', 'batch_normalization_27',
  'conv2d_19', 'batch_normalization_28', 'depthwise_conv2d_9', 'batch_normalization_29',
  'conv2d_20', 'batch_normalization_30', 'conv2d_21', 'batch_normalization_31',
  'depthwise_conv2d_10', 'batch_normalization_32', 'conv2d_22', 'batch_normalization_33',
  'conv2d_23', 'batch_normalization_34', 'depthwise_conv2d_11', 'batch_normalization_35',
  'conv2d_24', 'batch_normalization_36', 'conv2d_25', 'batch_normalization_37',
  'depthwise_conv2d_12', 'batch_normalization_38', 'conv2d_26', 'batch_normalization_39',
  'conv2d_27', 'batch_normalization_40', 'depthwise_conv2d_13', 'batch_normalization_41',
  'conv2d_28', 'batch_normalization_42', 'conv2d_29', 'batch_normalization_43', 
  'depthwise_conv2d_14', 'batch_normalization_44', 'conv2d_30', 'batch_normalization_45',
  'conv2d_31', 'batch_normalization_46', 'depthwise_conv2d_15', 'batch_normalization_47',
  'conv2d_32', 'batch_normalization_48', 'conv2d_33', 'batch_normalization_49',
  'depthwise_conv2d_16', 'conv2d_34', 'atch_normalization_50', 
  'dense_layer'
]

# ?? TODO: FIGURE OUT WEIGHTS, THEN FILL IN SHAPE FOR EACH WEIGHT
conv_shape = {
    'conv2d': [64, 3, 1, 1]
}

# ?? TODO: DOESNT WORK
def load_weights_from_FaceNet(FRmodel):
    # Load weights from csv files
    weights = WEIGHTS
    weights_dict = load_weights()

    # Set layer weights of the model
    for name in weights:
        if FRmodel.get_layer(name) != None:
            FRmodel.get_layer(name).set_weights(weights_dict[name])
        elif FRmodel.get_layer(name) != None:
            FRmodel.get_layer(name).set_weights(weights_dict[name])

def load_weights():
    # Set weights path
    dirPath = './weights'
    fileNames = filter(lambda f: not f.startswith('.'), os.listdir(dirPath))
    paths = {}
    weights_dict = {}

    for n in fileNames:
        paths[n.replace('.csv', '')] = dirPath + '/' + n

    for name in WEIGHTS:
        if 'conv' in name:
            conv_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
            conv_w = np.reshape(conv_w, conv_shape[name])
            conv_w = np.transpose(conv_w, (2, 3, 1, 0))
            conv_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
            weights_dict[name] = [conv_w, conv_b]     
        elif 'batch_normalization' in name:
            bn_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
            bn_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
            bn_m = genfromtxt(paths[name + '_m'], delimiter=',', dtype=None)
            bn_v = genfromtxt(paths[name + '_v'], delimiter=',', dtype=None)
            weights_dict[name] = [bn_w, bn_b, bn_m, bn_v]
        elif 'dense' in name:
            dense_w = genfromtxt(dirPath+'/dense_w.csv', delimiter=',', dtype=None)
            dense_w = np.reshape(dense_w, (128, 736))
            dense_w = np.transpose(dense_w, (1, 0))
            dense_b = genfromtxt(dirPath+'/dense_b.csv', delimiter=',', dtype=None)
            weights_dict[name] = [dense_w, dense_b]

    return weights_dict

def convert_embedding(image_path, model, path=True):
    if path == True:
        img1 = cv2.imread(image_path, 1)
    else:
        img1 = image_path
    img = img1[...,::-1]
    # img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    img = np.around(img / 255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding