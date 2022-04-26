"""Test Net using SVM"""
import os
import sys
import math
import keras 
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Layer
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
                          
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.callbacks import LearningRateScheduler
import time

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from matplotlib import pyplot as plt

from MobileFaceNetModel import *
from generate_dataset import *
from parameters import *
from face_rec_helper import convert_embedding

best_model_path =""
if(os.path.exists("bestmodel.txt")):
    with open('bestmodel.txt', 'r') as file:
        best_model_path = file.read()
    
with open("./path_dict.p", 'rb') as f:
    paths = pickle.load(f)
    
faces = []
for key in paths.keys():
    paths[key] = paths[key].replace("\\", "/")
    faces.append(key)

if os.path.exists(best_model_path) and best_model_path !="":
    print("Pre trained model found")
    FRmodel = keras.models.load_model(best_model_path,custom_objects={'triplet_loss': triplet_loss})
else:
    print("Custom trained model not found, loading one from scratch")
    FRmodel = build_model()
    # assign weights

X_train, Y_train, X_val, Y_val = [], [], [], []

database = {}
for face in faces:
    database[face] = []

for face in faces:
    for img in os.listdir(paths[face]):
        randVal = np.random.randint(0, 10)
        if (randVal <= 7):
            X_train.append(convert_embedding(os.path.join(paths[face],img), FRmodel).reshape(-1))
            Y_train.append(face)
        else:
            X_val.append(convert_embedding(os.path.join(paths[face],img), FRmodel).reshape(-1))
            Y_val.append(face)

def cross_validation(clf, X, y, k=5):
    scores = []
    kFold = StratifiedKFold(n_splits=k)

    for train_index, test_index in kFold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = y_pred = clf.predict(X_test)
        compScore = metrics.accuracy_score(y_test, y_pred)
        scores.append(compScore)
    
    return np.array(scores).mean()

def select_c_param(X_train, y_train, k=5, C_range=[]):
    bestScore = 0
    bestC = 0
    # Find trends in C and find best performace
    for param in C_range:
        testSVM = SVC(C=param, kernel='rbf', decision_function_shape="ovo")
        result = cross_validation(testSVM, X_train, y_train, k)
        if result > bestScore:
            bestScore = result
            bestC = param
        elif result == bestScore:
            if param < bestC:
                bestC = param
        print("C:", param)
        print("Performance:", result)
    # pring best
    print("Best C:", bestC)
    print("Best Performance:", bestScore)
    return bestC

C_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
X_train, Y_train = np.array(X_train), np.array(Y_train)
best_c = select_c_param(X_train, Y_train, 5, C_range)
clf = SVC(C=best_c, kernel='rbf', decision_function_shape="ovo")
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_val)
print(Y_val)
print(y_pred)

accuracy = metrics.accuracy_score(Y_val, y_pred)
print(accuracy)
# img1 = cv2.imread('./cropped/Angelina_Jolie/Angelina_Jolie_0001.png')
# img1 = np.around(img1 / 255.0, decimals=12)

# for face in range(len(faces)):
#     person = faces[face]
#     dist, detected = verify('./cropped/Angelina_Jolie/Angelina_Jolie_0001.png', person, database[person], FRmodel)
#     if detected == True and dist<min_dist:
#         min_dist = dist
#         identity = person
# print(identity)