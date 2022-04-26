"""Web Cam Detection and Recognition"""
import keras
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.layers import Layer
from keras import backend as K
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import imutils
import pickle
import sys

from FaceDetector import *
from face_rec_helper import *
from parameters import *
from MobileFaceNetModel import *
from generate_dataset import *

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
    
if(len(faces) == 0):
    print("No images in database.")
    sys.exit()

def triplet_loss(y_true, y_pred, alpha = ALPHA):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))    
    return loss

if os.path.exists(best_model_path) and best_model_path !="":
    print("Pre trained model found")
    FRmodel = keras.models.load_model(best_model_path,custom_objects={'triplet_loss': triplet_loss})
else:
    print("Custom trained model not found, loading one from scratch")
    FRmodel = build_model()
    # assign weights

def verification(image_path, identity, database, model):
    detected = False
    embedding = convert_embedding(image_path, model, False)
    min_dist = 1000
    for pic in database:
        # L2 Distance
        dist = np.linalg.norm(embedding - pic)
        if dist < min_dist:
            min_dist = dist
    print(identity + ' : ' +str(min_dist)+ ' ' + str(len(database)))
    if min_dist<THRESHOLD:
        detected = True
    else:
        detected = False
        
    return min_dist, detected


database = {}
for face in faces:
    database[face] = []

for face in faces:
    for img in os.listdir(paths[face]):
        database[face].append(convert_embedding(os.path.join(paths[face],img), FRmodel))

fd = FaceDetector('./fd_models/haarcascade_frontalface_default.xml')

camera = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') #Writes frames to an mp4 video file
out = cv2.VideoWriter('output.mp4', fourcc, 20, (800, 600) ) #out 

while True:
    ret, frame = camera.read()
    frame = imutils.resize(frame, width = 800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceRects = fd.detect(gray)
    for (x, y, w, h) in faceRects:
        foundFace = frame[y:y+h,x:x+w]
        foundFace = cv2.cvtColor(foundFace, cv2.COLOR_BGR2RGB)
        foundFace = cv2.resize(foundFace,(IMAGE_SIZE, IMAGE_SIZE))
        min_dist = 1000
        identity = ""
        detected  = False
        for face in range(len(faces)):
            person = faces[face]
            dist, detected = verification(foundFace, person, database[person], FRmodel)
            if detected == True and dist<min_dist:
                min_dist = dist
                identity = person
        if detected == True:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, identity, (x+ (w//2),y-2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), lineType=cv2.LINE_AA)
            
    cv2.imshow('frame', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
out.release()
cv2.destroyAllWindows()
