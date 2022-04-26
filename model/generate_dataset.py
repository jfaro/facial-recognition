"""Generate Training Dataset: create triplet pairs for FaceNet w/ triplet loss function"""
import os
import cv2
import numpy as np
from parameters import *
import pickle

input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

# Load Dataset Dictionary
with open("./path_dict.p", 'rb') as f:
    paths = pickle.load(f)

faces = []
for key in paths.keys():
    paths[key] = paths[key].replace("\\", "/")
    faces.append(key)

print(faces)

images = {}
for key in paths.keys():
    li = []
    for img in os.listdir(paths[key]):
        img1 = cv2.imread(os.path.join(paths[key],img))
        img2 = img1[...,::-1]
        # print(np.transpose(img2, (2,0,1)).shape)
        # print(np.around(img2 / 255.0, decimals=12).shape)
        li.append(np.around(img2 / 255.0, decimals=12))
    images[key] = np.array(li)

# images key: path --> value: list of images (numpy arrays (112, 112, 3))

# generate dataset for custom made training function for model
def generate_dataset(training_set=250):
    finalDataSet = np.zeros((training_set, 3, input_shape[0], input_shape[1], input_shape[2]))
    for i in range(training_set):
        positiveFace = faces[np.random.randint(len(faces))]
        negativeFace = faces[np.random.randint(len(faces))]
        
        while positiveFace == negativeFace:
            negativeFace = faces[np.random.randint(len(faces))]
        
        randInt1 = np.random.randint(len(images[positiveFace]))
        randInt2 = np.random.randint(len(images[positiveFace]))
        
        while randInt1 == randInt2:
            randInt2 = np.random.randint(len(images[positiveFace]))
        
        finalDataSet[i][0] = images[positiveFace][randInt1]
        finalDataSet[i][1] = images[positiveFace][randInt2]
        finalDataSet[i][2] = images[negativeFace][np.random.randint(len(images[negativeFace]))]
    
    return finalDataSet

# batch generator for model.fit() method
def batch_generator(batch_size=16):
    y_val = np.zeros((batch_size, 2, 1))
    anchors = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
    positives = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
    negatives = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
    
    while True:
        for i in range(batch_size):
            positiveFace = faces[np.random.randint(len(faces))]
            negativeFace = faces[np.random.randint(len(faces))]
            while positiveFace == negativeFace:
                negativeFace = faces[np.random.randint(len(faces))]

            positives[i] = images[positiveFace][np.random.randint(len(images[positiveFace]))]
            anchors[i] = images[positiveFace][np.random.randint(len(images[positiveFace]))]
            negatives[i] = images[negativeFace][np.random.randint(len(images[negativeFace]))]
        
        x_data = {'anchor': anchors,
                  'anchorPositive': positives,
                  'anchorNegative': negatives
                  }
        
        yield (x_data, [y_val, y_val, y_val])