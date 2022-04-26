import cv2
import numpy as np

# Face Detection using Haar Cascade Classifier
class FaceDetector: 
    def __init__(self, path):
        self.faceCascade = cv2.CascadeClassifier(path)

    def detect(self, image, scaleFactor= 1.02, minNeb = 15, minSize = (100, 100)):
        rects =  self.faceCascade.detectMultiScale(image, scaleFactor=scaleFactor,
                                                   minNeighbors=minNeb, minSize=minSize,
                                                   flags= cv2.CASCADE_SCALE_IMAGE)
        return rects