import numpy as np
import copy
import random
from time import perf_counter

def getFeatures(data, height, width, x, y, delimiter):
    score = 0
    features = []
    numRows = x
    rowPixels = height // numRows
    numCols = y
    colPixels = width // numCols
    prev1 = 0
    for _ in range(numRows):
        prev2 = 0
        for _ in range(numCols):
            for x in range(prev1, prev1 + rowPixels):
                for y in range(prev2, prev2 + colPixels):
                    if(data[x][y] == delimiter):
                        score += 1
            features.append(score)
            prev2 = prev2 + colPixels
        prev1 = prev1 + rowPixels
    return features

def cutData(data, labels, percentData):
    newData = []
    numData = len(data)
    newData = copy.deepcopy(data)
    newLabels = copy.deepcopy(labels)
    newNumData = int(numData * percentData)
    combined_list = list(zip(newData, newLabels))
    # shuffle the list of tuples
    random.shuffle(combined_list)
    # separate the tuples back into separate lists
    newData, newLabels = zip(*combined_list)
    newData = list(newData)
    newLabels = list(newLabels)
    newNumData = int(len(newData) * percentData)
    newData1 = []
    newLabels1 = []
    for i in range(newNumData):
        newData1.append(copy.deepcopy(newData[i]))
        newLabels1.append(copy.deepcopy(newLabels[i]))
    return newData1, newLabels1

def getFaceFeaturesProb(faces):
    featureProb = []
    features = []
    for face in faces:
        features.append(getFeatures(face, 70, 60, 70, 60, '#'))
    for i in range(70 * 60):
        count = 0
        for j, face in enumerate(faces):
            if(features[j][i]==1):
                count += 1
        if(count == 0):
            featureProb.append(1/len(faces))
        else:
            featureProb.append(count/len(faces))
    return featureProb

def getNotFaceFeatureProb(notFaces):
    featureProb = []
    features = []
    for notFace in notFaces:
        features.append(getFeatures(notFace, 70, 60, 70, 60, '#'))
    for i in range(70 * 60):
        count = 0
        for j, notFace in enumerate(notFaces):
            if(features[j][i]==0):
                count += 1
        if(count == 0):
            featureProb.append(1/len(notFaces))
        else:
            featureProb.append(count/len(notFaces))
    
    return featureProb

def probImgGivenFace(faces, featureProb):
    pass
def probImgGivenNotFace(notFaces, featureProb):  
    pass

def determineFace(img, probImgGivenFace, probFace, probImgGivenNotFace, probNotFace):
    probFaceGivenImg = probImgGivenFace(img) * probFace
    probNotFaceGivenImg = probImgGivenNotFace(img) * probNotFace