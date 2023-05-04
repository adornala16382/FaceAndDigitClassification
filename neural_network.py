# Import required libraries
import copy
import random
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from time import perf_counter

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score


def readDigitData(filename):
    counter = 0
    index = 0
    digitsData = [[]]
    with open("digitdata/" + filename, 'r') as f:
        for line in f:
            data_str = line[:28]
            digitsData[index].append(data_str)
            counter += 1
            if(counter == 28):
                digitsData.append([])
                counter = 0
                index += 1
    digitsData.pop()

    return digitsData

def readFaceData(filename):
    counter = 0
    index = 0
    facesData = [[]]
    with open("facedata/" + filename, 'r') as f:
        for line in f:
            data_str = line[:60]
            facesData[index].append(data_str)
            counter += 1
            if(counter == 70):
                facesData.append([])
                counter = 0
                index += 1
    facesData.pop()

    return facesData

def getFeatures(data, height, width, delimiter):
    features = []
    for i in range(height):
        for j in range(width):
            if(data[i][j]==delimiter):
                features.append(1)
            else:
                features.append(0)
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
def readFaceLabels(filename):
    data = []
    with open("facedata/" + filename, 'r') as f:
        for line in f:
            data_str = line[:1]
            if(data_str=='1'):
                data.append(True)
            elif(data_str=='0'):
                data.append(False)
    return data

def readDigitLabels(filename):
    data = []
    with open("digitdata/" + filename, 'r') as f:
        for line in f:
            data_str = line[:1]
            data.append(int(data_str))
    return data


def computeFaceAvgPredErr(testData, testLabels, trainingData, trainingLabels):

    acc = []
    for _ in range(5):
        mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
        mlp.fit(trainingData, trainingLabels)
        predict_test = mlp.predict(testData)
        count = 0
        for i, p in enumerate(predict_test):
            if(p == testLabels[i]):
                count+=1
        prob = count /len(predict_test)
        acc.append(prob)
    arr = np.array(acc)
    mean = np.mean(arr)
    std_dev = np.std(arr)

    return mean, std_dev

def getFaceStats():
    trainingData = readFaceData("facedatatrain")
    trainingLabels = readFaceLabels("facedatatrainlabels")
    validationData = readFaceData("facedatavalidation")
    validationLabels = readFaceLabels("facedatavalidationlabels")
    testData = readFaceData("facedatatest")
    testLabels = readFaceLabels("facedatatestlabels")
    

    for multiplier in range(1,11):
        newTrainingData, newTrainingLabels = cutData(trainingData, trainingLabels, multiplier * 0.1)
        t1_start = perf_counter()
        mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
        mlp.fit(newTrainingData, newTrainingLabels)
        t1_stop = perf_counter()  
        print("Elapsed time in seconds:"+str(t1_stop-t1_start))
        mean, std_dev = computeFaceAvgPredErr(testData, testLabels, newTrainingData, newTrainingLabels)
        print("Mean accuracy: "+str(mean * 100) + "%\t Std Dev: "+ str(std_dev * 100))

getFaceStats()
