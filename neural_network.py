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

def getFaceFeatures(data, height, width, delimiter):
    features = []
    for i in range(height):
        for j in range(width):
            if(data[i][j]==delimiter):
                features.append(1)
            else:
                features.append(0)
    return features

def getDigitFeatures(data, height, width, delimiter1, delimiter2):
    features = []
    for i in range(height):
        for j in range(width):
            if(data[i][j]==delimiter1):
                features.append(1)
            elif(data[i][j]==delimiter2):   
                features.append(2)
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


def computeFaceAvgPredErr(testData, testLabels, trainingFeatures, trainingLabels):

    acc = []
    for _ in range(5):
        mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
        mlp.fit(trainingFeatures, trainingLabels)
        testFeatures = []
        for face in testData:
            testFeatures.append(getFaceFeatures(face, 70, 60, '#'))
        predict_test = mlp.predict(testFeatures)
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

def computeDigitAvgPredErr(testData, testLabels, trainingFeatures, trainingLabels):

    acc = []
    for _ in range(5):
        mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
        mlp.fit(trainingFeatures, trainingLabels)
        testFeatures = []
        for digit in testData:
            testFeatures.append(getDigitFeatures(digit, 28, 28, '+', '#'))
        predict_test = mlp.predict(testFeatures)
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
        features = []
        for face in newTrainingData:
            features.append(getFaceFeatures(face, 70, 60, '#'))
        t1_start = perf_counter()
        mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
        mlp.fit(features, newTrainingLabels)
        t1_stop = perf_counter()  
        print("Elapsed time in seconds:"+str(t1_stop-t1_start)+"\t for "+str(multiplier*10)+"% of data")
        mean, std_dev = computeFaceAvgPredErr(testData, testLabels, features, newTrainingLabels)
        print("Mean accuracy: "+str(mean * 100) + "%\t Std Dev: "+ str(std_dev * 100))

def getDigitStats():
    trainingData = readDigitData("trainingimages")
    trainingLabels = readDigitLabels("traininglabels")
    validationData = readDigitData("validationimages")
    validationLabels = readDigitLabels("validationlabels")
    testData = readDigitData("testimages")
    testLabels = readDigitLabels("testlabels")
    

    for multiplier in range(1,11):
        newTrainingData, newTrainingLabels = cutData(trainingData, trainingLabels, multiplier * 0.1)
        features = []
        for digit in newTrainingData:
            features.append(getDigitFeatures(digit, 28, 28, '+', '#'))
        t1_start = perf_counter()
        mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
        mlp.fit(features, newTrainingLabels)
        t1_stop = perf_counter()  
        print("Elapsed time in seconds:"+str(t1_stop-t1_start)+"\t for "+str(multiplier*10)+"% of data")
        mean, std_dev = computeDigitAvgPredErr(testData, testLabels, features, newTrainingLabels)
        print("Mean accuracy: "+str(mean * 100) + "%\t Std Dev: "+ str(std_dev * 100))

#getFaceStats()

getDigitStats()
