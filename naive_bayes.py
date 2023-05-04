import copy
import random
from time import perf_counter
import numpy as np
import math

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

def getDigitFeaturesProb(digits):
    featureProb1 = [[] for i in range(10)]
    featureProb2 = [[] for i in range(10)]
    features1 = [[] for i in range(10)]
    features2 = [[] for i in range(10)]
    for j in range(10):
        for i, digit in enumerate(digits[j]):
            features1[j].append(getFeatures(digit, 28, 28, '+'))
            features2[j].append(getFeatures(digit, 28, 28, '#'))

    for num in range(10):
        for i in range(28 * 28):
            count1 = 0
            count2 = 0
            for j, digit in enumerate(digits[num]):
                if(features1[num][j][i]==1):
                    count1 += 1
                if(features2[num][j][i]==1):
                    count2 += 1
            if(count1 == 0):
                featureProb1[num].append(1/len(digits[num]))
            else:
                featureProb1[num].append(count1/len(digits[num]))
            if(count2 == 0):
                featureProb2[num].append(1/len(digits[num]))
            else:
                featureProb2[num].append(count2/len(digits[num]))

    return featureProb1, featureProb2

def getFaceFeaturesProb(faces):
    featureProb = []
    features = []
    for face in faces:
        features.append(getFeatures(face, 70, 60, '#'))
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
        features.append(getFeatures(notFace, 70, 60, '#'))
    for i in range(70 * 60):
        count = 0
        for j, notFace in enumerate(notFaces):
            if(features[j][i]==1):
                count += 1
        if(count == 0):
            featureProb.append(1/len(notFaces))
        else:
            featureProb.append(count/len(notFaces))
    
    return featureProb

def probImgGivenDigit(img, featureProb1, featureProb2, digit):
    
    features1 = getFeatures(img, 28, 28, '+')
    features2 = getFeatures(img, 28, 28, '#')
    sol = 1
    for i, feature in enumerate(features1):
        if feature==1:
            sol += math.log(featureProb1[digit][i])
        else:
            sol += math.log(1-featureProb1[digit][i])

    for i, feature in enumerate(features2):
        try:
            if feature==1:
                sol += math.log(featureProb2[digit][i])
            else:
                sol += math.log(1-featureProb2[digit][i])
        except:
           #print(featureProb2[digit][i])
            exit

    return sol

def probImgGivenFace(img, featureProb):
    
    features = getFeatures(img, 70, 60, '#')
    sol = 1
    for i, feature in enumerate(features):
        if feature==1:
            sol += math.log(featureProb[i])
        else:
            sol += math.log(1-featureProb[i])
    return sol


def probImgGivenNotFace(img, featureProb):  

    features = getFeatures(img, 70, 60, '#')
    sol = 1
    for i, feature in enumerate(features):
        if feature==1:
            sol += math.log(featureProb[i])
        else:
            sol += math.log(1-featureProb[i])
    return sol

def determineFace(img, faceFeatureProb, notFaceFeatureProb, probFace, probNotFace):

    probImgFace = probImgGivenFace(img, faceFeatureProb)
    probImgNotFace = probImgGivenNotFace(img, notFaceFeatureProb)

    probFaceGivenImg = probImgFace + math.log(probFace)
    probNotFaceGivenImg = probImgNotFace +math.log(probNotFace) 

    if(probFaceGivenImg > probNotFaceGivenImg):
        return True
    return False

def determineDigit(img, digitFeatureProb1, digitFeatureProb2, probDigit):
    maxScore = float('-inf')
    maxDigit = -1
    for i in range(10):
        probImgDigit = probImgGivenDigit(img, digitFeatureProb1, digitFeatureProb2, i)
        probDigitGivenImg = probImgDigit + math.log(probDigit[i])
        if(probDigitGivenImg > maxScore):
            maxScore = probDigitGivenImg
            maxDigit = i

    return maxDigit

def trainFaces(trainingData, trainingLabels, multiplier):
    newTrainingData, newTrainingLabels = cutData(trainingData, trainingLabels, multiplier * 0.1)
    faces = []
    notFaces = []
    for i, label in enumerate(newTrainingLabels):
        if(label == True):
            faces.append(newTrainingData[i])
        else:
            notFaces.append(newTrainingData[i])

    faceFeatureProb = getFaceFeaturesProb(faces)
    notFaceFeatureProb = getNotFaceFeatureProb(notFaces)
    probFace = len(faces)/(len(newTrainingLabels))

    return faceFeatureProb, notFaceFeatureProb, probFace, 1-probFace

def trainDigits(trainingData, trainingLabels, multiplier):
    newTrainingData, newTrainingLabels = cutData(trainingData, trainingLabels, multiplier * 0.1)

    digits = [[] for _ in range(10)]
    for i, label in enumerate(newTrainingLabels):
        digits[label].append(newTrainingData[i])

    digitFeatureProb1, digitFeatureProb2 = getDigitFeaturesProb(digits)

    probDigits = []
    for digit in digits:
        probDigits.append(len(digit)/len(newTrainingLabels))

    return digitFeatureProb1, digitFeatureProb2, probDigits

def testFaces(data, labels, faceFeatureProb, notFaceFeatureProb, probFace, probNotFace):
    count = 0
    for i,image in enumerate(data):
        count = count + 1 if determineFace(image, faceFeatureProb, notFaceFeatureProb, probFace, probNotFace) == labels[i] else count
    
    return count/len(labels)

def testDigits(data, labels, digitFeatureProb1, digitFeatureProb2, probDigits):
    count = 0
    for i,digit in enumerate(data):
        count = count + 1 if determineDigit(digit, digitFeatureProb1, digitFeatureProb2, probDigits) == labels[i] else count
    
    return count/len(labels)

def computeFaceAvgPredErr(testingData, testingLabels, trainingData, trainingLabels, multiplier):
    acc = []
    for _ in range(5):
        faceFeatureProb, notFaceFeatureProb, probFace, probNotFace = trainFaces(trainingData, trainingLabels, multiplier)    
        acc.append(testFaces(testingData, testingLabels, faceFeatureProb, notFaceFeatureProb, probFace, probNotFace))
    arr = np.array(acc)
    mean = np.mean(arr)
    std_dev = np.std(arr)
    return mean, std_dev

def computeDigitAvgPredErr(testingData, testingLabels, trainingData, trainingLabels, multiplier):
    acc = []
    for _ in range(5):
        digitFeatureProb1, digitFeatureProb2, probDigits = trainDigits(trainingData, trainingLabels, multiplier)    
        acc.append(testDigits(testingData, testingLabels, digitFeatureProb1, digitFeatureProb2, probDigits))
    arr = np.array(acc)
    mean = np.mean(arr)
    std_dev = np.std(arr)
    return mean, std_dev

def faceStatistics():
    trainingData = readFaceData("facedatatrain")
    trainingLabels = readFaceLabels("facedatatrainlabels")
    validationData = readFaceData("facedatavalidation")
    validationLabels = readFaceLabels("facedatavalidationlabels")
    testData = readFaceData("facedatatest")
    testLabels = readFaceLabels("facedatatestlabels")

    for multiplier in range(1,11):
        t1_start = perf_counter()
        faceFeatureProb, notFaceFeatureProb, probFace, probNotFace = trainFaces(trainingData, trainingLabels, multiplier) 
        t1_stop = perf_counter()  
        print("Elapsed time in seconds:"+str(t1_stop-t1_start))
        accuracy = testFaces(validationData, validationLabels, faceFeatureProb, notFaceFeatureProb, probFace, probNotFace)
        print(str(accuracy * 100) + "% accuracy for "+ str(multiplier * 10) + "% of the data")
        mean, std_dev = computeFaceAvgPredErr(testData, testLabels, trainingData, trainingLabels, multiplier)
        print("Mean accuracy: "+str(mean * 100) + "%\t Std Dev: "+ str(std_dev * 100))  

def digitStatistics():
    trainingData = readDigitData("trainingimages")
    trainingLabels = readDigitLabels("traininglabels")
    validationData = readDigitData("validationimages")
    validationLabels = readDigitLabels("validationlabels")
    testData = readDigitData("testimages")
    testLabels = readDigitLabels("testlabels")

    for multiplier in range(1,11):
        t1_start = perf_counter()
        digitFeatureProb1, digitFeatureProb2, probDigits = trainDigits(trainingData, trainingLabels, multiplier) 
        t1_stop = perf_counter()
        print("Elapsed time in seconds:"+str(t1_stop-t1_start))
        accuracy = testDigits(validationData, validationLabels, digitFeatureProb1, digitFeatureProb2, probDigits)
        print(str(accuracy * 100) + "% accuracy for "+ str(multiplier * 10) + "% of the data")
        mean, std_dev = computeDigitAvgPredErr(testData, testLabels, trainingData, trainingLabels, multiplier)
        print("Mean accuracy: "+str(mean * 100) + "%\t Std Dev: "+ str(std_dev * 100))    

#faceStatistics() 
digitStatistics()