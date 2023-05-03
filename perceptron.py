import numpy as np
import copy
import random
from time import perf_counter

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

def determineFace(weights, features, isFace, bias, train=False):
    scores = [weights[i] * features[0][i] for i in range(len(features[0]))]
    totalScore = sum(scores) + bias[0]
    alpha = 0.001
    if(totalScore < 0 and isFace):
        #fix weights
        if(train == True):
            bias[0] += 1
            for i in range(len(features[0])):
                weights[i] += (features[0][i]*alpha)
        return False
    
    if(totalScore >= 0 and not isFace):
        #fix weights
        if(train == True):
            bias[0] -= 1
            for i in range(len(features[0])):
                weights[i] -= (features[0][i]*alpha)
        return True
        
    return isFace

def trainDigits(trainingData, trainingLabels, multiplier, n, iters):
    newTrainingData, newTrainingLabels = cutData(trainingData, trainingLabels, multiplier * 0.1)
    features1 = []
    features2 = []
    for _, digit in enumerate(newTrainingData):
        features1.append(getFeatures(digit, 28, 28, n, '+'))
        features2.append(getFeatures(digit, 28, 28, n, '#'))
    features1 = np.array(features1)
    features2 = np.array(features2)
    features1 = features1.astype(np.float64)
    features2 = features2.astype(np.float64)
    bias = [0]
    weights = np.array([[[0] for _ in range(n**2)] for _ in range(10)])
    weights = weights.astype(np.float64)
    for _ in range(iters):
        for i, _ in enumerate(newTrainingData):
            determineDigit(weights, features1[i], features2[i], newTrainingLabels[i], bias, train=True)
    return weights, bias

def trainFaces(trainingData, trainingLabels, multiplier, n, iters):

    newTrainingData, newTrainingLabels = cutData(trainingData, trainingLabels, multiplier * 0.1)
    features = []
    for _, face in enumerate(newTrainingData):
        features.append(getFeatures(face, 70, 60, n, '#'))
    weights = [0 for _ in range(n**2)]
    bias = [0]
    for _ in range(iters):
        for i, _ in enumerate(newTrainingData):
            determineFace(weights, features[i], newTrainingLabels[i], bias, train=True)
    return weights, bias
    
def determineDigit(weights, features1, features2, digit, bias, train=False):
    maxScore = float('-inf')
    maxDigit = -1
    alpha = 0.1
    for j in range(10):
        scores1 = np.dot(features1, weights[j]) * 0.2
        scores2 = np.dot(features2, weights[j])
        totalScore = scores1 + scores2 + bias[0]
        #print(totalScore)
        if(totalScore > maxScore):
            maxScore = totalScore
            maxDigit = j

        if(train == True):
            if(totalScore < 0 and digit==j):
                #fix weights
                bias[0] += 1
                f1 = np.transpose(features1)
                f1 *= 0.2
                f2 = np.transpose(features2)
                newF = alpha * (f1 + f2)
                weights[j] += newF
            
            if(totalScore >= 0 and digit!=j):
                #fix weights
                bias[0] -= 1
                f1 = np.transpose(features1)
                f1 *= 0.2
                f2 = np.transpose(features2)
                newF = alpha * (f1 + f2)
                weights[j] -= newF
        
    return maxDigit

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

def getFaceAccuracy(data, labels, weights, features):
    count = 0
    for i, _ in enumerate(data):
        count = count + 1 if determineFace(weights, features[i], labels[i]) == labels[i] else count
    return count / len(labels)

def testDigits(data, labels, weights, features1, features2, bias):
    count = 0
    for i, _ in enumerate(data):
        count = count + 1 if determineDigit(weights, features1[i], features2[i], labels[i], bias) == labels[i] else count
    return count / len(labels)

def testFaces(data, labels, weights, features, bias):
    count = 0
    for i, _ in enumerate(data):
        count = count + 1 if determineFace(weights, features[i], labels[i], bias) == labels[i] else count
    return count / len(labels)

def getFeatures(data, height, width, n, delimiter):
    score = 0
    features = [[]]
    numRows = n
    rowPixels = height // numRows
    numCols = n
    colPixels = width // numCols
    prev1 = 0
    for _ in range(numRows):
        prev2 = 0
        for _ in range(numCols):
            for x in range(prev1, prev1 + rowPixels):
                for y in range(prev2, prev2 + colPixels):
                    if(data[x][y] == delimiter):
                        score += 1
            features[0].append(score)
            prev2 = prev2 + colPixels
        prev1 = prev1 + rowPixels
    return features

def computeDigitAvgPredErr(testingData, testingLabels, trainingData, trainingLabels, multiplier, n, iters, newFeatures1, newFeatures2):
    acc = []
    for _ in range(5):
        newWeights, newBias = trainDigits(trainingData, trainingLabels, multiplier, n, iters)    
        acc.append(testDigits(testingData, testingLabels, newWeights, newFeatures1, newFeatures2, newBias))
    arr = np.array(acc)
    mean = np.mean(arr)
    std_dev = np.std(arr)
    return mean, std_dev

def computeFaceAvgPredErr(testingData, testingLabels, trainingData, trainingLabels, multiplier, n, iters, newFeatures):
    acc = []
    for _ in range(5):
        newWeights, newBias = trainFaces(trainingData, trainingLabels, multiplier, n, iters)    
        acc.append(testFaces(testingData, testingLabels, newWeights, newFeatures, newBias))
    arr = np.array(acc)
    mean = np.mean(arr)
    std_dev = np.std(arr)
    return mean, std_dev
    
def getFaceStats(iters, n):
    trainingData = readFaceData("facedatatrain")
    trainingLabels = readFaceLabels("facedatatrainlabels")
    validationData = readFaceData("facedatavalidation")
    validationLabels = readFaceLabels("facedatavalidationlabels")
    testData = readFaceData("facedatatest")
    testLabels = readFaceLabels("facedatatestlabels")

    features = []
    for _, face in enumerate(trainingData):
        features.append(getFeatures(face, 70, 60, n, "#"))

    newFeatures = []
    for _, face in enumerate(validationData):
        newFeatures.append(getFeatures(face, 70, 60, n, "#"))

    newFeatures2 = []
    for _, face in enumerate(testData):
        newFeatures2.append(getFeatures(face, 70, 60, n, "#"))

    for multiplier in range(1,11):
        t1_start = perf_counter()
        newWeights, newBias = trainFaces(trainingData, trainingLabels, multiplier, n, iters)  
        t1_stop = perf_counter()  
        print("Elapsed time in seconds:"+str(t1_stop-t1_start))
        accuracy = testFaces(validationData, validationLabels, newWeights, newFeatures, newBias)
        print(str(accuracy * 100) + "% accuracy for "+ str(multiplier * 10) + "% of the data given "+str(iters)+" iterations")
        mean, std_dev = computeFaceAvgPredErr(testData, testLabels, trainingData, trainingLabels, multiplier, n, iters, newFeatures2)
        print("Mean accuracy: "+str(mean * 100) + "%\t Std Dev: "+ str(std_dev * 100))

def getDigitStats(iters, n):
    trainingData = readDigitData("trainingimages")
    trainingLabels = readDigitLabels("traininglabels")
    validationData = readDigitData("validationimages")
    validationLabels = readDigitLabels("validationlabels")
    testData = readDigitData("testimages")
    testLabels = readDigitLabels("testlabels")
    newFeatures1 = []
    newFeatures2 = []
    for _, digit in enumerate(validationData):
        newFeatures1.append(getFeatures(digit, 28, 28, n, '+'))
        newFeatures2.append(getFeatures(digit, 28, 28, n, '#'))
    newFeatures1 = np.array(newFeatures1)
    newFeatures2 = np.array(newFeatures2)
    newFeatures1 = newFeatures1.astype(np.float64)
    newFeatures2 = newFeatures2.astype(np.float64)

    newFeatures3 = []
    newFeatures4 = []
    for _, digit in enumerate(testData):
        newFeatures3.append(getFeatures(digit, 28, 28, n, '+'))
        newFeatures4.append(getFeatures(digit, 28, 28, n, '#'))
    newFeatures3 = np.array(newFeatures3)
    newFeatures4 = np.array(newFeatures4)
    newFeatures3 = newFeatures3.astype(np.float64)
    newFeatures4 = newFeatures4.astype(np.float64)

    for multiplier in range(1,11):
        t1_start = perf_counter()
        newWeights, newBias = trainDigits(trainingData, trainingLabels, multiplier, n, iters)
        t1_stop = perf_counter()  
        print("Elapsed time in seconds:"+str(t1_stop-t1_start))    
        accuracy = testDigits(validationData, validationLabels, newWeights, newFeatures1, newFeatures2, newBias)
        print(str(accuracy * 100) + "% accuracy for "+ str(multiplier * 10) + "% of the data given "+str(iters)+" iterations")
        mean, std_dev = computeDigitAvgPredErr(testData, testLabels, trainingData, trainingLabels, multiplier, n, iters, newFeatures3, newFeatures4)
        print("Mean accuracy: "+str(mean * 100) + "%\t Std Dev: "+ str(std_dev * 100))
# face statistics
#getFaceStats(iters = 1000, n=10)

# digit statistics
getDigitStats(iters = 1000, n=28)
