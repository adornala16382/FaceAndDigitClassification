import copy

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

def determineFace(weights, features, isFace, train=False):
    scores = ([weights[i] * feature for i, feature in enumerate(features)])
    totalScore = sum(scores)
    if(totalScore < 0 and isFace):
        #fix weights
        if(train == True):
            for i in range(len(weights)):
                weights[i] = weights[i] + features[i]
        return False
    
    if(totalScore >= 0 and not isFace):
        #fix weights
        if(train == True):
            for i in range(len(weights)):
                weights[i] = weights[i] - features[i]
        return True
        
    return isFace

def determineDigit(weights, features1, features2, digit, train=False):
    maxScore = float('-inf')
    maxDigit = -1
    for j in range(10):
        scores1 = ([weights[j][i] * feature * 0.2 for i, feature in enumerate(features1)])
        scores2 = ([weights[j][i] * feature for i, feature in enumerate(features2)])
        totalScore = sum(scores1) + sum(scores2)
        if(totalScore > maxScore):
            maxScore = totalScore
            maxDigit = j
        if(totalScore < 0 and digit==j):
            #fix weights
            if(train == True):
                for i in range(len(weights[j])):
                    weights[j][i] = weights[j][i] + ((features1[i]*0.2) + features2[i])
        
        if(totalScore >= 0 and digit!=j):
            #fix weights
            if(train == True):
                for i in range(len(weights[j])):
                    weights[j][i] = weights[j][i] - ((features1[i]*0.2) + features2[i])
    return maxDigit

def cutData(data, percentData):
    newData = []
    numData = len(data)
    newNumData = int(numData * percentData)
    for i in range(newNumData):
        newData.append(copy.deepcopy(data[i]))
    
    return newData

def getFaceAccuracy(data, labels, weights, features):
    count = 0
    for i, _ in enumerate(data):
        count = count + 1 if determineFace(weights, features[i], labels[i]) == labels[i] else count
    return count / len(labels)

def getDigitAccuracy(data, labels, weights, features1, features2):
    count = 0
    for i, _ in enumerate(data):
        count = count + 1 if determineDigit(weights, features1[i], features2[i], labels[i]) == labels[i] else count
    return count / len(labels)

def printData(data):
    for i,face in enumerate(data):
        for row in face:
            print(row)
        tmp = ""
        for j in range(20):
            tmp += str(i+1) + "   "
        print(tmp)

def getFeatures(data, height, width, n, delimiter):
    score = 0
    features = []
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
            features.append(score)
            prev2 = prev2 + colPixels
        prev1 = prev1 + rowPixels

    return features

def getFaceStats(trainingData, trainingLabels, validationData, validationLabels, iters, n, delimiter):
    features = []
    for _, face in enumerate(trainingData):
        features.append(getFeatures(face, 70, 60, n, delimiter))
    print(len(features))

    newFeatures = []
    for _, face in enumerate(validationData):
        newFeatures.append(getFeatures(face, 70, 60, n, delimiter))
    print(len(newFeatures))

    for multiplier in range(1,11):
        newTrainingData = cutData(trainingData, multiplier * 0.1)
        weights = [0 for _ in range(n**2)]
        for _ in range(iters):
            for i, _ in enumerate(newTrainingData):
                determineFace(weights, features[i], trainingLabels[i], train=True)
    
        accuracy = getFaceAccuracy(validationData, validationLabels, weights, newFeatures)
        print(str(accuracy * 100) + "% accuracy for "+ str(multiplier * 10) + "% of the data")

def getDigitStats(trainingData, trainingLabels, validationData, validationLabels, iters, n, delimiters):
    features1 = []
    features2 = []
    for _, digit in enumerate(trainingData):
        features1.append(getFeatures(digit, 28, 28, n, delimiters[0]))
        features2.append(getFeatures(digit, 28, 28, n, delimiters[1]))
    print(len(features1))
    print(len(features2))

    newFeatures1 = []
    newFeatures2 = []
    for _, digit in enumerate(validationData):
        newFeatures1.append(getFeatures(digit, 28, 28, n, delimiters[0]))
        newFeatures2.append(getFeatures(digit, 28, 28, n, delimiters[1]))
    print(len(newFeatures1))
    print(len(newFeatures2))

    for multiplier in range(1,11):
        newTrainingData = cutData(trainingData, multiplier * 0.1)
        weights = [[0 for _ in range(n**2)] for _ in range(10)]
        for _ in range(iters):
            for i, _ in enumerate(newTrainingData):
                determineDigit(weights, features1[i], features2[i], trainingLabels[i], train=True)
    
        accuracy = getDigitAccuracy(validationData, validationLabels, weights, newFeatures1, newFeatures2)
        print(str(accuracy * 100) + "% accuracy for "+ str(multiplier * 10) + "% of the data")

trainingData = readFaceData("facedatatrain")
trainingLabels = readFaceLabels("facedatatrainlabels")
validationData = readFaceData("facedatavalidation")
validationLabels = readFaceLabels("facedatavalidationlabels")
# face statistics
#getFaceStats(trainingData, trainingLabels, validationData, validationLabels, iters = 1000, n=10, delimiter='#')

trainingData = readDigitData("trainingimages")
trainingLabels = readDigitLabels("traininglabels")
validationData = readDigitData("validationimages")
validationLabels = readDigitLabels("validationlabels")
# digit statistics
getDigitStats(trainingData, trainingLabels, validationData, validationLabels, iters = 10, n=28, delimiters = ['+','#'])
