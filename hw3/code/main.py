from typing import Set
import Utils
import numpy as np
from sklearn.naive_bayes import MultinomialNB

def main():
    Utils.startTime()

    xTrainWords = removeStopWords(Utils.xTrain, Utils.stop)
    xTestWords = removeStopWords(Utils.xTest, Utils.stop)
    dictList = buildDict(xTrainWords)
    wordMatrix = buildMatrix(xTrainWords, dictList)
    
    # predict
    correct = 0
    for i in range(len(xTrainWords)):
        predicted = predict(xTrainWords[i], dictList, wordMatrix, Utils.yTrain)
        if predicted == Utils.yTrain[i]:
            correct += 1
    print("Train:", correct / len(Utils.yTrain))


    correct = 0
    for i in range(len(xTestWords)):
        predicted = predict(xTestWords[i], dictList, wordMatrix, Utils.yTrain)
        if predicted == Utils.yTest[i]:
            correct += 1
    print("Test:", correct / len(Utils.yTest))


    # sklearn experiment for verification
    clf = MultinomialNB()
    clf.fit(wordMatrix, Utils.yTrain)
    print("Train sklearn:", clf.score(wordMatrix, Utils.yTrain))
    print("Test sklearn:", clf.score(buildMatrix(xTestWords, dictList), Utils.yTest))

    Utils.endTime()

# remove stop words from strings and convert strings to words
def removeStopWords(data, stop):
    result = []
    for line in data:
        newWords = []
        words = line.split()

        for w in words:
            if w not in stop:
                newWords.append(w)

        result.append(newWords)
    return result

# build a list of words using dictionary
def buildDict(wordList):
    wordSet = set()
    for line in wordList:
        for word in line:
            wordSet.add(word)
    
    result = list(wordSet)
    result.sort()
    return result

# generate feature vectors
def buildMatrix(wordList, dictList):
    result = []
    for line in wordList:
        vec = [0] * len(dictList)
        for word in line:
            if word in dictList:
                vec[dictList.index(word)] = 1
        result.append(vec)
    
    # use numpy matrix for faster access later
    return np.array(result)

def predict(target, dictList, wordMatrix, labels):
    Py0 = Pyx('0', target, dictList, wordMatrix, labels)
    Py1 = Pyx('1', target, dictList, wordMatrix, labels)
    return '0' if Py0 > Py1 else '1'

# P(y | x) = P(y) * P(x1 | y) * ... * P(xn | y)
def Pyx(y, line, dictList, matrix, labels):
    result = Py(y, labels)
    for word in line:
        result *= Pxy(word, dictList, matrix, y, labels)
    return result

# P(y = 0 or 1)
def Py(y, labels):
    count = 0
    for label in labels:
        if y == label:
            count+=1
    return count / len(labels)

# P(x | y)
def Pxy(x, dictList, matrix, y, labels):
    countX = 0
    countY = 0
    if x in dictList:
        column = matrix[:,dictList.index(x)]
        for i in range(len(column)):
            # y match
            if labels[i] == y:
                countY += 1
                # x is 1
                if column[i]:
                    countX += 1
        # Laplace smoothing
        return (countX + 1) / (countY + 2)
    # the word is not in dict then return 1
    return 1

if __name__ == "__main__":
    main()