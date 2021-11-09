import sys
import os
# Import from sibling directory ..\api
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import Utils
import numpy as np

# main method to run
def run(iteration, kind = "train"):
    x_train, y_train = Utils.load_mnist(kind)

    weights = np.zeros(len(x_train[0]) * 10)
    mistake_list = [] # mistake of each iteration
    train_accuracy_list = []; # training set accuracy of each iteration
    test_accuracy_list = []; # test set accuracy of each iteration
    
    for i in range(iteration):
        print("PA Iteration: "+str(i))
        mistake = 0
        correct = 0
        for j in range(len(x_train)):
            allScores = []
            for k in range(10):
                allScores.append(np.dot(weights, F(x_train[j], k)))
            yScore = np.argmax(allScores); # find best label

            if yScore != y_train[j]: # wrong label
                mistake += 1
                weights = updateWeights(weights, x_train[j], y_train[j], yScore)
            else: # correct label
                correct += 1
        mistake_list.append(mistake)
        train_accuracy_list.append(correct / len(x_train))
        test_accuracy_list.append(test(weights))

    return (weights, mistake_list, train_accuracy_list, test_accuracy_list)

# updates weights with learning rate 1
def updateWeights(weights, xt, yt, y):
    learning_rate = (1-(np.dot(weights, F(xt, yt)) - np.dot(weights, F(xt, y)))) / (np.linalg.norm(np.subtract(F(xt, yt), F(xt, y))))
    newWeights = weights + np.dot(learning_rate, np.subtract(F(xt, yt), F(xt, y)))
    return newWeights

# run on test data with given weights
def test(weights):
    x_test, y_test = Utils.load_mnist("t10k")

    mistake = 0
    correct = 0
    for i in range(len(x_test)):
        allScores = []
        for k in range(10):
            allScores.append(np.dot(weights, F(x_test[i], k)))
        yScore = np.argmax(allScores); # max score label
        if yScore != y_test[i]:
            mistake += 1
        else:
            correct += 1
    return correct / len(x_test)

# Weight-vector representation 2
def F(x, y):
    vector = np.zeros(len(x) * 10)
    for i in range(len(x)):
        vector[len(x) * y + i] = x[i]
    return vector

# used for part d incremental runs
def incrementRun(iteration, size):
    x_train, y_train = Utils.load_mnist("train")

    weights = np.zeros(len(x_train[0]) * 10)
    data_size = []
    test_accuracy_list = [];
    
    for i in range(iteration):
        print("Iteration: "+str(i))
        mistake = 0
        correct = 0
        for j in range(len(x_train)):
            allScores = []
            for k in range(10):
                allScores.append(np.dot(weights, F(x_train[j], k)))
            yScore = np.argmax(allScores); # max score label

            if yScore != y_train[j]:
                mistake += 1
                weights = updateWeights(weights, x_train[j], y_train[j], yScore)
            else:
                correct += 1

            if (j + 1) % size == 0:
                print((i+1) * (j+1))
                test_accuracy_list.append(test(weights))
                data_size.append((i+1) * (j+1))

    return (weights, test_accuracy_list)