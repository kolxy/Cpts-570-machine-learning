import sys
import os
# Import from sibling directory ..\api
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import Utils
import numpy as np

def run(iteration, kind = "train"):
    x_train, y_train = Utils.load_mnist(kind)

    y_train = Utils.get_binary_label(y_train)

    weights = np.zeros(len(x_train[0]))

    test_accuracy_list = [];
    
    c = 0
    for i in range(iteration):
        avgWeights = np.zeros(len(x_train[0]))
        mistake = 0
        correct = 0
        for i in range(len(x_train)):
            if np.sign(np.dot(x_train[i], weights)) != np.sign(y_train[i]):
                weights = updateWeights(weights, x_train[i], y_train[i])
                avgWeights = avgWeights + np.dot(y_train[i] * c, x_train[i])
            c += 1
        weights = weights - (avgWeights / c)
        test_accuracy_list.append(test(weights))

    return (weights, test_accuracy_list)

def updateWeights(weights, xt, yt):
    newWeights = np.add(weights, yt * xt)
    return newWeights

def test(weights):
    x_test, y_test = Utils.load_mnist("t10k")

    y_test = Utils.get_binary_label(y_test)

    mistake = 0
    correct = 0
    for i in range(len(x_test)):
            if np.sign(np.dot(x_test[i], weights)) != np.sign(y_test[i]):
                mistake += 1
            else:
                correct += 1
    return correct / len(x_test)