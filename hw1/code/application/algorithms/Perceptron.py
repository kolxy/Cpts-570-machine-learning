import sys
import os
# Import from sibling directory ..\api
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import Utils
import numpy as np

# main method to run
def run(iteration, kind = "train"):
    x_train, y_train = Utils.load_mnist(kind)

    y_train = Utils.get_binary_label(y_train)

    weights = np.zeros(len(x_train[0]))
    mistake_list = [] # mistake of each iteration
    train_accuracy_list = []; # training set accuracy of each iteration
    test_accuracy_list = []; # test set accuracy of each iteration
    
    for i in range(iteration):
        mistake = 0
        correct = 0
        for j in range(len(x_train)):
            if np.sign(np.dot(x_train[j], weights)) != np.sign(y_train[j]):
                mistake += 1
                weights = updateWeights(weights, x_train[j], y_train[j])
            else:
                correct += 1
        mistake_list.append(mistake)
        train_accuracy_list.append(correct / len(x_train))
        test_accuracy_list.append(test(weights))

    return (weights, mistake_list, train_accuracy_list, test_accuracy_list)

# updates weights with learning rate 1
def updateWeights(weights, xt, yt):
    newWeights = np.add(weights, yt * xt)
    return newWeights

# run on test data with given weights
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

# used for part d incremental runs
def incrementRun(iteration, size):
    x_train, y_train = Utils.load_mnist("train")

    y_train = Utils.get_binary_label(y_train)

    weights = np.zeros(len(x_train[0]))
    
    data_size = []
    test_accuracy_list = []
    
    for i in range(iteration):
        mistake = 0
        correct = 0
        for j in range(len(x_train)):
            if np.sign(np.dot(x_train[j], weights)) != np.sign(y_train[j]):
                mistake += 1
                weights = updateWeights(weights, x_train[j], y_train[j])
            else:
                correct += 1
            
            if (j + 1) % size == 0:
                print((i+1) * (j+1))
                test_accuracy_list.append(test(weights))
                data_size.append((i+1) * (j+1))
    return (weights, data_size, test_accuracy_list)