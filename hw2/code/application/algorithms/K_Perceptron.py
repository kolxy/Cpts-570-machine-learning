import sys
import os
# Import from sibling directory ..\api
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import Utils
import numpy as np
import datetime

# init dataset
x_train, y_train = Utils.load_mnist()
x_validation = x_train[48000:]
y_validation = y_train[48000:]
x_train = x_train[:48000]
y_train = y_train[:48000]
x_test, y_test = Utils.load_mnist("t10k")
X = {
    "train": x_train,
    "validation": x_validation,
    "test": x_test
}
Y = {
    "train": y_train,
    "validation": y_validation,
    "test": y_test
}

# main method to run
def run(degree, iteration = 5):
    print("------------------Degree", str(degree), "------------------")
    # 10 x 48000
    alpha_table = np.zeros(shape=(10, len(x_train)))
    mistake_list = [] # mistake of each iteration
    
    for it in range(iteration):
        print("Iteration: "+str(it))
        mistake = 0
        correct = 0
        for i in range(len(x_train)):
            weights = np.zeros(len(alpha_table))
            # labels(10)
            for label in range(len(alpha_table)):
                # training size
                for j in range(len(x_train)):
                    # dot n x n costs a lot, save time on 0
                    if alpha_table[label][j] != 0:
                        weights[label] += alpha_table[label][j] * ((np.dot(x_train[i], x_train[j])+1) ** degree)
            yScore = np.argmax(weights); # find best label

            if yScore != y_train[i]: # prediction wrong
                mistake += 1
                alpha_table[yScore][i] -= 1
                alpha_table[y_train[i]][i] += 1
            else:
                correct += 1
        mistake_list.append(mistake)
        print(mistake)
    training_score = test(alpha_table, degree, "train")
    validation_score = test(alpha_table, degree, "validation")
    testing_score = test(alpha_table, degree, "test")
    return (mistake_list, training_score, validation_score, testing_score)

# run on test data with given weights
def test(alpha_table, degree, mode):
    print(mode, datetime.now().strftime("%Y/%m/%d, %H:%M:%S"))
    correct = 0
    x = X[mode]
    y = Y[mode]
    for i in range(len(x)):
        weights = np.zeros(len(alpha_table))
        # labels(10)
        for label in range(len(alpha_table)):
            # number of training data
            for j in range(len(alpha_table[label])):
                # save time on 0
                if alpha_table[label][j] != 0:
                    weights[label] += alpha_table[label][j] * ((np.dot(x[i], x_train[j])+1) ** degree)
        yScore = np.argmax(weights); # find best label
        if yScore == y[i]:
            correct += 1
    
    print(correct / len(x))
    return correct / len(x)
