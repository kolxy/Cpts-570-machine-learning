from datetime import datetime
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn
import os

from . import Utils
from .algorithms import K_Perceptron, Decision_Tree

PATH_ASSETS = os.getcwd() + "/application/assets"

# Orchastra inputs to methods
def run():
    functions = {
        "1a": p1a,
        "1b": p1b,
        "1c": p1c,
        "2": p2,
        "3": p3
    }
    print("Choose problem (example: 1a):")
    command = input()
    if command in functions:
        print("Executing " + command + " ...")
        date_time = datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        print("Starting Time: ", date_time)

        functions[command]()

        date_time = datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
        print("Finishing Time: ", date_time)
    else:
        print("Invalid command, now exiting...")

# Common method for running SVM
def SVM(C = 1, kernel = "rbf", degree = 3, max_iter = -1):
    # data initiation
    print("C: ", C)
    x_train, y_train = Utils.load_mnist()
    x_validation = x_train[48000:]
    y_validation = y_train[48000:]
    x_train = x_train[:48000]
    y_train = y_train[:48000]
    x_test, y_test = Utils.load_mnist("t10k")

    clf = SVC(C = C, kernel = kernel, degree = degree, max_iter = max_iter)
    clf.fit(x_train, y_train)

    training_score = clf.score(x_train, y_train)
    validation_score = clf.score(x_validation, y_validation)
    test_score = clf.score(x_test, y_test)
    support_vectors = clf.support_vectors_

    print(training_score, ",", validation_score, ",", test_score)

    return (training_score, validation_score, test_score, support_vectors)

def p1a():
    C = []
    training_scores = []
    validation_scores = []
    testing_scores = []
    for i in range(-4, 5):
        currentC = pow(10, i)
        training_score, validation_score, testing_score, support_vectors = SVM(
            C = currentC,
            kernel = "linear",
            max_iter = -1
        )
        # add C parameter
        C.append(str(currentC))
        training_scores.append(training_score)
        validation_scores.append(validation_score)
        testing_scores.append(testing_score)

    print(C)
    print(training_scores)
    print(validation_scores)
    print(testing_scores)

    plt.plot(C, training_scores, label = "Training Scores")
    plt.plot(C, validation_scores, label = "Validation Scores")
    plt.plot(C, testing_scores, label = "Testing Scores")
    plt.title("Accuracy for different C parameters")
    plt.xlabel("C Parameter")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(PATH_ASSETS + "/1a.png")
    return

def p1b():
    C = 0.0001 # result of 1a
    xTrain, yTrain = Utils.load_mnist()
    xTest, yTest = Utils.load_mnist("t10k")
    clf = SVC(
        C = C,
        kernel = "linear", 
        max_iter = -1)
    clf.fit(xTrain, yTrain)
    print("Best C is: ", C)
    
    yTest_i = clf.predict(xTest) # predicted labels

    # Because we have to use predicted result for confusion matrix
    # using metrics.accuracy_score() becomes a better approach then SVC.score()
    score = metrics.accuracy_score(yTest, yTest_i)
    print("Accuracy: ", score)

    confusion = metrics.confusion_matrix(yTest, yTest_i)
    print("confusion_matrix: \n", confusion)

    # Heatmap of confusion matrix with Seaborn
    # Uncomment this section to plot !!!
    # sn.set()
    # sn.heatmap(confusion, annot=True, fmt="d")
    # plt.yticks(rotation=0)
    # plt.tick_params(axis='both', which='major', labelbottom = False, bottom=False, top = False, labeltop=True)
    # plt.show()
    return

def p1c():
    degrees = []
    training_scores = []
    validation_scores = []
    testing_scores = []
    support_vectors = []
    for i in range(1, 5):
        kernel = "poly"
        if i == 1:
            kernel = "linear"

        training_score, validation_score, testing_score, support_vectors_ = SVM(
            C = 0.1,
            kernel = kernel,
            max_iter = -1,
            degree = i
        )
        # add C parameter
        degrees.append(str(i))
        print("degree: ", i)
        training_scores.append(training_score)
        validation_scores.append(validation_score)
        testing_scores.append(testing_score)
        support_vectors.append(len(support_vectors_))
        print(support_vectors_)

    print("degrees: ", degrees)
    print("Training score: ", training_scores)
    print("Validation score: ", validation_scores)
    print("Testing score: ", testing_scores)
    print("Support Vectors: ", support_vectors)

    # convert to string for better visual
    for i in range(len(degrees)):
        degrees[i] = str(degrees[i])

    plt.plot(degrees, training_scores, label = "Training Scores")
    plt.plot(degrees, validation_scores, label = "Validation Scores")
    plt.plot(degrees, testing_scores, label = "Testing Scores")
    plt.title("Accuracy for different kernel degrees")
    plt.xlabel("degrees")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(PATH_ASSETS + "/1c.png")

    # I plotted graph separately with data from previous print("Supporting_vectors:")
    # Therefore no code for plotting support vectors here, but graphs are in PDF
    return

def p2():
    degree = 2 # best poly degree from problem 1

    mistake, training, validation, testing = K_Perceptron.run(degree)

    print("Mistakes: ", mistake)
    print("Training:", training)
    print("Validation:", validation)
    print("Testing:", testing)

    iteration = ["1", "2", "3", "4", "5"]

    plt.plot(iteration, mistake)
    plt.title("Mistakes per iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Mistakes")
    plt.savefig("application/assets/2.png")
    return

def p3():
    # load data
    data = Utils.load_breast()
    train_data = data[:int(len(data) * 0.7)]
    validation_data = data[int(len(data) * 0.7):int(len(data) * 0.8)]
    test_data = data[int(len(data) * 0.8):]

    tree = Decision_Tree.ID3(train_data)

    # print result
    train_accuracy = Decision_Tree.classify(tree, train_data)
    validation_accuracy = Decision_Tree.classify(tree, validation_data)
    test_accuracy = Decision_Tree.classify(tree, test_data)
    print("Training: ", train_accuracy)
    print("Validation: ", validation_accuracy)
    print("Testing: ", test_accuracy)
    print("--------------prune---------------")
    
    prunedTree = Decision_Tree.prune(tree, tree)
    # print result
    train_accuracy = Decision_Tree.classify(prunedTree, train_data)
    validation_accuracy = Decision_Tree.classify(prunedTree, validation_data)
    test_accuracy = Decision_Tree.classify(prunedTree, test_data)
    print("Training: ", train_accuracy)
    print("Validation: ", validation_accuracy)
    print("Testing: ", test_accuracy)
    return
