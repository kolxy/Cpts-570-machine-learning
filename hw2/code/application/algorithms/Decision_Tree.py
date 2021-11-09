import numpy as np
import datetime
import sys
import os
import math
import copy

from pandas.core.algorithms import value_counts
from pandas.io.sql import pandasSQL_builder
# Import from sibling directory ..\api
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import Utils
data = Utils.load_breast()
train_data = data[:int(len(data) * 0.7)]
validation_data = data[int(len(data) * 0.7):int(len(data) * 0.8)]
test_data = data[int(len(data) * 0.8):]


def ID3(data):
    root = buildTree(data)
    return root

def countLabels(data, index):
    labelCount = {}
    for row in data:
        label = row[index]
        if label not in labelCount.keys():
            labelCount[label] = 0
        labelCount[label] += 1
    
    return labelCount

def entropy(data):
    # count y
    labelCount = countLabels(data, -1)
    entropy = 0.0
    for label in labelCount:
        p = float(labelCount[label]) / float(len(data))
        entropy -= p * math.log2(p)

    return entropy

def splitData(data, column, value):
    less = []
    greater = []
    for row in data:
        if float(row[column]) < value:
            less.append(row)
        else:
            greater.append(row)
    return less, greater

def buildTree(data, level = 0):
    node = Node(data = data, level = level)

    yCount = countLabels(data, -1)
    # if pure node, return
    if len(yCount.keys()) <= 1:
        node.label = list(yCount.keys())[0]
        return node
    
    min_entropy = 1
    best_feature = -1
    best_value = -1
    less_data = []
    greater_data = []

    # iterate features
    for columnIndex in range(0, 9):

        # values for each feature
        valueCount = countLabels(data, columnIndex)
        allValues = list(valueCount.keys())
        # iterate feature values
        for i, value in enumerate(allValues):
            if (i < len(allValues) - 1):
                splitValue = float(value) + (float(allValues[i+1]) - float(value)) / 2

                less, greater = splitData(data, columnIndex, splitValue)
                pless = float(len(less)) / float(len(data))
                current_enctropy = pless * entropy(less) + (1 - pless) * entropy(greater)
                # find lowest entropy = highest info gain
                if current_enctropy < min_entropy:
                    min_entropy = current_enctropy
                    best_feature = columnIndex
                    best_value = splitValue
                    less_data = less
                    greater_data = greater
    
    node.attribute = best_feature
    node.value = best_value

    node.less = buildTree(less_data, level+1)
    node.greater = buildTree(greater_data, level+1)
    
    return node

def printNode(node):
    ret = ""
    for i in range(node.level):
        ret += "   "
    if (node.label):
        ret += "label: " + str(node.label)
    else:
        ret += str(node.attribute) + ", " + str(node.value)
    print(ret)

    if (node.less):
        printNode(node.less)
    if (node.greater):
        printNode(node.greater)

def classify(root, input):
    correct = 0
    for data in input:
        if data[-1] == predict(root, data):
            correct += 1
    return correct / len(input)

def predict(node, data):
    if (node.label):
        return node.label

    if float(data[node.attribute]) < node.value:
        return predict(node.less, data)
    else:
        return predict(node.greater, data)

# takes current node and root node
def prune(node, root):
    # current node is leaf
    if node.label:
        return

    lessLeaf = True
    greaterLeaf = True
    # left is not a leaf node
    if (node.less != None) & (node.less.label == None):
        lessLeaf = False
        prune(node.less, root)
    # greater is not a leaf node
    if (node.greater != None) & (node.greater.label == None):
        greaterLeaf = False
        prune(node.greater, root)
    
    # if child node has leaf node
    if lessLeaf | greaterLeaf:
        beforeAcc = classify(root, validation_data)
        node.label = majority(node.data)
        afterAcc = classify(root, validation_data)
        # prune fail
        if beforeAcc >= afterAcc:
            node.label = None
    
    return root

def majority(data):
    labelCount = countLabels(data, -1)
    res = max(labelCount, key=labelCount.get)
    return res

class Node():
    def __init__(self, data = [], attribute = None, value = None, less = None, greater = None, label = None, level = 0):
        self.data = data
        self.attribute = attribute
        self.value = value
        self.less = less
        self.greater = greater
        self.label = label
        self.level = level