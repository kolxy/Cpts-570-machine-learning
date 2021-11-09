import os
import gzip
import numpy as np
import matplotlib.pyplot as plt

PATH = os.getcwd() + "/application/data"

# copied straight of data origin github
def load_mnist(kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(PATH,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(PATH,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

# transform multi-class y to binary
def get_binary_label(y_train):
    mydata = []
    for item in y_train:
        if (item % 2) == 0:
            mydata.append(1)
        else:
            mydata.append(-1)
    return mydata

# common method for graphing
def generate_graph(x, y, title, xlabel, ylabel):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()