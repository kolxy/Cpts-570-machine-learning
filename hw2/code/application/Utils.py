import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def load_breast():
    df = pd.read_csv(PATH+"/breast-cancer-wisconsin.data", header=None, usecols=list(range(1, 11)))
    df = df[~df[df.isin(["?"])].any(axis=1)].astype(float)
    data = df.to_numpy()
    return data