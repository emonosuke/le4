from sklearn.datasets import load_wine, load_iris
import numpy as np

def class_mapping(x):
    if x == 1:
        return -1
    else:
        return 1

def load_data_iris():
    X, y = load_iris(return_X_y=True)
    X = X[y >= 1]
    y = y[y >= 1]
    y = map(class_mapping, y)
    npX = np.array(X)
    npy = np.array(list(y))
    npy = npy.astype(np.float64)
    return npX, npy

def load_data_wine():
    X, y = load_wine(return_X_y=True)
    X = X[y >= 1]
    y = y[y >= 1]
    y = map(class_mapping, y)
    npX = np.array(X)
    npy = np.array(list(y))
    npy = npy.astype(np.float64)
    return npX, npy
    