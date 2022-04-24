import numpy as np


def softmax(x):
    orignalShape = x.shape
    N = x.shape[0]

    if (len(x.shape)) > 1:
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((N, 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((N, 1))
    else:
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp

    assert x.shape == orignalShape
    return x


def sigmoid(x):
    return 1. / (1. + np.exp(-x))