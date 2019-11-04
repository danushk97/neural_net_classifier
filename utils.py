import numpy as np

def relu_derivative(da, cache):
    z = cache
    dz = np.array(da, copy=True)

    dz[z <= 0] = 0

    return dz

def sigmoid_derivative(da, cache):
    z = cache

    a = 1 / (1+np.exp(z))
    dz = da * a * (1 - a)

    return dz