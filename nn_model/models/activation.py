import numpy as np


def relu(Z):

    A = np.maximum(0, Z)

    return A


def relu_gradient(Z):

    s = np.ones(Z.shape)
    s[Z < 0] = 0

    return s


def leaky_relu(Z):

    A = np.ones(Z.shape)
    A[Z < 0] = 0.01
    A = A * Z

    return A


def leaky_relu_gradient(Z):

    s = np.ones(Z.shape)
    s[Z < 0] = 0.01

    return s


def sigmoid(Z):

    A = 1/(1+np.exp(-Z))

    return A


def sigmoid_gradient(Z):

    s = sigmoid(Z) * (1 - sigmoid(Z))

    return s


def tanh(Z):

    A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + (np.exp(-Z)))

    return A


def tanh_gradient(Z):

    s = 1 - np.power(tanh(Z), 2)

    return s


def softmax(Z):
    
    t = np.exp(Z)
    s = np.sum(t, 0)
    A = t / s
    
    return A


def softmax_gradient(Z):
    
    s = Z - Z ** 2
    
    return s