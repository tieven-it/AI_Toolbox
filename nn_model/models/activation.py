import numpy as np

# 激活函数
def relu(Z):

    A = np.maximum(0, Z)

    return A


def leaky_relu(Z):

    A = np.ones(Z.shape)
    A[Z < 0] = 0.01
    A = A * Z

    return A


def sigmoid(Z):

    A = 1/(1+np.exp(-Z))

    return A


def tanh(Z):

    A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + (np.exp(-Z)))

    return A

# 激活函数的导数
def relu_gradient(Z):

    A = np.ones(Z.shape)
    A[Z < 0] = 0

    return A


def leaky_relu_gradient(Z):

    A = np.ones(Z.shape)
    A[Z < 0] = 0.01

    return A


def sigmoid_gradient(Z):

    A = sigmoid(Z) * (1 - sigmoid(Z))

    return A


def tanh_gradient(Z):

    A = 1 - np.power(tanh(Z), 2)

    return A
