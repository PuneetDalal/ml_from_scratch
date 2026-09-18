import numpy as np
def sigmoid(x):
    return 1/(1+(np.e)**(-x))
def sigmoid_derivative(x):
    s= 1/(1+(np.e)**(-x))
    return s*(1-s)
def relu(x):
    return np.maximum(0,x)
def relu_derivative(x):
    if (x<0) :
        return 0
    return 1