import numpy as np 
def mse(true,prediction):
    return np.mean((np.array(true)-np.array(prediction))**2)
def mse_derivative(prediction,true):
    return (2*(prediction-true)/len(true))
def binary_cross_entropy(prediction,true):
    return -1*(np.mean(true*np.log(prediction)+(1-true)*np.log(1-prediction)))
def bce_derivative(prediction, true):
    return (prediction - true) / (prediction * (1 - prediction) * len(true))