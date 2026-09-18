import numpy as np
import matplotlib.pyplot as plt
from utils import activations as act
from utils import losses
def predict (X,w,b):
    return act.sigmoid(X@w+b)
def compute_gradient(X,y,a):
    dz= (a-y)/len(y)
    dw = X.T@dz
    db = np.sum(dz)
    return dw , db
def train (X,y,lr,epochs):
    w = np.zeros((X.shape[1], 1))
    b=0.0
    loss_history=[]
    for epoch in range(epochs):
        y_pred= predict(X,w,b)
        loss = losses.binary_cross_entropy(y_pred,y)
        loss_history.append(loss)
        dw,db=compute_gradient(X,y,y_pred)
        w -= lr * dw
        b -= lr * db

    return w, b, loss_history
