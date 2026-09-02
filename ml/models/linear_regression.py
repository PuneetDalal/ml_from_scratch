import numpy as np
import matplotlib.pyplot as plt
from utils import losses 
from utils import activations
def predict(X, w, b):
    return X @ w + b
def compute_gradients(X, y, y_pred):
    dw = X.T @ losses.mse_derivative(y_pred, y)
    db = np.sum(losses.mse_derivative(y_pred, y))
    return dw, db
def train(X, y, lr, epochs):

    w = np.zeros((X.shape[1], 1))
    b = 0.0

    loss_history = []

    for epoch in range(epochs):

        # 1. Make predictions
        y_pred = predict(X, w, b)

        # 2. Calculate and store loss
        loss = losses.mse(y_pred, y)
        loss_history.append(loss)

        # 3. Calculate gradients
        dw, db = compute_gradients(X, y, y_pred)

        # 4. Update parameters
        w -= lr * dw
        b -= lr * db

    return w, b, loss_history


if __name__ == "__main__":
    n = 100
    true_w = 0.5
    true_b = 2.9
    noise_scale = 0.5

    X = np.random.randn(n, 1)
    y = true_w * X + true_b + np.random.randn(n, 1) * noise_scale

    w, b, loss_history = train(X, y, lr=0.01, epochs=1000)

    print("True w:", true_w)
    print("Trained w:", w)
    print("True b:", true_b)
    print("Trained b:", b)
    
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.show()