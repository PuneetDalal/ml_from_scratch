import numpy as np
def predict(X, w, b):
    return X @ w + b
X_test = np.array([
    [1],
    [2],
    [3]
])

w_test = np.array([[2]])

b_test = 1

predictions = predict(X_test, w_test, b_test)

print(predictions)