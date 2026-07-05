# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import classification_report

def load_data(path):
    data = np.genfromtxt(path, delimiter=',')
    return data[:, 1:], data[:, [0]]

def init_params(n_input=22, n_hidden=8, n_output=1, seed=0):
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-0.5, 0.5, size=(n_input, n_hidden))
    w2 = rng.uniform(-0.5, 0.5, size=(n_hidden, n_output))
    b1 = np.zeros((1, n_hidden))
    b2 = np.zeros((1, n_output))
    
    return {"W1": w1, "b1": b1, "W2": w2, "b2": b2}


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X, params):
    z1 = X @ params["W1"] + params["b1"]
    a1 = sigmoid(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = sigmoid(z2)

    return (a2, (z1, a1, z2, a2))

def compute_loss(y, A2):
    A2 = np.clip(A2, 1e-12, 1 - 1e-12)
    return np.mean(-(y * np.log(A2) + (1 - y) * np.log(1 - A2)))

def backward(X, y, cache, params):
    Z1, A1, Z2, A2 = cache
    W1 = params["W1"]
    W2 = params["W2"]
    n = X.shape[0]
    d2 = A2 - y
    dW2 = np.transpose(A1) @ d2 / n
    db2 = np.mean(d2, axis=0, keepdims=True)
    d1 = (d2 @ np.transpose(W2)) * A1 * (1 - A1)
    dW1 = np.transpose(X) @ d1 / n
    db1 = np.mean(d1, axis=0, keepdims=True)

    return (dW1, db1, dW2, db2)

def update(params, grads, lr):
    dW1, db1, dW2, db2 = grads
    params["W1"] = params["W1"] - lr * dW1
    params["b1"] = params["b1"] - lr * db1
    params["W2"] = params["W2"] - lr * dW2
    params["b2"] = params["b2"] - lr * db2

    return params


def train(X, y, params, lr=0.5, epochs=2000):
    for i in range(epochs):
        A2, cache = forward(X, params)
        loss = compute_loss(y, A2)
        gradients = backward(X, y, cache, params)
        params = update(params, gradients, lr)

        if  i % 100 == 0:
            print(f"epoch {i} loss {loss}")

    return params

def evaluate(X, y, params):
    A2, cache = forward(X, params)
    predictions = np.where(A2 >= 0.5, 1, 0)

    print(classification_report(y, predictions))

    return np.mean(predictions == y) * 100

if __name__ == "__main__":
    X_train, y_train = load_data("spect_train.txt")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape) 
    values, counts = np.unique(y_train, return_counts=True)
    print("label balance:", dict(zip(values.astype(int), counts)))

    params = init_params()
    A2, cache = forward(X_train, params)
    print("starting loss:", compute_loss(y_train, A2))
    params = train(X_train, y_train, params)
    print("train accuracy:", evaluate(X_train, y_train, params), "%")
    X_test, y_test = load_data("spect_test.txt")
    print("test accuracy:", evaluate(X_test, y_test, params), "%")