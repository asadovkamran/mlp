# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

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

def cross_validate(X, y, n_hidden, k=5, epochs=2000):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    scores = []

    for train_idx, val_idx in skf.split(X, y.ravel()):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        params = init_params(n_hidden=n_hidden)
        params_trained = train(X_tr, y_tr, params)
        A2, _ = forward(X_val, params_trained)
        pred = np.where(A2 >= 0.5, 1, 0)
        scores.append(f1_score(y_val.ravel(), pred.ravel(), average='macro', zero_division=0))
        

    return np.mean(scores), np.std(scores)

def aggregate(params):
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    rng = np.random.default_rng()
    W1 = np.column_stack((W1, rng.uniform(-0.01, 0.01, size=(W1.shape[0], 1))))
    b1 = np.hstack((b1, np.zeros((1, 1))))
    W2 = np.vstack((W2, np.zeros((1, 1))))

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def reduce(params):
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    i = np.argmin(np.abs(params["W2"]))
    W1 = np.delete(W1, i, axis=1)
    b1 = np.delete(b1, i, axis=1)
    W2 = np.delete(W2, i, axis=0)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def optimize(params, X, y, baseline_loss, tolerance = 0.02):
    while params["W2"].shape[0] > 1:
        trial =  reduce(params)
        trial = train(X, y, trial, epochs=1000)
        A2, _ = forward(X, trial)
        new_loss = compute_loss(y, A2)

        if new_loss <= baseline_loss + tolerance:
            params = trial
            print(f"pruned to {params["W2"].shape[0]} neurons, loss: {new_loss:.4f}")
        else:
            print(f"cancel pruning: cut couldn't recover. new_loss: {new_loss:.4f}")
            break
        print(f"final shape: {params["W2"].shape[0]}")

    return params

def grow(params, X, y, max_neurons=20, min_improvement=0.002):
    A2, _ = forward(X, params)
    current_loss = compute_loss(y, A2)

    while params["W2"].shape[0] < max_neurons:
        trial = aggregate(params)
        trial = train(X, y, trial, epochs=5000)
        A2, _ = forward(X, trial)
        new_loss = compute_loss(y, A2)

        if new_loss < current_loss - min_improvement:
            params = trial
            current_loss = new_loss
            print(f"grew to {params['W2'].shape[0]} neurons, loss: {new_loss:.4f}")
        else:
            print(f"stopping: extra neuron didn't help. loss: {new_loss:.4f}")
            break

    return params

def train(X, y, params, lr=0.5, epochs=4000):
    for i in range(epochs):
        A2, cache = forward(X, params)
        loss = compute_loss(y, A2)
        gradients = backward(X, y, cache, params)
        params = update(params, gradients, lr)

    return params

def evaluate(X, y, params):
    A2, cache = forward(X, params)
    predictions = np.where(A2 >= 0.2, 1, 0)

    print(classification_report(y, predictions))

    return np.mean(predictions == y) * 100

if __name__ == "__main__":
    # X_train, y_train = load_data("spect_train.txt")
    # X_test, y_test = load_data("spect_test.txt")
    # print("X_train shape:", X_train.shape)
    # print("y_train shape:", y_train.shape) 
    # values, counts = np.unique(y_train, return_counts=True)
    # print("label balance:", dict(zip(values.astype(int), counts)))

    X, y = make_moons(n_samples=300, noise=0.2, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    params = train(X_train, y_train, init_params(n_input=2, n_hidden=1))
    params = grow(params, X_train, y_train)
    print("test acc:", evaluate(X_test, y_test, params))

    # params = init_params()
    # A2, cache = forward(X_train, params)
    # print("starting loss:", compute_loss(y_train, A2))
    # params = train(X_train, y_train, params)
    # A2, _ = forward(X_train, params)
    # baseline_loss = compute_loss(y_train, A2)
    # print(f"baseline loss: {baseline_loss:.4f}")
    # print("train accuracy:", evaluate(X_train, y_train, params), "%")
    # X_test, y_test = load_data("spect_test.txt")
    # print("test accuracy:", evaluate(X_test, y_test, params), "%")

    # optimize(params, X_train, y_train, baseline_loss)

    # print("Cross validation")
    # for nh in [1, 2, 3, 4, 5, 6, 8]:
    #     mean, std = cross_validate(X_train, y_train, n_hidden=nh)
    #     print(f"{nh} neurons: macro-F1 {mean:.3f} ± {std:.3f}")