import numpy as np
import pandas as pd
import mytools
import os
from matplotlib import pyplot as plt


def linear_regression():
    data = load_data(os.path.dirname(__file__) + '/dataset/data.csv', ',', np.float64)
    X = data[:, 0:-1]
    y = data[:, -1]
    m = len(y)
    n = data.shape[1]
    X, mu, sigma = feature_normalization(X)
    X = np.hstack((np.ones((m, 1)), X))
    theta = np.zeros((n,))
    theta, j_history = gradient_descent(X, y, theta, 0.01)
    print(j_history[-1])
    plt.plot(list(range(len(j_history))), j_history)
    plt.show()


def compute_cost(X, y, theta):
    m = len(y)
    j = np.dot((np.dot(X, theta) - y), (np.dot(X, theta) - y)) / (2 * m)
    return j


def load_data(file_name, split, data_type):
    return np.loadtxt(file_name, delimiter=split, dtype=data_type)


def gradient_descent(X, y, theta, alpha, iter_nums=400):
    # 样本容量
    m = len(y)

    j_history = [compute_cost(X, y, theta)]

    for _ in range(iter_nums):
        gradient = np.dot((np.dot(X, theta) - y), X) / m
        theta = theta - alpha * gradient
        j_history.append(compute_cost(X, y, theta))

    return theta, j_history


def plot_X(X):
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()


def feature_normalization(X):
    X_norm = np.array(X)

    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X_norm = (X_norm - mu) / sigma

    return X_norm, mu, sigma


if __name__ == '__main__':
    linear_regression()

