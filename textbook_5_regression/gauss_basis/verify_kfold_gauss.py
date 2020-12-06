import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from textbook_5_regression.gauss_basis import gauss_basis_one as gau

# Load data =====
current_dir = os.path.dirname(__file__)
parent_dir = str(Path(current_dir).resolve().parent)
loaded_data = np.load(parent_dir + "/base_data/base_data_1.npz")
X = loaded_data["X"]
X_min = loaded_data["X_min"]
X_max = loaded_data["X_max"]
X_n = loaded_data["X_n"]
T = loaded_data["T"]

# K-fold cross validation =====
def kfold_gauss_func(x, t, m, k):
    n = x.shape[0]
    mse_train = np.zeros(k)
    mse_test = np.zeros(k)
    for i in range(0, k):
        x_train = x[np.fmod(range(n), k) != i]
        t_train = t[np.fmod(range(n), k) != i]
        x_test = x[np.fmod(range(n), k) == i]
        t_test = t[np.fmod(range(n), k) == i]
        wm = gau.fit_gauss_func(x_train, t_train, m)
        mse_train[i] = gau.mse_gauss_func(x_train, t_train, wm)
        mse_test[i] = gau.mse_gauss_func(x_test, t_test, wm)
    return mse_train, mse_test


if __name__ == '__main__':
    M = range(2, 12)
    K = 16
    Cv_Gauss_train = np.zeros((K, len(M)))
    Cv_Gauss_test = np.zeros((K, len(M)))
    for i in range(0, len(M)):
        Cv_Gauss_train[:, i], Cv_Gauss_test[:, i] = kfold_gauss_func(X, T, M[i], K)
    mean_Gauss_train = np.sqrt(np.mean(Cv_Gauss_train, axis=0))
    mean_Gauss_test = np.sqrt(np.mean(Cv_Gauss_test, axis=0))

    # Create graph =====
    plt.figure(figsize=(4, 3))
    plt.plot(M, mean_Gauss_train, marker="o", linestyle="-", color="k",
             markerfacecolor="w", label="training")
    plt.plot(M, mean_Gauss_test, marker="o", linestyle="-", color="cornflowerblue",
             markeredgecolor="black", label="test")
    plt.legend(loc="lower left", fontsize=10)
    plt.ylim(0, 20)
    plt.grid(True)
    plt.show()
