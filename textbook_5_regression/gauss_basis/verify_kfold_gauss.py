import numpy as np
import os
from pathlib import Path
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


M = 4
K = 4
print(kfold_gauss_func(X, T, M, K))
