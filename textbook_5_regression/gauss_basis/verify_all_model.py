import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from textbook_5_regression.gauss_basis import gauss_basis_one as gau
from textbook_5_regression.gauss_basis import optimize_solution as modelA
from textbook_5_regression.gauss_basis import verify_kfold_gauss as gau_k


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
def kfold_model_A(x, t, k):
    n = len(x)
    mse_train = np.zeros(k)
    mse_test = np.zeros(k)
    for i in range(0, k):
        x_train = x[np.fmod(range(n), k) != i]
        t_train = t[np.fmod(range(n), k) != i]
        x_test = x[np.fmod(range(n), k) == i]
        t_test = t[np.fmod(range(n), k) == i]
        wm = modelA.fit_model_A(np.array([169, 113, 0.2]), x_train, t_train)
        mse_train[i] = modelA.mse_model_A(wm, x_train, t_train)
        mse_test[i] = modelA.mse_model_A(wm, x_test, t_test)
    return mse_train, mse_test


# Main =====
# Verify gauss basis function =====
K = 16
M = range(2, 10)
Cv_Gauss_train = np.zeros((K, len(M)))
Cv_Gauss_test = np.zeros((K, len(M)))
for i in range(0, len(M)):
    Cv_Gauss_train[:, i], Cv_Gauss_test[:, i] = gau_k.kfold_gauss_func(X, T, M[i], K)
mean_Gauss_train = np.sqrt(np.mean(Cv_Gauss_train, axis=0))
mean_Gauss_test = np.sqrt(np.mean(Cv_Gauss_test, axis=0))

# Verify model A(powell) method =====
Cv_A_train, Cv_A_test = kfold_model_A(X, T, K)
mean_A_test = np.sqrt(np.mean(Cv_A_test))
print("Gauss(M=3) SD={0:.2f}".format(mean_Gauss_test[1]))
print("Model A SD={0:.2f} cm".format(mean_A_test))
SD = np.append(mean_Gauss_test[0:8], mean_A_test)
M = range(9)
label = ["M=2", "M=3", "M=4", "M=5", "M=6", "M=7", "M=8", "M=9", "Model A"]

# Create graph =====
plt.figure(figsize=(5, 3))
plt.bar(M, SD, tick_label=label, align="center", facecolor="cornflowerblue")
plt.ylim(0, 20)
plt.show()
