import numpy as np
import matplotlib.pyplot as plt
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

# Create test data & training data =====
X_test = X[:int(X_n / 4)]
T_test = T[:int(X_n / 4)]
X_train = X[int(X_n / 4):]
T_train = T[int(X_n / 4):]

# Main =====
plt.figure(figsize=(10, 2.5))
plt.subplots_adjust(wspace=0.3)
M = [2, 4, 7, 9]
for i in range(len(M)):
    plt.subplot(1, len(M), i + 1)
    W = gau.fit_gauss_func(X_train, T_train, M[i])
    gau.show_gauss_func(W)
    plt.plot(X_train, T_train, marker="o",
             linestyle="None", color="white",
             markeredgecolor="black", label="training")
    plt.plot(X_test, T_test, marker="o",
             linestyle="None", color="cornflowerblue",
             markeredgecolor="black", label="test")
    plt.legend(loc="lower right", fontsize=10, numpoints=1)
    plt.xlim(X_min, X_max)
    plt.ylim(120, 180)
    plt.grid(True)
    mse = gau.mse_gauss_func(X_test, T_test, W)
    plt.title("M={0:d}, SD={1:.1f}".format(M[i], np.sqrt(mse)))
plt.show()
