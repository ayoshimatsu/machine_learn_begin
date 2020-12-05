import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from textbook_5_regression.gauss_basis import gauss_basis_one as gau

# Load data
current_dir = os.path.dirname(__file__)
parent_dir = str(Path(current_dir).resolve().parent)
loaded_data = np.load(parent_dir + "/base_data/base_data_1.npz")
X = loaded_data["X"]
X_min = loaded_data["X_min"]
X_max = loaded_data["X_max"]
X_n = loaded_data["X_n"]
T = loaded_data["T"]

# Main =====
plt.figure(figsize=(10, 2.5))
plt.subplots_adjust(wspace=0.3)
M = [2, 4, 7, 9]  # M = number of gaussian basis func
for i in range(len(M)):
    plt.subplot(1, len(M), i+1)
    W = gau.fit_gauss_func(X, T, M[i])  # W = weight of predicted formula
    gau.show_gauss_func(W)  # show calculated formula
    plt.plot(X, T, marker="o", linestyle="None",
             color="cornflowerblue", markeredgecolor="black")  # plot actual data
    plt.xlim(X_min, X_max)
    plt.grid(True)
    plt.ylim(130, 180)
    mse = gau.mse_gauss_func(X, T, W)  # calculate mean squared error from result
    print("M={0:d}, SD={1:.1f}".format(M[i], np.sqrt(mse)))
plt.show()
