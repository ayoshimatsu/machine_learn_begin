import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Load data
current_dir = os.path.dirname(__file__)
parent_dir = str(Path(current_dir).resolve().parent)
loaded_data = np.load(parent_dir + "/base_data/base_data_1.npz")
X = loaded_data["X"]
X_min = loaded_data["X_min"]
X_max = loaded_data["X_max"]
X_n = loaded_data["X_n"]
T = loaded_data["T"]

def gauss(x, mu, s):
    return np.exp(-(x - mu)**2 / (2 * s**2))

# Calculate formula result =====
def gauss_func(w, x):
    m = len(w) - 1
    mu = np.linspace(5, 30, m)
    s = mu[1] - mu[0]
    y = np.zeros_like(x)
    for j in range(m):
        y = y + w[j] * gauss(x, mu[j], s)
    y = y + w[m]
    return y

# Mean squared error =====
def mse_gauss_func(x, t, w):
    y = gauss_func(w, x)
    mse = np.mean((y - t)**2)
    return mse

# Linear gaussian basis func model =====
# Calculate weight
def fit_gauss_func(x, t, m):
    mu = np.linspace(5, 30, m)
    s = mu[1] - mu[0]
    n = x.shape[0]  # number of row
    phi = np.ones((n, m+1))  # 1 matrix
    for j in range(m):
        phi[:, j] = gauss(x, mu[j], s)
    phi_T = np.transpose(phi)
    b = np.linalg.inv(phi_T.dot(phi))
    c = b.dot(phi_T)
    w = c.dot(t)
    return w

# Show result of gaussian basis func graph =====
def show_gauss_func(w):
    xb = np.linspace(X_min, X_max, 100)
    y = gauss_func(w, xb)
    plt.plot(xb, y, c=[.5, .5, .5], lw=4)


# Main =====
if __name__ == '__main__':
    plt.figure(figsize=(4, 4))
    M = 3  # M = number of gaussian basis func
    W = fit_gauss_func(X, T, M)  # W = weight of predicted formula
    show_gauss_func(W)  # show calculated formula
    plt.plot(X, T, marker="o", linestyle="None",
             color="cornflowerblue", markeredgecolor="black")  # plot actual data
    plt.xlim(X_min, X_max)
    plt.grid(True)
    mse = mse_gauss_func(X, T, W)  # calculate mean squared error from result
    print("W=" + str(np.round(W, 1)))
    print("SD={0:.2f} cm".format(np.sqrt(mse)))
    plt.show()
