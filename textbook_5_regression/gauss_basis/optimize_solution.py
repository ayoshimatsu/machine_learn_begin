import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize

# Load data
current_dir = os.path.dirname(__file__)
parent_dir = str(Path(current_dir).resolve().parent)
loaded_data = np.load(parent_dir + "/base_data/base_data_1.npz")
X = loaded_data["X"]
X_min = loaded_data["X_min"]
X_max = loaded_data["X_max"]
X_n = loaded_data["X_n"]
T = loaded_data["T"]

# Calculation model =====
def model_A(x, w):
    y = w[0] - w[1] * np.exp(-w[2] * x)
    return y
# Show model A =====
def show_model_A(w):
    xb = np.linspace(X_min, X_max, 100)
    y = model_A(xb, w)
    plt.plot(xb, y, c=[.5, .5, .5], lw=4)
# MSE of model A =====
def mse_model_A(w, x, t):
    y = model_A(x, w)
    mse = np.mean((y - t)**2)
    return mse
# Optimize parameter of model A =====
def fit_model_A(w_init, x, t):
    res1 = minimize(mse_model_A, w_init, args=(x, t), method="powell")
    return res1.x


# Main =====
if __name__ == '__main__':
    plt.figure(figsize=(4, 4))
    W_init = [100, 1, 1]
    W = fit_model_A(W_init, X, T)
    print("w0={0:.1f}, w1={1:.1f}, w2={2:.1f}".format(W[0], W[1], W[2]))
    show_model_A(W)
    plt.plot(X, T, marker="o", linestyle="None",
             color="cornflowerblue", markerfacecolor="black")
    plt.xlim(X_min, X_max)
    plt.grid(True)
    mse = mse_model_A(W, X, T)
    print("SD={0:.2f} cm".format(np.sqrt(mse)))
    plt.show()
