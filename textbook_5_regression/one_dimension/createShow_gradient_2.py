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


# Mean error function =====
def mse_line(x, t, w):
    y = w[0] * x + w[1]
    mse = np.mean((y-t)**2)
    return mse

# Gradient of mean error
def dmse_line(x, t, w):
    y = w[0] * x + w[1]
    d_w0 = 2 * np.mean((y-t) * x)
    d_w1 = 2 * np.mean(y-t)
    return d_w0, d_w1

def fit_line_num(x, t):
    # init parameter =====
    w_init = [10.0, 165.0]  # init parameter
    alpha = 0.001  # Learning rate
    tau_max = 100000  # Max of repetition
    eps = 0.1  # Threshold of gradient to stop repetition

    w_hist = np.zeros([tau_max, 2])
    w_hist[0, :] = w_init
    for tau in range(1, tau_max):
        dmse = dmse_line(x, t, w_hist[tau - 1])
        w_hist[tau, 0] = w_hist[tau - 1, 0] - alpha * dmse[0]
        w_hist[tau, 1] = w_hist[tau - 1, 1] - alpha * dmse[1]
        if max(np.absolute(dmse)) < eps:
            break
    w0 = w_hist[tau, 0]
    w1 = w_hist[tau, 1]
    w_hist = w_hist[:tau, :]
    return w0, w1, dmse, w_hist


# Invoke gradient method =====
W0, W1, dMSE, W_history = fit_line_num(X, T)
np.savez(current_dir + "/gradient_data_2.npz",
         X=X, X_min=X_min, X_max=X_max, X_n=X_n, T=T,
         W0=W0, W1=W1, dMSE=dMSE, W_history=W_history)

# Show result =====
print("Repetition: {0}".format(W_history.shape[0]))
print("W=[{0:.6f}, {1:.6f}]".format(W0, W1))
print("dMSE=[{0:.6f}, {1:.6f}]".format(dMSE[0], dMSE[1]))
print("MSE={0:.6f}".format(mse_line(X, T, [W0, W1])))

# Show contour graph =====
plt.figure(figsize=(4, 4))
wn = 100
w0_range = [-25, 25]
w1_range = [120, 170]
w0 = np.linspace(w0_range[0], w0_range[1], wn)
w1 = np.linspace(w1_range[0], w1_range[1], wn)
ww0, ww1 = np.meshgrid(w0, w1)
J = np.zeros((len(w0), len(w1)))
for i0 in range(wn):
    for i1 in range(wn):
        J[i1, i0] = mse_line(X, T, (w0[i0], w1[i1]))

cont = plt.contour(ww0, ww1, J, 30, colors="black",
                   levels=[100, 1000, 10000, 100000])
cont.clabel(fmt="%d", fontsize=8)
plt.xlabel('$w_0$', fontsize=14)
plt.ylabel('$w_1$', fontsize=14)
plt.grid(True)
plt.plot(W_history[:, 0], W_history[:, 1], ".-", color="gray",
         markersize=10, markeredgecolor="cornflowerblue")
plt.show()
