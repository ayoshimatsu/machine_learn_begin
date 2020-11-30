import numpy as np
import matplotlib.pyplot as plt
import os

# Load data
current_dir = os.path.dirname(__file__)
loaded_data = np.load(current_dir + "/gradient_data_2.npz")

X = loaded_data["X"]
X_min = loaded_data["X_min"]
X_max = loaded_data["X_max"]
X_n = loaded_data["X_n"]
T = loaded_data["T"]
W0 = loaded_data["W0"]
W1 = loaded_data["W1"]
dMSE = loaded_data["dMSE"]
W_history = loaded_data["W_history"]

def show_line(w):
    xb = np.linspace(X_min, X_max, 100)
    y = w[0] * xb + w[1]
    plt.plot(xb, y, color=(.5, .5, .5), linewidth=4)

# Mean error function =====
def mse_line(x, t, w):
    y = w[0] * x + w[1]
    mse = np.mean((y-t)**2)
    return mse


plt.figure(figsize=(4, 4))
W = np.array([W0, W1])
mse = mse_line(X, T, W)
print("w0={0:.3f}, w1={1:.3f}".format(W0, W1))
print("SD={0:.3f} cm".format(np.sqrt(mse)))
show_line(W)
plt.plot(X, T, marker="o", linestyle="None",
         color="cornflowerblue", markeredgecolor="black")
plt.xlim(X_min, X_max)
plt.grid(True)
plt.show()
