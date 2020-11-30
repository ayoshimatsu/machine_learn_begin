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


# Calculate error
xn = 100  # resolution of contour
w0_range = [-25, 25]
w1_range = [120, 170]
w0 = np.linspace(w0_range[0], w0_range[1], xn)
w1 = np.linspace(w1_range[0], w1_range[1], xn)
ww0, ww1 = np.meshgrid(w0, w1)
J = np.zeros((len(w0), len(w1)))
for i0 in range(len(w0)):
    for i1 in range(len(w1)):
        J[i1, i0] = mse_line(X, T, (w0[i0], w1[i1]))

# Show graph =====
plt.figure(figsize=(8, 6))
plt.subplots_adjust(wspace=0.5)

ax1 = plt.subplot(2, 2, 1)
ax1.plot(X, T, marker="o", linestyle="None", markeredgecolor="black",
         color="cornflowerblue")
plt.xlim(X_min, X_max)
plt.grid(True)

ax2 = plt.subplot(2, 2, 3, projection="3d")
ax2.plot_surface(ww0, ww1, J, rstride=10, cstride=10, alpha=0.3,
                color="blue", edgecolor="black")
ax2.set_xticks([-20, 0, 20])
ax2.set_yticks([120, 140, 160])
ax2.set_xlabel('$w_0$', fontsize=14)
ax2.set_ylabel('$w_1$', fontsize=14)
ax2.view_init(20, -60)

plt.subplot(2, 2, 4)
cont = plt.contour(ww0, ww1, J, 30, colors="black",
                   levels=[100, 1000, 10000, 100000])
cont.clabel(fmt="%d", fontsize=8)
plt.xlabel('$w_0$', fontsize=14)
plt.ylabel('$w_1$', fontsize=14)
plt.grid(True)
plt.show()
