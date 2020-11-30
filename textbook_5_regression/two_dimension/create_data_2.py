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

# Create two dimension data
X0 = X
X0_min = 5
X0_max = 30
np.random.seed(seed=1)
X1 = 23 * (T / 100)**2 + 2 * np.random.random(X_n)
X1_min = 40
X1_max = 75

print(np.round(X0, 2))
print(np.round(X1, 2))
print(np.round(T, 2))

# Show two dimension data
def show_data2(ax, x0, x1, t):
    for i in range(len(x0)):
        ax.plot([x0[i], x0[i]], [x1[i], x1[i]],
                [120, t[i]], color="gray")
        ax.plot(x0, x1, t, "o",
                color="cornflowerblue", markeredgecolor="black",
                markersize=6, markeredgewidth=0.5)
        ax.view_init(elev=35, azim=-75)

def show_plane(ax, w):
    px0 = np.linspace(X0_min, X0_max, 5)
    px1 = np.linspace(X1_min, X1_max, 5)
    px0, px1 = np.meshgrid(px0,px1)
    y = w[0]*px0 + w[1]*px1 + w[2]
    ax.plot_surface(px0, px1, y, rstride=1, cstride=1, alpha=0.3,
                    color="blue", edgecolor="black")

# MSE of plane =====
def mse_plane(x0, x1, t, w):
    y = w[0] * x0 + w[1] * x1 + w[2]
    mse = np.mean((y - t)**2)
    return mse


plt.figure(figsize=(6, 5))
ax = plt.subplot(1, 1, 1, projection="3d")
W = [1.5, 1, 90]
show_plane(ax, W)
show_data2(ax, X0, X1, T)
mse = mse_plane(X0, X1, T, W)
print("SD={0:.2f} cm".format(np.sqrt(mse)))
plt.show()
