import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def f(w0, w1):  # original formula
    return w0**2 + 2 * w0 * w1 + 3

def df_dw0(w0, w1):  # partial differentiation of w0
    return 2 * w0 + 2 * w1

def df_dw1(w0, w1):  # partial differentiation of w1
    return 2 * w0 + 0 * w1

w_range = 10
dw = 1
w0 = np.arange(-w_range, w_range + dw, dw)
w1 = np.arange(-w_range, w_range + dw, dw)

ww0, ww1 = np.meshgrid(w0, w1)
ff = np.zeros((len(w0), len(w1)))
dff_dw0 = np.zeros((len(w0), len(w1)))
dff_dw1 = np.zeros((len(w0), len(w1)))
for i0 in range(len(w0)):
    for i1 in range(len(w1)):
        ff[i1, i0] = f(w0[i0], w1[i1])
        dff_dw0[i1, i0] = df_dw0(w0[i0], w1[i1])
        dff_dw1[i1, i0] = df_dw1(w0[i0], w1[i1])

plt.figure(figsize=(9, 4))
plt.subplots_adjust(wspace=0.3)
plt.subplot(1, 2, 1)
cont = plt.contour(ww0, ww1, ff, 10, colors='k')  # contour
cont.clabel(fmt='%d', fontsize=8)
plt.xticks(range(-w_range, w_range + 1, 1))
plt.yticks(range(-w_range, w_range + 1, 1))
plt.xlim(-w_range - 0.5, w_range + 0.5)
plt.ylim(-w_range - 0.5, w_range + 0.5)
plt.xlabel('$w_0$', fontsize=14)
plt.ylabel('$w_1$', fontsize=14)

plt.subplot(1, 2, 2)
plt.quiver(ww0, ww1, dff_dw0, dff_dw1)  # gradient vector
plt.xlabel('$w_0$', fontsize=14)
plt.ylabel('$w_1$', fontsize=14)
plt.xticks(range(-w_range, w_range + 1, 1))
plt.yticks(range(-w_range, w_range + 1, 1))
plt.xlim(-w_range - 0.5, w_range + 0.5)
plt.ylim(-w_range - 0.5, w_range + 0.5)

plt.figure(figsize=(9, 4))
ax = plt.subplot(1, 1, 1, projection="3d")
ax.plot_surface(ww0, ww1, ff, rstride=1, cstride=1, alpha=0.3,
                color='blue', edgecolor='black')
ax.set_xlabel('$w_0$', fontsize=14)
ax.set_ylabel('$w_1$', fontsize=14)
# ax.set_zticks((0, 1, 2))
ax.view_init(75, -95)

plt.show()
