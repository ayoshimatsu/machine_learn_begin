import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def function3(x0, x1):
    ans = (2 * x0**2 + x1**2) * np.exp(-(2 * x0**2 + x1**2))
    return ans

xn = 50
x0 = np.linspace(-2, 2, xn)
x1 = np.linspace(-2, 2, xn)
y = np.zeros((xn, xn))


for i0 in range(xn):
    for i1 in range(xn):
        y[i1, i0] = function3(x0[i0], x1[i1])

print(np.round(y, 2))

"""
plt.figure(figsize=(3.5, 3))
plt.gray()
plt.pcolor(y)
plt.colorbar()
plt.show()
"""

xx0, xx1 = np.meshgrid(x0, x1)
"""
plt.figure(figsize=(5, 3.5))
ax = plt.subplot(1, 1, 1, projection="3d")
ax.plot_surface(xx0, xx1, y, rstride=1, cstride=1, alpha=0.3,
                color='blue', edgecolor='black')
ax.set_zticks((0, 0.2, 0.4))
ax.view_init(75, -95)
plt.show()
"""

plt.figure(1, figsize=(6, 6))
cont = plt.contour(xx0, xx1, y, 5, colors="black")
cont.clabel(fmt="%.2f", fontsize=8)
plt.xlabel("$x_0$", fontsize=14)
plt.ylabel("$x_1$", fontsize=14)
plt.show()
