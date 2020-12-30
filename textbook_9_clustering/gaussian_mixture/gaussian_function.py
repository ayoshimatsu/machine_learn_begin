import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from textbook_9_clustering.clustering import create_data as cd

current_dir = os.path.dirname(__file__)
parent_dir = str(Path(current_dir).resolve().parent)
# x2_data, t3_data = cd.create_data(parent_dir + "/data_ch9.npz")  # x: 2 dimension, t: 3 dimension
wk = np.load(parent_dir + "/data_ch9.npz")
X = wk['X']
X_range0 = wk['X_range0']
X_range1 = wk['X_range1']

# gaussian function -----------------------------
def gauss(x, mu, sigma):
    N, D = x.shape
    c1 = 1 / (2 * np.pi)**(D / 2)
    c2 = 1 / (np.linalg.det(sigma)**(1 / 2))
    inv_sigma = np.linalg.inv(sigma)
    c3 = x - mu
    c4 = np.dot(c3, inv_sigma)
    c5 = np.zeros(N)
    for d in range(D):
        c5 = c5 + c4[:, d] * c3[:, d]
    p = c1 * c2 * np.exp(-c5 / 2)
    return p

# gaussian mixture =====
# x: input
# pi: coefficient of mixture
def mixgauss(x, pi, mu, sigma):
    N, D = x.shape
    K = len(pi)  # number of class
    p = np.zeros(N)
    for k in range(K):
        p = p + pi[k] * gauss(x, mu[k, :], sigma[k, :, :])
    return p

# 混合ガウス等高線表示 ----------------------
def show_contour_mixgauss(pi, mu, sigma):
    xn = 40  # 等高線表示の解像度
    x0 = np.linspace(X_range0[0], X_range0[1], xn)
    x1 = np.linspace(X_range1[0], X_range1[1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    x = np.c_[np.reshape(xx0, [xn * xn, 1]), np.reshape(xx1, [xn * xn, 1])]
    f = mixgauss(x, pi, mu, sigma)
    f = f.reshape(xn, xn)
    plt.contour(x0, x1, f, 10, colors='gray')

# 混合ガウス 3D 表示 ---------------------------
def show3d_mixgauss(ax, pi, mu, sigma):
    xn = 40  # 等高線表示の解像度
    x0 = np.linspace(X_range0[0], X_range0[1], xn)
    x1 = np.linspace(X_range1[0], X_range1[1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    x = np.c_[np.reshape(xx0, [xn * xn, 1]), np.reshape(xx1, [xn * xn, 1])]
    f = mixgauss(x, pi, mu, sigma)
    f = f.reshape(xn, xn)
    ax.plot_surface(xx0, xx1, f, rstride=2, cstride=2, alpha=0.3,
                    color='blue', edgecolor='black')


if __name__ == '__main__':
    x = np.array([[1, 2], [2, 2], [3, 4]])  # input
    pi = np.array([0.3, 0.7])
    mu = np.array([[1, 1], [2, 2]])
    sigma = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 1]]])
    print(mixgauss(x, pi, mu, sigma))

    # test  -----------------------------------
    pi = np.array([0.2, 0.4, 0.4])
    mu = np.array([[-2, -2], [-1, 1], [1.5, 1]])
    sigma = np.array([[[.5, 0], [0, .5]], [[1, 0.25], [0.25, .5]], [[.5, 0], [0, .5]]])

    Fig = plt.figure(1, figsize=(8, 3.5))
    Fig.add_subplot(1, 2, 1)
    show_contour_mixgauss(pi, mu, sigma)
    plt.grid(True)

    Ax = Fig.add_subplot(1, 2, 2, projection='3d')
    show3d_mixgauss(Ax, pi, mu, sigma)
    Ax.set_zticks([0.05, 0.10])
    Ax.set_xlabel('$x_0$', fontsize=14)
    Ax.set_ylabel('$x_1$', fontsize=14)
    Ax.view_init(40, -100)
    plt.xlim(X_range0)
    plt.ylim(X_range1)
    plt.show()
