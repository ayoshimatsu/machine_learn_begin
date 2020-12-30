import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from textbook_9_clustering.gaussian_mixture import gaussian_function as gau_func

current_dir = os.path.dirname(__file__)
parent_dir = str(Path(current_dir).resolve().parent)
# x2_data, t3_data = cd.create_data(parent_dir + "/data_ch9.npz")  # x: 2 dimension, t: 3 dimension
wk = np.load(parent_dir + "/data_ch9.npz")
X = wk['X']
X_range0 = wk['X_range0']
X_range1 = wk['X_range1']

# initial configure ------------------------------------
N = X.shape[0]  # number of data
K = 3  # number of class
Pi = np.array([0.33, 0.33, 0.34])  # initial coefficient
Mu = np.array([[-2, 1], [-2, 0], [-2, -1]])  # initial center vector
Sigma = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]])  # covariance matrix
Gamma = np.c_[np.ones((N, 1)), np.zeros((N, 2))]  # ratio of attribution
X_col=np.array([[0.4, 0.6, 0.95], [1, 1, 1], [0, 0, 0]])  # class color

# データの図示 ------------------------------
def show_mixgauss_prm(x, gamma, pi, mu, sigma):
    N, D = x.shape
    gau_func.show_contour_mixgauss(pi, mu, sigma)
    for n in range(N):
        col = gamma[n,0]*X_col[0] + gamma[n,1]*X_col[1] + gamma[n,2]*X_col[2]
        plt.plot(x[n, 0], x[n, 1], 'o', color=tuple(col),
                 markeredgecolor='black', markersize=6, alpha=0.5)
    for k in range(K):
        plt.plot(mu[k, 0], mu[k, 1], marker='*', markerfacecolor=tuple(X_col[k]),
                 markersize=15, markeredgecolor='k', markeredgewidth=1)
    plt.grid(True)

# gamma を更新する (E Step) -------------------
def e_step_mixgauss(x, pi, mu, sigma):
    N, D = x.shape
    K = len(pi)
    y = np.zeros((N, K))
    for k in range(K):
        y[:, k] = gau_func.gauss(x, mu[k, :], sigma[k, :, :])  # KxN
    gamma = np.zeros((N, K))
    for n in range(N):
        wk = np.zeros(K)
        for k in range(K):
            wk[k] = pi[k] * y[n, k]  # coefficient x gauss
        gamma[n, :] = wk / np.sum(wk)
    return gamma

# Pi, Mu, Sigma を更新する (M step) ------------
def m_step_mixgauss(x, gamma):
    N, D = x.shape
    N, K = gamma.shape
    # pi を計算
    pi = np.sum(gamma, axis=0) / N
    # mu を計算
    mu = np.zeros((K, D))
    for k in range(K):
        for d in range(D):
            mu[k, d] = np.dot(gamma[:, k], x[:, d]) / np.sum(gamma[:, k])
    # sigma を計算
    sigma = np.zeros((K, D, D))
    for k in range(K):
        for n in range(N):
            wk = x - mu[k, :]
            wk = wk[n, :, np.newaxis]
            sigma[k, :, :] = sigma[k, :, :] + gamma[n, k] * np.dot(wk, wk.T)
        sigma[k, :, :] = sigma[k, :, :] / np.sum(gamma[:, k])
    return pi, mu, sigma

# 混合ガウスの目的関数 ----------------------
def nlh_mixgauss(x, pi, mu, sigma):
    # x: N x D
    # pi: K x 1 : coefficient(weight)
    # mu: K x D
    # sigma: K x D x D
    # output lh: N x K
    N, D = x.shape
    K = len(pi)
    y = np.zeros((N, K))
    for k in range(K):
        y[:, k] = gau_func.gauss(x, mu[k, :], sigma[k, :, :])  # K x N
    lh = 0
    for n in range(N):
        wk = 0
        for k in range(K):
            wk = wk + pi[k] * y[n, k]
        lh = lh + np.log(wk)
    return -lh

# one loop test ----------------------------------
# Gamma = e_step_mixgauss(X, Pi, Mu, Sigma)
# Pi, Mu, Sigma = m_step_mixgauss(X, Gamma)
# plt.figure(1, figsize=(4, 4))
# show_mixgauss_prm(X, Gamma, Pi, Mu, Sigma)
# plt.show()

# multiple loop test ------------------------------
# overwrite initial parameter
Pi = np.array([0.3, 0.3, 0.4])
Mu = np.array([[2, 2], [-2, 0], [2, -2]])
Sigma = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]])
Gamma = np.c_[np.ones((N, 1)), np.zeros((N, 2))]

max_it = 20  # number of repetition
"""
i_subplot = 1
plt.figure(1, figsize=(10, 6.5))
for it in range(0, max_it):
    Gamma = e_step_mixgauss(X, Pi, Mu, Sigma)
    if it < 4 or it > 17:
        plt.subplot(2, 3, i_subplot)
        show_mixgauss_prm(X, Gamma, Pi, Mu, Sigma)
        plt.title("{0:d}".format(it + 1))
        plt.xticks(range(X_range0[0], X_range0[1]), "")
        plt.yticks(range(X_range1[0], X_range1[1]), "")
        i_subplot=i_subplot+1
    Pi, Mu, Sigma = m_step_mixgauss(X, Gamma)
plt.show()
"""
# objective function =====
it = 0
Err = np.zeros(max_it)  # distortion measure
for it in range(0, max_it):
    Gamma = e_step_mixgauss(X, Pi, Mu, Sigma)
    Err[it] = nlh_mixgauss(X, Pi, Mu, Sigma)
    Pi, Mu, Sigma = m_step_mixgauss(X, Gamma)

print(np.round(Err, 2))
plt.figure(1, figsize=(4, 4))
plt.plot(np.arange(max_it) + 1, Err, color='k', linestyle='-', marker='o')
plt.grid(True)
plt.show()
