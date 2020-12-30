import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

# data to create init data =====
N = 100
K = 3
Mu = np.array([[-.5, -.5], [.5, 1.0], [1, -.5]])  # 分布の中心
Sig = np.array([[.7, .7], [.8, .3], [.3, .8]])  # 分布の分散
Pi = np.array([0.4, 0.8, 1])  # 累積確率
X_range0 = [-3, 3]
X_range1 = [-3, 3]
X_col = ['cornflowerblue', 'black', 'white']

# test data =====
Mu_test = np.array([[-2, 1], [-2, 0], [-2, -1]])
R = np.c_[np.ones((N, 1), dtype=int), np.zeros((N, 2), dtype=int)]

# create data --------------------------------
def create_data(file):
    np.random.seed(1)
    T3 = np.zeros((N, 3), dtype=np.uint8)
    X = np.zeros((N, 2))

    for n in range(N):
        wk = np.random.rand()
        for k in range(K):
            if wk < Pi[k]:
                T3[n, k] = 1
                break
        for k in range(2):
            X[n, k] = (np.random.randn() * Sig[T3[n, :] == 1, k] + Mu[T3[n, :] == 1, k])

    np.savez(file, X=X, X_range0=X_range0, X_range1=X_range1)
    return X, T3

# calculate distance =====
def step1_kmeans(x0, x1, mu):
    N = len(x0)
    r = np.zeros((N, K))

    for n in range(N):
        wk = np.zeros(K)
        for k in range(K):
            wk[k] = (x0[n] - mu[k, 0])**2 + (x1[n] - mu[k, 1])**2
        r[n, np.argmin(wk)] = 1
    return r

# calculate mean of every class data =====
def step2_kmeans(x0, x1, r):
    mu = np.zeros((K, 2))
    for k in range(K):
        mu[k, 0] = np.sum(r[:, k] * x0) / np.sum(r[:, k])
        mu[k, 1] = np.sum(r[:, k] * x1) / np.sum(r[:, k])
    return mu

# 目的関数 ----------------------------------
def distortion_measure(x0, x1, r, mu):
    # 入力は 2 次元に限っている
    N = len(x0)
    J = 0
    for n in range(N):
        for k in range(K):
            J = J + r[n, k] * ((x0[n] - mu[k, 0])**2 + (x1[n] - mu[k, 1])**2)
    return J

# show init data ------------------------------
def show_data(x):
    plt.figure(1, figsize=(4, 4))
    plt.plot(x[:, 0], x[:, 1], linestyle='none', marker='o', markersize=6,
             markeredgecolor='black', color='gray', alpha=0.8)
    plt.grid(True)
    plt.xlim(X_range0)
    plt.ylim(X_range1)
    plt.show()

# show test data =====
def show_prm(x, r, mu, col, title):
    for k in range(K):
        # データ分布の描写
        plt.plot(x[r[:, k] == 1, 0], x[r[:, k] == 1, 1],
                 marker='o',
                 markerfacecolor=col[k], markeredgecolor='k',
                 markersize=6, alpha=0.5, linestyle='none')
        # データの平均を「星マーク」で描写
        plt.plot(mu[k, 0], mu[k, 1], marker='*',
                 markerfacecolor=col[k], markersize=15,
                 markeredgecolor='k', markeredgewidth=1)
    plt.grid(True)
    plt.title(title)


if __name__ == '__main__':
    current_dir = os.path.dirname(__file__)
    parent_dir = str(Path(current_dir).resolve().parent)

    x2_data, t3_data = create_data(parent_dir + "/data_ch9.npz")  # x: 2 dimension, t: 3 dimension
    R = np.c_[np.ones((N, 1), dtype=int), np.zeros((N, 2), dtype=int)]
    max_it = 10
    it = 0
    DM = np.zeros(max_it)  # 歪み尺度の計算結果を入れる
    for it in range(0, max_it):  # K-means 法
        R = step1_kmeans(x2_data[:, 0], x2_data[:, 1], Mu_test)
        DM[it] = distortion_measure(x2_data[:, 0], x2_data[:, 1], R, Mu_test)  # 歪み尺度
        Mu_test = step2_kmeans(x2_data[:, 0], x2_data[:, 1], R)
    print(np.round(DM, 2))
    plt.figure(1, figsize=(4, 4))
    plt.plot(DM, color='black', linestyle='-', marker='o')
    plt.ylim(40, 80)
    plt.grid(True)
    plt.show()

    # J = distortion_measure(x2_data[:, 0], x2_data[:, 1], R, Mu_test)
    # print(J)

    """ show result by graph
    x2_data, t3_data = create_data()  # x: 2 dimension, t: 3 dimension
    plt.figure(1, figsize=(10, 6.5))
    max_it = 6
    for it in range(0, max_it):
        plt.subplot(2, 3, it + 1)
        R = step1_kmeans(x2_data[:, 0], x2_data[:, 1], Mu_test)  # result of one step
        show_prm(x2_data, R, Mu_test, X_col, "{0:d}".format(it+1))

        plt.xticks(range(X_range0[0], X_range0[1]), "")
        plt.yticks(range(X_range1[0], X_range1[1]), "")

        Mu_test = step2_kmeans(x2_data[:, 0], x2_data[:, 1], R)
        # show_data(x2_data)
    plt.show()
    """