import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from textbook_7_deep.two_feed_forward import data_helper as dh


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

# forward neural network =====
def FNN(wv, M, K, x):
    N, D = x.shape  # 入力次元
    w = wv[:M * (D + 1)] # 中間層ニューロンへの重み
    w = w.reshape(M, (D + 1))
    v = wv[M * (D + 1):] # 出力層ニューロンへの重み
    v = v.reshape((K, M + 1))
    b = np.zeros((N, M + 1))  # 中間層ニューロンの入力総和
    z = np.zeros((N, M + 1))  # 中間層ニューロンの出力
    a = np.zeros((N, K))  # 出力層ニューロンの入力総和
    y = np.zeros((N, K))  # 出力層ニューロンの出力
    for n in range(N):  # number of input
        # 中間層の計算
        for m in range(M):  # number of middle layer
            b[n, m] = np.dot(w[m, :], np.r_[x[n, :], 1])  # linear output
            z[n, m] = sigmoid(b[n, m])  # sigmoid of output
        # 出力層の計算
        z[n, M] = 1  # ダミーニューロン
        wkz = 0
        for k in range(K):  # number of output layer
            a[n, k] = np.dot(v[k, :], z[n, :])
            wkz = wkz + np.exp(a[n, k])
        for k in range(K):
            y[n, k] = np.exp(a[n, k]) / wkz  # Ratio. Sum of them is 1.
    return y, a, z, b

def CE_FNN(wv, M, K, x, t):
    N, D = x.shape
    y, a, z, b = FNN(wv, M, K, x)
    ce = -np.dot(t.reshape(-1), np.log(y.reshape(-1))) / N
    return ce

def dCE_FNN_num(wv, M, K, x, t):
    epsilon = 0.001
    dwv = np.zeros_like(wv)
    for iwv in range(len(wv)):
        wv_modified = wv.copy()
        wv_modified[iwv] = wv[iwv] - epsilon
        mse1 = CE_FNN(wv_modified, M, K, x, t)
        wv_modified[iwv] = wv[iwv] + epsilon
        mse2 = CE_FNN(wv_modified, M, K, x, t)
        dwv[iwv] = (mse2 - mse1) / (2 * epsilon)
    return dwv

# -- dWVの表示 ------------------
def show_WV(wv, M):
    N = wv.shape[0]
    plt.bar(range(1, M * 3 + 1), wv[:M * 3], align="center", color='black')
    plt.bar(range(M * 3 + 1, N + 1), wv[M * 3:], align="center", color='cornflowerblue')
    plt.xticks(range(1, N + 1))
    plt.xlim(0, N + 1)


if __name__ == '__main__':
    x_input, t_result = dh.create_data(dh.N_data, dh.K_dist, dh.Pi, dh.Sig, dh.Mu)
    train_input, train_result, test_input, test_result = dh.distribute_data_into_test_and_training(x_input, t_result, 0.5)

    M = 2
    K = 3
    nWV = M * 3 + K * (M + 1)
    WV = np.random.normal(0, 1, nWV)
    print(WV)
    dWV = dCE_FNN_num(WV, M, K, train_input[:2, :], train_result[:2, :])
    print(dWV)
    plt.figure(1, figsize=(5, 3))
    show_WV(dWV, M)
    plt.show()


    """
    WV = np.ones(15)
    M = 2
    K = 3
    print(FNN(WV, M, K, train_input[:2, :]))
    print(CE_FNN(WV, M, K, train_input[:2, :], train_result[:2, :]))
    """