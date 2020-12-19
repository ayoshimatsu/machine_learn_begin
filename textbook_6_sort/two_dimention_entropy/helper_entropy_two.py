import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Create data --------------------------------
np.random.seed(seed=1)  # 乱数を固定
N_data = 100  # データの数
K_dist = 3  # 分布の数

Pi = np.array([0.4, 0.8, 1])  # ratio of class
Sig = np.array([[.7, .7], [.8, .3], [.3, .8]])  # 分布の分散
Mu = np.array([[-.5, -.5], [.5, 1.0], [1, -.5]])  # 分布の中心

X_range0 = [-3, 3]  # X0 の範囲 , 表示用
X_range1 = [-3, 3]  # X1 の範囲 , 表示用

def create_data(data_num, dist_num, class_ratio, data_dist, data_center):
    t3_data = np.zeros((data_num, 3), dtype=np.uint8)  # for 2 class
    t2_data = np.zeros((data_num, 2), dtype=np.uint8)  # for 3 class
    x_data = np.zeros((data_num, 2))
    for data_index in range(data_num):
        ratio = np.random.rand()
        for dist_index in range(dist_num):  # (B)
            if ratio < class_ratio[dist_index]:
                t3_data[data_index, dist_index] = 1
                break
        for dimen_index in range(2):
            x_data[data_index, dimen_index] = (np.random.randn() * data_dist[t3_data[data_index, :] == 1, dimen_index]
                                               + data_center[t3_data[data_index, :] == 1, dimen_index])
    t2_data[:, 0] = t3_data[:, 0]
    t2_data[:, 1] = t3_data[:, 1] | t3_data[:, 2]
    return x_data, t2_data, t3_data

# Calculation of logistic ----------
# 2 class sort ====
def logistic2(x0, x1, w):
    y = 1 / (1 + np.exp(-(w[0] * x0 + w[1] * x1 + w[2])))
    return y

# Error of entropy =====
def cee_logistic2(w, x, t):
    x_n = x.shape[0]
    y = logistic2(x[:, 0], x[:, 1], w)
    cee = 0
    for n in range(len(y)):
        cee = cee - (t[n, 0]*np.log(y[n]) + (1 - t[n, 0])*np.log(1 - y[n]))
    cee = cee / x_n
    return cee

# Differentiate error of entropy =====
def dcee_logistic2(w, x, t):
    x_n = x.shape[0]
    y = logistic2(x[:, 0], x[:, 1], w)
    dcee = np.zeros(3)
    for n in range(len(y)):
        dcee[0] = dcee[0] + (y[n] - t[n, 0]) * x[n, 0]
        dcee[1] = dcee[1] + (y[n] - t[n, 0]) * x[n, 1]
        dcee[2] = dcee[2] + (y[n] - t[n, 0])
    dcee = dcee / x_n
    return dcee

def fit_logistic2(w_init, x, t):
    res = minimize(cee_logistic2, w_init, args=(x, t),
                   jac=dcee_logistic2, method="CG")
    return res.x

# 3 class sort =====
def logistic3(x0, x1, w):
    w = w.reshape(K_dist, 3)
    n = len(x1)
    y = np.zeros((n, K_dist))
    for k in range(K_dist):
        y[:, k] = np.exp(w[k, 0] * x0 + w[k, 1] * x1 + w[k, 2])
    wk = np.sum(y, axis=1)
    wk = y.T / wk
    y = wk.T
    return y

# Error of entropy =====
def cee_logistic3(w, x, t):
    x_n = x.shape[0]
    y = logistic3(x[:, 0], x[:, 1], w)
    cee = 0
    data_num, k_num = y.shape
    for n in range(data_num):
        for k in range(k_num):
            cee = cee - (t[n, k] * np.log(y[n, k]))
    cee = cee / x_n
    return cee

# Differentiate error of entropy =====
def dcee_logistic3(w, x, t):
    x_n = x.shape[0]
    y = logistic3(x[:, 0], x[:, 1], w)
    dcee = np.zeros((K_dist, 3))  # number of classes and number of input
    data_num, k_num = y.shape
    for n in range(data_num):
        for k in range(k_num):
            dcee[k, :] = dcee[k, :] - (t[n, k] - y[n, k])*np.r_[x[n, :], 1]
    dcee = dcee / x_n
    return dcee.reshape(-1)

def fit_logistic3(w_init, x, t):
    res = minimize(cee_logistic3, w_init, args=(x, t), jac=dcee_logistic3, method="CG")
    return res.x


# データ表示 --------------------------
def show_data2(x, t):
    wk, dist_num = t.shape
    color = [[.5, .5, .5], [1, 1, 1], [0, 0, 0]]
    for k in range(dist_num):
        plt.plot(x[t[:, k] == 1, 0], x[t[:, k] == 1, 1],
                 linestyle='none', markeredgecolor='black',
                 marker='o', color=color[k], alpha=0.8)
    plt.grid(True)

# Show logistic model =====
def show3d_logistic2(ax, w):
    xn = 50
    x0 = np.linspace(X_range0[0], X_range0[1], xn)
    x1 = np.linspace(X_range1[0], X_range1[1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    y = logistic2(xx0, xx1, w)
    ax.plot_surface(xx0, xx1, y, color='blue', edgecolor='gray',
                    rstride=5, cstride=5, alpha=0.3)

# Only for 2 distribution data =====
def show_data2_3d(ax, x, t):
    color = [[.5, .5, .5], [1, 1, 1]]
    for i in range(2):
        ax.plot(x[t[:, i] == 1, 0], x[t[:, i] == 1, 1], 1 - i,
                marker='o', color=color[i], markeredgecolor='black',
                linestyle='none', markersize=5, alpha=0.8)
    ax.view_init(elev=25, azim=-30)

def show_contour_logistic2(w):
    xn = 30  # パラメータの分割数
    x0 = np.linspace(X_range0[0], X_range0[1], xn)
    x1 = np.linspace(X_range1[0], X_range1[1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    y = logistic2(xx0, xx1, w)
    cont = plt.contour(xx0, xx1, y, levels=(0.2, 0.5, 0.8),
                       colors=['k', 'cornflowerblue', 'k'])
    cont.clabel(fmt='%.1f', fontsize=10)
    plt.grid(True)

def show_contour_logistic3(w):
    xn = 30  # パラメータの分割数
    x0 = np.linspace(X_range0[0], X_range0[1], xn)
    x1 = np.linspace(X_range1[0], X_range1[1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    y = np.zeros((xn, xn, 3))
    for i in range(xn):
        wk = logistic3(xx0[:, i], xx1[:, i], w)
        for j in range(3):
            y[:, i, j] = wk[:, j]
    for j in range(3):
        cont = plt.contour(xx0, xx1, y[:, :, j],
                           levels=(0.5, 0.9), colors=['cornflowerblue', 'k'])
        cont.clabel(fmt='%.1f', fontsize=9)
    plt.grid(True)


if __name__ == '__main__':
    # test ---
    x_data, t2_data, t3_data = create_data(N_data, K_dist, Pi, Sig, Mu)
    W = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # y = logistic3(x_data[:3, 0], x_data[:3, 1], W)
    # print(cee_logistic3(W, x_data, t3_data))
    print(dcee_logistic3(W, x_data, t3_data))
