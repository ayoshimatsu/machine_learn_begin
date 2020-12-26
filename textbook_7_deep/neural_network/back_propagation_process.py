from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from textbook_7_deep.neural_network import data_helper as dh
from textbook_7_deep.neural_network import calculate_helper as ch

def show_activation3d(ax, v, v_ticks, title_str):
    f = v.copy()
    f = f.reshape(xn, xn)
    f = f.T
    ax.plot_surface(xx0, xx1, f, color='blue', edgecolor='black',
                    rstride=1, cstride=1, alpha=0.5)
    ax.view_init(70, -110)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticks(v_ticks)
    ax.set_title(title_str, fontsize=18)


if __name__ == '__main__':
    M = 2
    K = 3
    xn = 15  # 等高線表示の解像度
    x0 = np.linspace(dh.X_range0[0], dh.X_range0[1], xn)
    x1 = np.linspace(dh.X_range1[0], dh.X_range1[1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    x = np.c_[np.reshape(xx0, [xn * xn, 1]), np.reshape(xx1, [xn * xn, 1])]
    nWV = M * 3 + K * (M + 1)
    WV = np.random.normal(0, 1, nWV)  # init weight
    y, a, z, b = ch.fNN(WV, M, K, x)

    fig = plt.figure(1, figsize=(12, 9))
    plt.subplots_adjust(left=0.075, bottom=0.05, right=0.95,
                        top=0.95, wspace=0.4, hspace=0.4)
    for m in range(M):
        ax = fig.add_subplot(3, 4, 1 + m * 4, projection='3d')
        show_activation3d(ax, b[:, m], [-10, 10], '$b_{0:d}$'.format(m))
        ax = fig.add_subplot(3, 4, 2 + m * 4, projection='3d')
        show_activation3d(ax, z[:, m], [0, 1], '$z_{0:d}$'.format(m))

    for k in range(K):
        ax = fig.add_subplot(3, 4, 3 + k * 4, projection='3d')
        show_activation3d(ax, a[:, k], [-5, 5], '$a_{0:d}$'.format(k))
        ax = fig.add_subplot(3, 4, 4 + k * 4, projection='3d')
        show_activation3d(ax, y[:, k], [0, 1], '$y_{0:d}$'.format(k))

    plt.show()
