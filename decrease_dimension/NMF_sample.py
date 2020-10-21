from sklearn.decomposition import NMF
from sklearn.datasets.samples_generator import make_blobs
from mpl_toolkits.mplot3d import Axes3D, axes3d
import matplotlib.pyplot as plt
import numpy as np

# 法線ベクトルを指定して平面をプロットする関数
# axes：サブプロット
# vector：法線ベクトル
# point：平面上の点
# xrange,yrange,zrange：x軸,y軸,z軸の範囲
# loc：法線ベクトルの始点(デフォルトは原点)
# vcolor：法線ベクトルの色
# pcolor,alpha：平面の色,透明度

def nvector_plane(axes, vector, point,
                  xrange, yrange, zrange,
                  loc = [0, 0, 0],
                  vcolor="red", pcolor="blue", alpha=0.5):

    # 軸ラベルの設定
    axes.set_xlabel("x", fontsize = 16)
    axes.set_ylabel("y", fontsize = 16)
    axes.set_zlabel("z", fontsize = 16)

    # 軸範囲の設定
    axes.set_xlim(xrange[0], xrange[1])
    axes.set_ylim(yrange[0], yrange[1])
    axes.set_zlim(zrange[0], zrange[1])

    # 格子点の作成
    x = np.arange(xrange[0], xrange[1], 0.2)
    y = np.arange(yrange[0], yrange[1], 0.2)
    xx, yy = np.meshgrid(x, y)

    # 平面の方程式
    zz = point[2] - (vector[0]*(xx-point[0])+vector[1]*(yy-point[1])) / vector[2]

    # 平面をプロット
    ax.plot_surface(xx, yy, zz, color=pcolor, alpha=alpha)

    # 法線ベクトルをプロット
    """
    axes.quiver(loc[0], loc[1], loc[2],
                vector[0], vector[1], vector[2],
                color = vcolor, length = 1, arrow_length_ratio = 0.2)
    """


# 3点を通る平面をプロットする関数
# axes：サブプロット
# point：平面上の点
# xrange,yrange,zrange：x軸,y軸,z軸の範囲
# loc：法線ベクトルの始点(デフォルトは原点)
# vcolor：法線ベクトルの色
# pcolor,alpha：平面の色,透明度

def point_plane(axes, p0, p1, p2,
                xrange, yrange, zrange,
                loc=[0, 0, 0],
                vcolor="red", pcolor="blue", alpha=0.5):
    u = p1 - p0
    v = p2 - p0
    w = np.cross(u, v)
    w = 0.5 * np.abs(xrange[0] - yrange[1]) * w / np.sqrt(np.sum(w ** 2))

    nvector_plane(axes, w, p0,
                  xrange, yrange, zrange,
                  loc=loc, vcolor=vcolor, pcolor=pcolor, alpha=alpha)



centers = [[5, 10, 5],
           [10, 4, 10],
           [6, 8, 8]]

center_array = np.array(centers)  # change to numpy matrix
print(center_array)
X, _ = make_blobs(centers=center_array)  # centersを中心としたデータを生成
n_components = 2  # 潜在変数の数
model = NMF(n_components=n_components)
model.fit(X)
W = model.transform(X)  # 分解後の行列
H = model.components_  # change data from 2D to 3D
WH = np.dot(W, H)
print(W)
print(H)
#print(WH)

# 3D graph
figure = plt.figure()
ax = Axes3D(figure, elev=20, azim=-110)  # rotate graph
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Z")

x_ax_lim = [0, 12]
y_ax_lim = [0, 12]
z_ax_lim = [0, 12]
p0 = np.array(H[0, :])
p1 = np.array(H[1, :])
p2 = np.array([0, 0, 0])
g = (p0 + p1 + p2) / 3
point_plane(ax, p0, p1, p2, x_ax_lim, y_ax_lim, z_ax_lim, loc=g)

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='r')
ax.scatter(WH[:, 0], WH[:, 1], WH[:, 2], c='b')
ax.scatter(H[:, 0], H[:, 1], H[:, 2], c='green')

plt.show()
