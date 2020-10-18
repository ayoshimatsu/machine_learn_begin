from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
from mpl_toolkits.mplot3d import Axes3D, axes3d
import matplotlib.pyplot as plt
import mglearn
import numpy as np

X, y = make_blobs(centers=4, random_state=8)
y = y % 2

# 2 dimensions graph
# plt.xlabel("feature 0")
# plt.ylabel("feature 1")
# linear_svc = LinearSVC().fit(X, y)
# mglearn.plots.plot_2d_separator(linear_svc, X)
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)

# 3 dimensions graph
X_new = np.hstack([X, X[:, 1:] ** 2])
print(X)
print(X[:, 1:])
print(X_new)
figure = plt.figure()
#ax = Axes3D(figure, elev=-160, azim=-40)
mask = y == 0
#ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b')
#ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^')
#ax.set_xlabel("feature 0")
#ax.set_ylabel("feature 1")
#ax.set_zlabel("Square of feature 1")

linear_svm_3d = LinearSVC().fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_
print(coef)
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)
XX, YY = np.meshgrid(xx, yy)
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
#ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=.3)
#plt.show()


# result in 2D
ZZ = YY ** 2
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
print(XX)
print(XX.shape)
print(dec)
print(dec.reshape(XX.shape))
plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()],
             cmap=mglearn.cm2, alpha=.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
plt.show()
