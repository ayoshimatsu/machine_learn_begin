from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# create data
centers = [(-1, -0.125), (0.5, 0.5)]
X, y = make_blobs(n_samples=50, n_features=2, centers=centers, cluster_std=0.3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# create model
model = LinearSVC()
model.fit(X_train, y_train)  # 学習
# predict result
y_pred = model.predict(X_test)
print(accuracy_score(y_pred, y_test))  # 評価
print(model.intercept_)
print(model.coef_)
# create graph
fig = plt.figure(figsize=(6, 6))  # create plot object
# ax1
ax = fig.add_subplot(111)
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
# set color to result data
color_codes = {0: 'red', 1: 'green'}
color_codes2 = {0: 'orange', 1: 'blue'}
colors_train = [color_codes[x] for x in y_train]
colors_test = [color_codes2[x] for x in y_test]
# border line (create by myself)
x_axis = np.arange(-2, 2, 0.05)
y_axis = np.arange(-2, 2, 0.05)
y_axis_line = (- model.intercept_ - model.coef_[:, 0] * x_axis) / model.coef_[:, 1]
# create mesh
X1, X2 = np.meshgrid(x_axis, y_axis)
plot_X = np.c_[X1.ravel(), X2.ravel()]
print(plot_X)
plot_y = model.predict(plot_X)
print(plot_X.T[0])
print(plot_X.T[0][plot_y <= 0])
print(plot_y)
plt.scatter(plot_X.T[0][plot_y <= 0], plot_X.T[1][plot_y <= 0], marker="o", color="blue", alpha=0.2)
plt.scatter(plot_X.T[0][plot_y > 0], plot_X.T[1][plot_y > 0], marker="o", color="red", alpha=0.2)
ax.scatter(X_train[:, 0], X_train[:, 1], color=colors_train)
ax.scatter(X_test[:, 0], X_test[:, 1], facecolor="None", edgecolor=colors_test, marker="D")
ax.plot(x_axis, y_axis_line)
plt.show()
