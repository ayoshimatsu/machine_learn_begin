from sklearn.svm import SVC
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# create data
X, y = make_gaussian_quantiles(n_features=2, n_classes=2, n_samples=300)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# predict result
model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_pred, y_test))
print(model.dual_coef_)
# create graph
fig = plt.figure(figsize=(6, 6))  # create plot object
ax = fig.add_subplot(111)
x_ax_min, x_ax_max = -3, 3
y_ax_min, y_ax_max = -3, 3
ax.set_xlim([x_ax_min, x_ax_max])
ax.set_ylim([y_ax_min, y_ax_max])
# show mesh in graph
x_mesh = np.arange(x_ax_min, x_ax_max, 0.05)
y_mesh = np.arange(y_ax_min, y_ax_max, 0.05)
X_mesh, Y_mesh = np.meshgrid(x_mesh, y_mesh)
mesh_plot_X = np.c_[X_mesh.ravel(), Y_mesh.ravel()]
mesh_predict_y = model.predict(mesh_plot_X)
# mesh_color_codes = {0: 'orange', 1: 'blue'}
# colors = [mesh_color_codes[x] for x in mesh_predict_y]
plt.scatter(mesh_plot_X.T[0][mesh_predict_y <= 0], mesh_plot_X.T[1][mesh_predict_y <= 0], marker="o", color="orange", alpha=0.2)
plt.scatter(mesh_plot_X.T[0][mesh_predict_y > 0], mesh_plot_X.T[1][mesh_predict_y > 0], marker="o", color="gray", alpha=0.2)
# set color and show train data
color_codes = {0: 'red', 1: 'green'}
colors = [color_codes[x] for x in y_train]
ax.scatter(X_train[:, 0], X_train[:, 1], color=colors)
# set color and show predict data
color_codes = {0: 'purple', 1: 'blue'}
colors = [color_codes[x] for x in y_pred]
ax.scatter(X_test[:, 0], X_test[:, 1], facecolor="None", edgecolor=colors, marker="D")

plt.show()
