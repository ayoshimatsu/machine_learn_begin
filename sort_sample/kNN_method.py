from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# create data
X, y = make_moons(noise=0.3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train)
print(y_train)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)  # 学習
y_pred = model.predict(X_test)
accuracy_score(y_pred, y_test)  # 評価

# frame of graph
fig = plt.figure(figsize=(6, 6))  # create plot object
ax = fig.add_subplot(111)
x_ax_min, x_ax_max = -2, 3
y_ax_min, y_ax_max = -1.5, 1.5
ax.set_xlim([x_ax_min, x_ax_max])
ax.set_ylim([y_ax_min, y_ax_max])
# create mesh grid
x_mesh = np.arange(x_ax_min, x_ax_max, 0.05)
y_mesh = np.arange(y_ax_min, y_ax_max, 0.05)
X_mesh, Y_mesh = np.meshgrid(x_mesh, y_mesh)
mesh_plot_X = np.c_[X_mesh.ravel(), Y_mesh.ravel()]
mesh_predict_y = model.predict(mesh_plot_X)
plt.scatter(mesh_plot_X.T[0][mesh_predict_y <= 0], mesh_plot_X.T[1][mesh_predict_y <= 0], marker="o", color="orange", alpha=0.2)
plt.scatter(mesh_plot_X.T[0][mesh_predict_y > 0], mesh_plot_X.T[1][mesh_predict_y > 0], marker="o", color="cyan", alpha=0.2)
# learning data
color_codes = {0: 'red', 1: 'green'}
colors = [color_codes[x] for x in y_train]
ax.scatter(X_train[:, 0], X_train[:, 1], color=colors)
plt.show()