from sklearn.datasets import samples_generator
from sklearn.manifold import LocallyLinearEmbedding
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

data, color = samples_generator.make_swiss_roll(n_samples=1500)
n_neighbors = 15  # 近傍点の数
n_components = 2  # 削減後の次元数
model = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components)
model.fit(data)

result = model.transform(data)
change = model.transform([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
print(data)
print(change)
print(change.T)

test = np.dot(data, change)

print(result)
print(test)

figure3d = plt.figure()
# 3D graph
ax = Axes3D(figure3d, elev=20, azim=-110)  # rotate graph
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color)
# 2D graph
figure2d = plt.figure()
ax = figure2d.add_subplot(111)
ax.scatter(result[:, 0], result[:, 1], c=color)
figure3d = plt.figure()
ax = figure3d.add_subplot(111)
ax.scatter(test[:, 0], test[:, 1], c=color)
plt.show()
