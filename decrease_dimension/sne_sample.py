from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

data = load_digits()
n_components = 3  # 削減後の次元を2に設定
model = TSNE(n_components=n_components)
result = model.fit_transform(data.data)

print(data.data)
print(data.data[0])
print(data.data[1])
print(data.data.size)
print(result)

"""
figure2d = plt.figure()
ax = figure2d.add_subplot(111)
ax.scatter(result[:, 0], result[:, 1])
plt.show()
"""

figure3d = plt.figure()
ax = Axes3D(figure3d, elev=20, azim=-110)  # rotate graph
ax.scatter(result[:, 0], result[:, 1], result[:, 2])
plt.show()
