from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# download data
data = load_wine()
X = data.data[:, [0, 9]]  # take out 1st and 10th data

# create cluster of model
model = KMeans(n_clusters=3)

# predict data
predict = model.fit_predict(X)

# set color
color_codes = {0: '#00FF00', 1: '#FF0000', 2: '#0000FF'}
colors = [color_codes[x] for x in predict]

# set data to graph
plt.scatter(X[:, 0], X[:, 1], color=colors)
plt.show()
