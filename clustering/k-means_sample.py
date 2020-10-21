from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

data = load_iris()
n_clusters = 3  # クラスタ数を3に設定
model = KMeans(n_clusters=n_clusters)
model.fit(data.data)
print(model.labels_)  # 各データ点が所属するクラスタ
print(model.cluster_centers_)  # fit()によって計算された重心
