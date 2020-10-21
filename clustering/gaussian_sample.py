from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture

data = load_iris()
n_components = 3 # ガウス分布の数
model = GaussianMixture(n_components=n_components)
model.fit(data.data)
print(data.feature_names)
print(data.data)
print(data.target)
print(model.predict(data.data)) # クラスを予測
print(model.means_) # 各ガウス分布の平均
print(model.covariances_) # 各ガウス分布の分散
