from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

data = load_iris()
n_components = 2 # 削減後の次元を2に設定
model = PCA(n_components=n_components)
model = model.fit(data.data)
print(model.transform(data.data)) # 変換したデータ
print(data.feature_names)
print(model.get_covariance())
print(model.explained_variance_ratio_)
print(model.components_[0])
print(model.components_[1])
