from sklearn.decomposition import TruncatedSVD

data = [[1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1]]

n_components = 2 # 潜在変数の数
model = TruncatedSVD(n_components=n_components)
model.fit(data)
print(model.transform(data)) # 変換したデータ
print(model.explained_variance_ratio_) # 寄与率
print(sum(model.explained_variance_ratio_)) # 累積寄与率