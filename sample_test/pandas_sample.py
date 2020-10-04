import pandas as pd
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

data = load_wine()
df_X = pd.DataFrame(data.data, columns=data.feature_names)
df_y = pd.DataFrame(data.target, columns=["kind(target)"])
df = pd.concat([df_X, df_y], axis=1)
print(df.head())
print(df.corr())
print(df.describe())

plt.hist(df.loc[:, "alcohol"])
plt.show()


