from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)
model = RandomForestClassifier()
model.fit(X_train, y_train)  # 学習
y_pred = model.predict(X_test)

print(model.criterion)
print(model.n_estimators)
print(data.feature_names)
print(model.feature_importances_)
print(y_test)
print(y_pred)
print(accuracy_score(y_pred, y_test))

fig = plt.figure(figsize=(6, 6))  # create plot object
ax = fig.add_subplot(111)
x = np.arange(len(model.feature_importances_))

ax.bar(x, model.feature_importances_, 1)
plt.show()
