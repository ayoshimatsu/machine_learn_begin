import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Data about two condition
X_train = np.r_[np.random.normal(3, 1, size=50), np.random.normal(-1, 1, size=50)].reshape((100, -1))
y_train = np.r_[np.ones(50), np.zeros(50)]

model = LogisticRegression()
model.fit(X_train, y_train)
print(model.coef_)
print(model.intercept_)
print(model.predict_proba([[0], [1], [2]])[:, 1])
# create graph
fig = plt.figure(figsize=(6, 6))  # create plot object
ax = fig.add_subplot(111)
x_axis = np.linspace(-4, 4, 201).reshape((201, -1))  # change to matrix(vector)
y_axis = 1 / (1 + np.exp(-(model.coef_ * x_axis + model.intercept_)))  # sigmoid formula
ax.plot(x_axis, y_axis, color="green", linestyle="dashed")
ax.scatter(X_train, y_train, color="gray", marker="D")
ax.scatter([[0], [1], [2]], model.predict_proba([[0], [1], [2]])[:, 1], color="red")
plt.show()
