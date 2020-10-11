import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

train_size = 20
test_size = 12
train_X = np.random.uniform(low=0, high=1.2, size=train_size)  # create random numbers
test_X = np.random.uniform(low=0.1, high=1.3, size=test_size)  # create random numbers
train_X.sort()
test_X.sort()
train_y = np.sin(train_X * 2 * np.pi) + np.random.normal(0, 0.2, train_size)  # + standard deviation
test_y = np.sin(test_X * 2 * np.pi) + np.random.normal(0, 0.2, test_size)

poly = PolynomialFeatures(6)  # 6 degree
train_poly_X = poly.fit_transform(train_X.reshape(train_size, 1))  # ?
test_poly_X = poly.fit_transform(test_X.reshape(test_size, 1))  # ?

model = Ridge(alpha=0.1)
model.fit(train_poly_X, train_y)
train_pred_y = model.predict(train_poly_X)
test_pred_y = model.predict(test_poly_X)

print(train_poly_X)

print(mean_squared_error(train_pred_y, train_y))
print(mean_squared_error(test_pred_y, test_y))

print(model.intercept_)
print(model.coef_)

# create graph
fig = plt.figure(figsize=(10, 6))  # create plot object
ax = fig.add_subplot(111)
ax.set_ylim([-2, 2])
x_sin = np.linspace(0, 1.4, 201)
ax.plot(x_sin, np.sin(x_sin * 2 * np.pi), color="green")
ax.plot(x_sin, (x_sin ** 6) * model.coef_[6] + (x_sin ** 5) * model.coef_[5] + (x_sin ** 4) * model.coef_[4]
        + (x_sin ** 3) * model.coef_[3] + (x_sin ** 2) * model.coef_[2] + (x_sin ** 1) * model.coef_[1]
        + model.intercept_)
ax.scatter(train_X, train_y, color="gray")
ax.plot(train_X, train_pred_y, color="red")
ax.plot(test_X, test_pred_y, color="yellow")

plt.show()
