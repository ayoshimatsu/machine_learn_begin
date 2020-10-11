from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

X = [[10.0], [8.0], [13.0], [9.0], [11.0], [14.0], [6.0], [4.0], [12.0], [7.0], [5.0]]
y = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]

# crete model
model = LinearRegression()
model.fit(X, y)

# debug. confirm the result.
y_pred = model.predict([[0], [1]])
print(model.intercept_)
print(model.coef_)
print(y_pred)

# Line of result. variables for graph.
x_line = np.arange(0, 17)
y_line = model.coef_ * x_line + model.intercept_
print(len(X))


# Create graph
fig = plt.figure(figsize=(6, 8))  # create plot object
ax = fig.add_subplot(211)
ax.scatter(X, y)
ax.plot(x_line, y_line, color="red")

# test for graph
ax2 = fig.add_subplot(212)
x_line2 = np.arange(-17, 17)
y_line2 = x_line2 ** 3 + 10 * x_line2 + 1
ax2.plot(x_line2, y_line2, color="green")

plt.show()
