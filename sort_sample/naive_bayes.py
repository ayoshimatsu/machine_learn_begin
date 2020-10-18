from sklearn.naive_bayes import MultinomialNB
import numpy as np

# create matrix for learning
X_train = [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1]]

y_train = [1, 1, 1, 0, 0, 0]
model = MultinomialNB()
model.fit(X_train, y_train) # 学習
print(model.intercept_)
print(model.coef_)

learning_matrix = np.array(model.coef_)
data = [[1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]]
data_array = np.array(data).T
print(data_array)
print(np.dot(learning_matrix, data_array))
print(model.predict(data))
