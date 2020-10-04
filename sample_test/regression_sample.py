from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# pandas frame
# X_pdFrame = pd.DataFrame(data.data, columns=data.feature_names)

# take out elements to use from list
X_10th = X[:, :10]  # 1st ~ 10th

model = LogisticRegression()
model.fit(X_10th, y)
y_predict_10th = model.predict(X_10th)
print(accuracy_score(y, y_predict_10th))
