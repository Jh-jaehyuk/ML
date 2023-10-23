from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = fetch_california_housing()
x_data = dataset.data
y_data = dataset.target
#print(x_data.shape) (20640, 8)
#print(y_data.shape) (20640,)

plt.figure(1)
plt.plot(x_data, y_data, 'ro')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Dataset')
plt.show()


x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, test_size=0.3, random_state=1231)
estimator = LinearRegression()
estimator.fit(x_train, y_train)

y_predict = estimator.predict(x_train)
score = metrics.r2_score(y_train, y_predict)
print(score)
