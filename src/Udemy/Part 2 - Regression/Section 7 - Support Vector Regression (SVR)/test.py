#!/usr/bin/env python3

import pandas as pd
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

a = sc_X.fit_transform(X)
b = sc_y.fit_transform(y)


X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


c = sc_X.inverse_transform(a)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

import numpy as np
#y_pred = regressor.predict(6.5)

x_test = 6.5
x_test_scaled = sc_X.transform(np.array([x_test]))
y_pred_scaled = regressor.predict(x_test_scaled)
y_pred = sc_y.inverse_transform(y_pred_scaled)

import matplotlib.pyplot as plt
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()