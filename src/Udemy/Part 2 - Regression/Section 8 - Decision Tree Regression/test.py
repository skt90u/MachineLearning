# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#
import pandas as pd
#
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
#
# from sklearn.tree import DecisionTreeRegressor
# regressor = DecisionTreeRegressor(random_state = 0)
# regressor.fit(X, y)
#
# y_pred = regressor.predict(6.5)
#
# import matplotlib.pyplot as plt
# plt.scatter(X, y, color = 'red')
# plt.plot(X, regressor.predict(X), color = 'blue')
# plt.title('Truth or Bluff (Decision Tree Regression)')
# plt.xlabel('Position')
# plt.ylabel('Salary')
# plt.show()
#
# import numpy as np
# X_grid = np.arange(min(X), max(X), 0.01)
# X_grid = X_grid.reshape(len(X_grid), 1)
# plt.scatter(X, y, color = 'red')
# plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
# plt.title('Truth or Bluff (Decision Tree Regression)')
# plt.xlabel('Position')
# plt.ylabel('Salary')
# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

s = pd.Series([1,3,5,np.nan,6,8])