import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression

heldout = [0.95, 0.90, 0.75, 0.50, 0.01]
rounds = 20
digits = datasets.load_digits()
X, y = digits.data, digits.target

iris = datasets.load_iris()

clf = Perceptron();
xx = 1. - np.array(heldout)


rng = np.random.RandomState(42)
yy = []
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90, random_state=rng)
clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

clf.sparsify()

# plt.legend(loc="upper right")
# plt.xlabel("Proportion train")
# plt.ylabel("Test Error Rate")
# plt.show()

str = "0123456"
str[33:]
lst = [1,2,3,4]
lst.append

def f(ham: str, eggs: str = 'eggs') -> str:
     print("Annotations:", f.__annotations__)
     print("Arguments:", ham, eggs)
     return ham + ' and ' + eggs
 
