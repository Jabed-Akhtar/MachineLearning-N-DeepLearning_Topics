from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

X, y, coefficients = make_regression(
    n_samples=50,
    n_features=1,
    n_informative=1,
    n_targets=1,
    noise=5,
    coef=True,
    random_state=1
)

alpha = 1

n, m = X.shape
I = np.identity(m)

w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + alpha * I), X.T), y)

#plt.scatter(X, y)
#plt.plot(X, w*X, color='red')

rr = Ridge(alpha=1)
rr.fit(X, y)
w = rr.coef_

#plt.scatter(X, y, color='blue')
#plt.plot(X, w*X, c='orange')
#plt.show()

rr = Ridge(alpha=10)
rr.fit(X, y)
w = rr.coef_[0]
#plt.scatter(X, y, color='blue')
#plt.plot(X, w*X, color='orange')
#plt.show()

rr = Ridge(alpha=100)
rr.fit(X, y)
w = rr.coef_[0]
plt.scatter(X, y, color='blue')
plt.plot(X, w*X, color='orange')
plt.show()
