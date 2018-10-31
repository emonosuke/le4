import numpy as np
from svr import SVRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# #############################################################################
# Generate sample data
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

# #############################################################################
# Add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))

# #############################################################################
# Fit regression model
svr_rbf = SVRegressor(ker_type='-g', p=0.1)
svr_lin = SVRegressor(ker_type='-n', p=0.1)
svr_poly = SVRegressor(ker_type='-p', p=2)

svr_rbf.fit(X, y, c=1e3, eps=0.1)
svr_lin.fit(X, y, c=1e3, eps=0.1)
svr_poly.fit(X, y, c=1e3, eps=0.1)
y_rbf = svr_rbf.predict(X)
y_lin = svr_lin.predict(X)
y_poly = svr_poly.predict(X)

# #############################################################################
# Look at the results
lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

plt.scatter(X, y, color='darkorange', label='data')
svr_sk = 