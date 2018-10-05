import numpy as np
import cvxopt
from cvxopt import matrix
from utils import load_data

eps = 1e-12

# d = 2
def kernel(a, b):
    return (1.0 + np.dot(a, b)) * (1.0 + np.dot(a, b))

# Lagrange 乗数 α_k の値を求める
# in: X, y
def solve(X, y):
    # X と y の shape が異なる場合に ERROR としたい
    num = y.shape[0]
    tP = np.zeros((num, num)) # size num * num の numpy 配列を初期化
    for i in range(num):
        for j in range(num):
            tP[i][j] = y[i] * y[j] * np.dot(X[i], X[j])
    P = matrix(tP)
    q = matrix(-np.ones(num))
    G = matrix(np.diag([-1.0] * num))
    h = matrix(np.zeros(num))
    A = matrix(y, (1, num))
    b = matrix(0.0)

    # minimize (1/2)x^T * P * x + q^T * x
    # subject to G * x <= h, A * x = b
    sol = cvxopt.solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)
    print(sol)
    print(sol["x"])
    print(sol["primal objective"])
    return sol["x"]

# Lagrange 乗数 α_k から識別器のパラメータ w, th を得る 
def get_params(X, y, alpha):
    num = y.shape[0]
    w = 0
    th = 0
    for i in range(num):
        w += alpha[i] * y[i] * X[i]
    for i in range(num):
        if alpha[i] > 1e-6:
            th = np.dot(w, X[i]) - y[i]
            print(i, th)
        else:
            print(i, "a_k = 0")
    return w, th

def get_ker_params(X, y, alpha):
    num = y.shape[0]
    for j in range(num):
        if alpha[j] < 1e-6: continue
        th = 0
        for i in range(num):
            th += alpha[i] * y[i] * kernel(X[i], X[j])
        print(j, th - y[j])
    return 1, th


X, y = load_data('./sample_linear.dat')
ans = solve(X, y)
print(ans)
w, th = get_params(X, y, np.array(ans))

"""
X = np.array([[1.], [2.], [5.], [7.], [8.]])
y = np.array([1., 1., -1., -1., 1.])
ans = solve(X, y)
print(ans)
w, th = get_ker_params(X, y, np.array(ans))
"""
