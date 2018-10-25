import numpy as np
import cvxopt
from cvxopt import matrix
from utils import load_data

class SVRegressor(object):
    """
    SVR(Support Vector Regressor) を実装する
    """

    def __init__(self):
        print('Initialize SVR')

    def fit(self, X, y):
        self.__n = y.shape[0]
        self.__X = X
        self.__y = y
        self.__eps = 0.1
        self.__C = 1000.0
        # self.__a(alpha), self.__b(alpha*) を設定する 
        self.__setLagrange()
        # self.__w, self.__theta を設定する
        self.__setClassifier()    
    
    def predict(self, tX):
        pass
    
    def __setLagrange(self):
        tP = np.zeros((self.__n*2, self.__n*2)) # size 2n * 2n の numpy 配列 tP を 0 で初期化
        for i in range(self.__n*2):
            for j in range(self.__n*2):
                if i < self.__n and j < self.__n:
                    tP[i][j] = np.dot(self.__X[i], self.__X[j])
                elif i < self.__n and j >= self.__n:
                    tP[i][j] = (-1) * np.dot(self.__X[i], self.__X[j - self.__n])
                elif i >= self.__n and j < self.__n:
                    tP[i][j] = (-1) * np.dot(self.__X[i - self.__n], self.__X[j])
                else:
                    # i >= self.__n and j >= self.__n
                    tP[i][j] = np.dot(self.__X[i - self.__n], self.__X[j - self.__n])
        
        P = matrix(tP)

        tq = []
        for i in range(self.__n):
            tq.append(self.__eps - self.__y[i])
        for i in range(self.__n):
            tq.append(self.__eps + self.__y[i])
        q = matrix(np.array(tq))
        g1 = matrix(np.diag([-1.0] * self.__n*2))
        g2 = matrix(np.diag([1.0] * self.__n*2))
        G = matrix(np.concatenate((g1, g2), axis=0))
        h1 = np.zeros(self.__n*2)
        h2 = np.array([self.__C for i in range(self.__n*2)])
        h = matrix(np.concatenate((h1, h2), axis=0))
        a1 = np.array([-1.0] * self.__n)
        a2 = np.array([1.0] * self.__n)
        tA = np.concatenate((a1, a2), axis=0)
        A = matrix(tA, (1, self.__n*2))
        b = matrix(0.0)

        # minimize (1/2)x^TPx + q^Tx
        # subject to Gx <= h, Ax = b
        sol = cvxopt.solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)
        if sol['status'] == 'optimal':
            print("cvxopt.solvers.qp: optimization succeeded")
        else:
            print("cvxopt.solvers.qp: optimization failed")
            sys.exit(1)
        ans = np.array(sol["x"])
        # a, bを確認する
        print("alpha: ")
        for idx, a in enumerate(ans):
            print(idx, a)
    
    def __setClassifier(self):
        pass

if __name__ == "__main__":
    input_file = './data/sample40.dat'
    X, y = load_data(input_file)

    svr = SVRegressor()
    svr.fit(X, y)
