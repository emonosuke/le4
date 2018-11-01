import numpy as np
import cvxopt
from cvxopt import matrix
from utils import load_data
import math

class SVRegressor(object):
    """
    SVR(Support Vector Regressor) を実装する

    Attributes
    ----------
    """

    def __init__(self, ker_type, p, c, eps):
        """
        カーネルに用いる関数とそのパラメータを設定する
        """
        ker_dict = {
            '-n': lambda a, b: np.dot(a, b),
            '-p': self.__kernelPolynomial,
            '-g': self.__kernelGauss
            }
        self.__kf = ker_dict[ker_type]
        self.__p = p
        self.__C = c
        self.__eps = eps

    def fit(self, X, y):
        self.__n = y.shape[0]
        self.__X = X
        self.__y = y
        # self.__a(alpha), self.__b(alpha*) を設定する 
        self.__setLagrange()
        # self.__bias を設定する
        self.__setClassifier()    
    
    def predict(self, tX):
        if tX.ndim == 1:
            batch_size = 1
        else:
            batch_size = tX.shape[0]
        preds = []
        ab = np.array([(self.__a[j] - self.__b[j]) for j in range(self.__n)])
        ab = np.reshape(ab, (-1, ))

        for i in range(batch_size):
            K = self.__kf(self.__X, tX[i])
            preds.append(np.dot(ab, K) - self.__bias)
        return np.array(preds)
    
    def score(self, tX, tY, mth='MSE'):
        predY = self.predict(tX)

        if len(predY) != len(tY):
            print("SVRegressor::predict: testY and predY length don't match")
            return None
        
        num = len(tY)
        
        if mth == 'MSE':
            # 平均二乗誤差
            diff = [(tY[i] - predY[i]) for i in range(num)] 
            return (1/num) * np.dot(diff, diff)
        elif mth == 'MAE':
            # 平均絶対誤差
            diff = [math.abs(tY[i] - predY[i]) for i in range(num)]
            return (1/num) * np.sum(diff)
        elif mth == 'R2':
            # 決定係数
            midY = np.mean(tY)
            udiff = [(tY[i] - predY[i]) for i in range(num)]
            ddiff = [(tY[i] - midY) for i in range(num)]
            return 1.0 - np.dot(udiff, udiff) / np.dot(ddiff, ddiff)
        
    def __setLagrange(self):
        tP = np.zeros((self.__n*2, self.__n*2)) # size 2n * 2n の numpy 配列 tP を 0 で初期化
        for i in range(self.__n*2):
            for j in range(self.__n*2):
                if i < self.__n and j < self.__n:
                    tP[i][j] = self.__kf(self.__X[i], self.__X[j])
                elif i < self.__n and j >= self.__n:
                    tP[i][j] = (-1) * self.__kf(self.__X[i], self.__X[j - self.__n])
                elif i >= self.__n and j < self.__n:
                    tP[i][j] = (-1) * self.__kf(self.__X[i - self.__n], self.__X[j])
                else:
                    # i >= self.__n and j >= self.__n
                    tP[i][j] = self.__kf(self.__X[i - self.__n], self.__X[j - self.__n])
        
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
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)
        if sol['status'] == 'optimal':
            print("cvxopt.solvers.qp: optimization succeeded")
        else:
            print("cvxopt.solvers.qp: optimization failed")
            sys.exit(1)
        res = np.array(sol["x"])
        self.__a = res[0 : self.__n]
        self.__b = res[self.__n : self.__n*2]
        """
        print("alpha: ")
        for idx, ai in enumerate(self.__a):
            print(idx, ai)
        print("alpha*: ")
        for idx, bi in enumerate(self.__b):
            print(idx, bi)
        """
    
    def __setClassifier(self):
        biases = []

        a_sv = []
        b_sv = []

        for idx in range(self.__n):
            if self.__a[idx] > 0.01 and self.__a[idx] < 0.99 * self.__C:
                a_sv.append(idx)
            if self.__b[idx] > 0.01 and self.__b[idx] < 0.99 * self.__C:
                b_sv.append(idx)
        
        """
        print("support vectors from alpha: ", a_sv)
        print("support vectors from alpha*: ", b_sv)
        """
        
        # 0 < a_n < C を満たす a_n について、 bias = -y_n + eps + sum((a_k - b_k) * kf(x_n, x_k))
        for sv in a_sv:
            bias = -self.__y[sv] + self.__eps
            for j in range(self.__n):
                bias += (self.__a[j] - self.__b[j]) * self.__kf(self.__X[sv], self.__X[j])
            biases.append(bias)

        for sv in b_sv:
            bias = -self.__y[sv] - self.__eps
            for j in range(self.__n):
                bias += (self.__a[j] - self.__b[j]) * self.__kf(self.__X[sv], self.__X[j])
            biases.append(bias)
        
        # print("biases candidates: ", [biases[i][0] for i in range(len(biases))])
        self.__bias = np.mean(biases)
    
    def __kernelPolynomial(self, a, b):
        return (1.0 + np.dot(a, b)) ** self.__p
    
    def __kernelGauss(self, a, b):
        if a.ndim == 1:
            return np.exp(-(np.linalg.norm(a - b) ** 2) / (2 * self.__p ** 2))
        return np.exp(-(np.linalg.norm(a - b, axis=1) ** 2) / (2 * self.__p ** 2))

# for DEBUG
if __name__ == "__main__":
    input_file = './data/sample40.dat'
    X, y = load_data(input_file)

    svr = SVRegressor()
    svr.fit(X, y)
