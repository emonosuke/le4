import numpy as np
import cvxopt
from cvxopt import matrix

class SVClassifier(object):
    """
    private: 
    X, y, n - 訓練データ n はその個数
    kf(kernel function), p, q - kernel function のそのパラメータ
    alpha, th - 識別器の内部パラメータ
    """

    def __init__(self, ker_type, p, q):
        print(p, q)
        self.__setKernelFunc(ker_type)
        self.__p = p
        self.__q = q

    def fit(self, X, y):
        self.__n = y.shape[0]
        self.__X = X
        self.__y = y
        # self.__alpha を設定する 
        self.__setLagrange()
        # self.__th を設定する
        self.__setClassifier()
    
    def predict(self, tX):
        batch_size = tX.shape[0]
        preds = []
        for i in range(batch_size):
            res = 0
            for j in range(self.__n):
                res += self.__alpha[j] * self.__y[j] * self.__kf(self.__X[j], tX[i])
            res -= self.__th
            preds.append(np.sign(res))
        return np.array(preds)
    
    def __setLagrange(self):
        tP = np.zeros((self.__n, self.__n)) # size n * n の numpy 配列 tP を 0 で初期化
        for i in range(self.__n):
            for j in range(self.__n):
                tP[i][j] = self.__y[i] * self.__y[j] * self.__kf(self.__X[i], self.__X[j])
        
        P = matrix(tP)
        q = matrix(-np.ones(self.__n))
        G = matrix(np.diag([-1.0] * self.__n))
        h = matrix(np.zeros(self.__n))
        A = matrix(self.__y, (1, self.__n))
        b = matrix(0.0)

        # minimize (1/2)x^TPx + q^Tx
        # subject to Gx <= h, Ax = b
        sol = cvxopt.solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)

        self.__alpha = np.array(sol["x"])
        print("alpha: ")
        for i in range(self.__n):
            print(i, self.__alpha[i])
    
    def __setKernelFunc(self, ker_type):
        if ker_type == 'p':
            self.__kf = self.__kernelPolynomial
        elif ker_type == 'g': 
            self.__kf = self.__kernelGauss
        elif ker_type == 's':
            self.__kf = self.__kernelSigmoid
        else:
            # expect ker_type == 'n'
            self.__kf = lambda a, b: np.dot(a, b)
    
    def __setClassifier(self):
        ths = []
        for j in range(self.__n):
            if self.__alpha[j] < 1e-6: continue
            th = 0
            for i in range(self.__n):
                th += self.__alpha[i] * self.__y[i] * self.__kf(self.__X[i], self.__X[j])
            th -= self.__y[j]
            ths.append(th)
        print("theta candidates: ", [ths[i][0] for i in range(len(ths))])
        self.__th = np.mean(ths)

    def __kernelPolynomial(self, a, b):
        return (1.0 + np.dot(a, b)) ** self.__p
    
    def __kernelGauss(self, a, b):
        return np.exp(-(np.linalg.norm(a - b) ** 2) / (2 * self.__p ** 2))

    def __kernelSigmoid(self, a, b):
        return np.tanh(self.__p * np.dot(a, b) + self.__q)
