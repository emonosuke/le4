import numpy as np
import cvxopt
from cvxopt import matrix
import tqdm
import sys

class HardSVC(object):
    """
    ハードマージンSVMを実装する

    private: 
    X, y, n - 訓練データ n はその個数
    kf(kernel function), p - kernel function のそのパラメータ
    alpha, theta - SVM の内部パラメータ
    """

    def __init__(self, ker_type, p):
        # defalut 値で初期化
        self.__p = 2.0
        if ker_type == '-p':
            self.__kf = self.__kernelPolynomial
        elif ker_type == '-g': 
            self.__kf = self.__kernelGauss
        else:
            # expect ker_type == 'n'
            self.__kf = lambda a, b: np.dot(a, b)
        
        self.__p = p

    def fit(self, X, y):
        self.__n = y.shape[0]
        self.__X = X
        self.__y = y
        # self.__alpha を設定する 
        self.__setLagrange()
        # self.__theta を設定する
        self.__setClassifier()    
    
    def predict(self, tX):
        if tX.ndim == 1:
            batch_size = 1
        else:
            batch_size = tX.shape[0]
        preds = []
        ay = np.array([self.__alpha[j] * self.__y[j] for j in range(self.__n)])
        ay = np.reshape(ay, (-1,))
        # print("model predicting: ")
        for i in range(batch_size):
            K = self.__kf(self.__X, tX[i])
            preds.append(np.sign(np.dot(ay, K) - self.__theta))
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
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)
        if sol['status'] == 'optimal':
            print("cvxopt.solvers.qp: optimization succeeded")
        else:
            print("cvxopt.solvers.qp: optimization failed")
            sys.exit(1)
        self.__alpha = np.array(sol["x"])
        # print("alpha: ")
        # for i in range(self.__n):
        #    print(i, self.__alpha[i])
    
    def __setClassifier(self):
        ths = []
        for j in range(self.__n):
            if abs(self.__alpha[j]) < 1e-6: continue # サポートベクターのみ考える
            th = 0
            for i in range(self.__n):
                th += self.__alpha[i] * self.__y[i] * self.__kf(self.__X[i], self.__X[j])
            th -= self.__y[j]
            ths.append(th)
        
        W = 0
        for i in range(self.__n):
            W += self.__alpha[i] * self.__y[i] * self.__X[i]

        # print("theta candidates: ", [ths[i][0] for i in range(len(ths))])
        self.__theta = np.mean(ths)
        # W = , θ = の形で識別器を出力する
        print("HardSVM W = {}, θ = {}".format(W, self.__theta))

    def __kernelPolynomial(self, a, b):
        return (1.0 + np.dot(a, b)) ** self.__p
    
    def __kernelGauss(self, a, b):
        if a.ndim == 1:
            return np.exp(-(np.linalg.norm(a - b) ** 2) / (2 * self.__p ** 2))
        return np.exp(-(np.linalg.norm(a - b, axis=1) ** 2) / (2 * self.__p ** 2))

class SoftSVC(object):
    """
    ソフトマージンSVMを実装する

    private: 
    X, y, n - 訓練データ n はその個数
    kf(kernel function), p - kernel function のそのパラメータ
    c - ソフトマージンのためのハイパーパラメータ
    alpha, theta - SVM の内部パラメータ
    """

    def __init__(self, ker_type, p, c):
        # defalut 値で初期化
        self.__p = 2.0
        if ker_type == '-p':
            self.__kf = self.__kernelPolynomial
        elif ker_type == '-g': 
            self.__kf = self.__kernelGauss
        else:
            # expect ker_type == 'n'
            self.__kf = lambda a, b: np.dot(a, b)
        
        self.__p = p
        self.__C = c

    def fit(self, X, y):
        self.__n = y.shape[0]
        self.__X = X
        self.__y = y
        # self.__alpha を設定する 
        self.__setLagrange()
        # self.__theta を設定する
        self.__setClassifier()    
    
    def predict(self, tX):
        if tX.ndim == 1:
            batch_size = 1
        else:
            batch_size = tX.shape[0]
        preds = []
        ay = np.array([self.__alpha[j] * self.__y[j] for j in range(self.__n)])
        ay = np.reshape(ay, (-1,))
        # print("model predicting: ")
        for i in range(batch_size):
            K = self.__kf(self.__X, tX[i])
            preds.append(np.sign(np.dot(ay, K) - self.__theta))
        return np.array(preds)
    
    def __setLagrange(self):
        tP = np.zeros((self.__n, self.__n)) # size n * n の numpy 配列 tP を 0 で初期化
        for i in range(self.__n):
            for j in range(self.__n):
                tP[i][j] = self.__y[i] * self.__y[j] * self.__kf(self.__X[i], self.__X[j])
        
        P = matrix(tP)
        q = matrix(-np.ones(self.__n))
        # 0 <= alpha <= C
        g1 = np.diag([-1.0] * self.__n)
        g2 = np.diag([1.0] * self.__n)
        G = matrix(np.concatenate((g1, g2), axis=0))
        h1 = np.zeros(self.__n)
        h2 = np.array([self.__C for i in range(self.__n)])
        h = matrix(np.concatenate((h1, h2), axis=0))
        A = matrix(self.__y, (1, self.__n))
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
        self.__alpha = np.array(sol["x"])
        # print("alpha: ")
        # for i in range(self.__n):
        #    print(i, self.__alpha[i])
    
    def __setClassifier(self):
        ths = []

        # 自由サポートベクターのインデックス取得
        # 0 < alpha < C を満たす
        support_vectors = []

        for idx in range(self.__n):
            # 丸め誤差を考慮する
            if self.__alpha[idx] > 1e-6 and self.__alpha[idx] < 0.99 * self.__C:
                support_vectors.append(idx)

        print("free support vectors: ")
        print(support_vectors)
        for sv in support_vectors:
            th = 0
            for j in range(self.__n):
                th += self.__alpha[j] * self.__y[j] * self.__kf(self.__X[sv], self.__X[j])
            th -= self.__y[sv]
            ths.append(th)

        W = 0
        for i in range(self.__n):
            W += self.__alpha[i] * self.__y[i] * self.__X[i]

        print("theta candidates: ", [ths[i][0] for i in range(len(ths))])
        self.__theta = np.mean(ths)

        # W = , θ = の形で識別器を出力する
        print("SoftSVM W = {}, θ = {}".format(W, self.__theta))

    def __kernelPolynomial(self, a, b):
        return (1.0 + np.dot(a, b)) ** self.__p
    
    def __kernelGauss(self, a, b):
        if a.ndim == 1:
            return np.exp(-(np.linalg.norm(a - b) ** 2) / (2 * self.__p ** 2))
        return np.exp(-(np.linalg.norm(a - b, axis=1) ** 2) / (2 * self.__p ** 2))
