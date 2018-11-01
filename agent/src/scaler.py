import numpy as np

class MinMaxScaler(object):
    """
    データの正規化を行う
    x_norm = (x(i) - x_min) / (x_max - x_min)
    """
    def fit(self, X):
        self.__min = np.min(X, axis=0)
        self.__max = np.max(X, axis=0)
    def transform(self, X):
        X_norm = (X - self.__min) / (self.__max - self.__min)
        return X_norm
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class StandardScaler(object):
    """
    データの標準化を行う
    x_std = (x(i) - x_mean) / x_std
    """
    def fit(self, X):
        self.__mean = np.mean(X, axis=0)
        self.__std = np.std(X, axis=0)
    def transform(self, X):
        # 0 除算を防ぐため微小値を加算
        X_std = (X - self.__mean) / (self.__std + 1e-8)
        return X_std
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

if __name__ == '__main__':
    X = np.array([[5, 2, 3, 4], [3, 3, 3, 3], [1, 2, 3, 4]])
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    print(X_std)
