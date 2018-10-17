import numpy as np

class MinMaxScaler(object):
    """
    データの正規化を行う
    """
    def fit(self, X):
        self.__min = np.min(X, axis=0)
        self.__max = np.max(X, axis=0)
    def transform(self, X):
        X_std = (X - self.__min) / (self.__max - self.__min)
        return X_std
    def fit_transform(self, X):
        self.fit(X)
        X_std = self.transform(X)
        return X_std

if __name__ == '__main__':
    X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    sc = MinMaxScaler()
    sc.fit(X)
    X = sc.transform(X)
    print(X)
