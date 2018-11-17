from utils import load_sanfrancisco, cross_val_regression
from scaler import StandardScaler
from svr import SVRegressor
import numpy as np
import sys
import math

data_num = 2000
hf = data_num // 2

X, y = load_sanfrancisco(data_num)

# 前半を訓練データ, 後半を評価データとする(load_sanfranciscoの時点でランダムサンプリングされている)
X_train = X[:hf]
y_train = y[:hf]

svr = SVRegressor(ker_type='-g', p=1.0, c=1.0, eps=0.1)

mse = cross_val_regression(X_train, y_train, svr, mth='MSE')
rmse = math.sqrt(mse)
print('RMSE score: ', rmse)
mae = cross_val_regression(X_train, y_train, svr, mth='MAE')
print('MAE score: ', mae)
print('RMSE / MAE: ', rmse / mae)
