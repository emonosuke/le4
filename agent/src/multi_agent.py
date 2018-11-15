#!/usr/bin/env python

from utils import load_sanfrancisco, cross_val_regression
from scaler import StandardScaler
from svr import SVRegressor
import numpy as np
import sys

data_num = 2000
hf = data_num // 2

X, y = load_sanfrancisco(data_num)

# 前半を訓練データ, 後半を評価データとする(load_sanfranciscoの時点でランダムサンプリングされている)
X_train = X[:hf]
y_train = y[:hf]
X_test = X[hf:]
y_test = y[hf:]

# 正規化する
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

def SVRAgent(ker_type='-g', p=9.5, c=27000, eps=0.09):
    svr = SVRegressor(ker_type, p, c, eps)

    # cross_val_regression で正規化したのち交差検証している
    score = cross_val_regression(X_train, y_train, svr, mth='R2')
    print('cross val score: ', score)

    print('start SVR fit...')
    svr.fit(X_train_std, y_train)

    print('start SVR predict...')
    y_pred = svr.predict(X_test_std)

    y_offer = y_offer = 0.85 * y_pred

    return y_offer

print('=== SVRAgent1 ===')
y_offer1 = SVRAgent(ker_type='-g', p=9.5, c=27000, eps=0.09)
print('=== SVRAgent2 ===')
y_offer2 = SVRAgent(ker_type='-g', p=1.0, c=1.0, eps=0.1)

# 利益
# リストの価格の30%は原価とする
revenue1 = 0
revenue2 = 0

cnt1 = 0
cnt2 = 0

for i in range(hf):
    minprice = min(min(y_offer1[i], y_offer2[i]), y_test[i])
    if minprice == y_offer1[i]:
        cnt1 += 1
        revenue1 += (minprice - y_test[i] * 0.3)
    elif minprice == y_offer2[i]:
        cnt2 += 1
        revenue2 += (minprice - y_test[i] * 0.3)

print("revenue by SVR agent1: ", revenue1)
print("the agent1 wins {}/{} times".format(cnt1, hf))
print("revenue1 per 1: ", revenue1 / cnt1)
print("revenue by SVR agent2: ", revenue2)
print("the agent2 wins {}/{} times".format(cnt2, hf))
print("revenue2 per 1: ", revenue2 / cnt2)
