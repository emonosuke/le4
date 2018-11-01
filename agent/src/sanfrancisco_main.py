#!/usr/bin/env python
"""
csv データに関して GridSearch を行う
Usage:
    ./dat_main.py
"""

from utils import cross_val_regression
from svr import SVRegressor
import numpy as np
import pandas as pd
from scaler import StandardScaler
import sys
import matplotlib.pyplot as plt

def dollartofloat(s):
    """
    '$1,300.00' 等の価格を表す str を float へ変換
    """
    s = s.replace('$', '')
    s = s.replace(',', '')
    return float(s)

def addothers(s):
    """
    Apartment, House, Condominium, Guest suite 以外の type は Others とする
    """
    if s in ['Apartment', 'House', 'Condominium', 'Guest suite']:
        return s
    else:
        return 'Others'

if __name__ == '__main__':

    df = pd.read_csv('./data/San Francisco-listings.csv')

    # 特徴量は 7 つに絞り、1000個のデータをランダムサンプリングする
    df = df[['latitude', 'longitude', 'accommodates', 'property_type', 'room_type', 'number_of_reviews', 'review_scores_rating', 'price']]
    df = df.sample(n=500)

    # y を作成
    prices = df['price'].values
    y = np.array([dollartofloat(p) for p in prices])

    # 欠損値(review_scores_rating)を 0 へ変換
    df = df.fillna(0)

    # 数値データはそのまま X へ
    df_processed = df[['latitude', 'longitude', 'accommodates', 'number_of_reviews', 'review_scores_rating']]
    X = df_processed.values

    # room_type は onehot vector に変換して X へ merge
    df_room = pd.get_dummies(df[['room_type']], drop_first=True)
    X = np.hstack((X, df_room.values))

    # property_type は Apartment, House, Condominium, Guest Suit, Others に分類したのち、one-hot vector に変換して X へ merge
    df_property = df[['property_type']]
    df_property = df_property.applymap(addothers)
    df_property = pd.get_dummies(df_property, drop_first=True)
    X = np.hstack((X, df_property.values))

    X = X.astype(np.float32)
    # 確認
    print(type(X))
    print(X)
    print("X: ", X.shape, "y: ", y.shape)

    # X に StandardScalerを適用
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # Gauss カーネルのパラメータ p と C に関して Grid Search を行う
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    plist = [0.5, 1.0, 5.0]
    clist = [1.0, 10.0, 100.0, 1000.0]

    max_score = 0
    for idx, c in enumerate(clist):
        p_score = []
        for p in plist:
            print("=== Grid Search p={}, c={} ===".format(p, c))
            svr = SVRegressor(ker_type='-g', p=p, c=c, eps=0.1)
            score = cross_val_regression(X, y, svr, k=5)
            p_score.append(score)
        max_score = max(max_score, max(p_score))
        plt.plot(plist, p_score, color=colors[idx], marker='o', label='c={}'.format(c))
    
    plt.xlabel("parameter C")
    plt.ylabel("MSE score")
    plt.legend()
    plt.show()
    print("max score: ", max_score)

