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

    # prices を float へ変換
    df['price'] = df['price'].map(dollartofloat)

    # prices > 500 のデータを削除
    df.drop(df.index[df.price > 500], inplace=True)

    # 特徴量は 7 つに絞り、500個のデータをランダムサンプリングする
    df = df[['latitude', 'longitude', 'accommodates', 'property_type', 'room_type', 'number_of_reviews', 'review_scores_rating', 'price']]
    df = df.sample(n=500)

    # y を作成
    y = np.array(df['price'].values)

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

    # Gauss カーネルのパラメータ p と C に関して Grid Search を行う
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # [2^-4, 2^-2, 2^0, 2^2, 2^4]
    plist = [pow(2, i) for i in range(-2, 9, 2)]
    logplist = [i for i in range(-2, 9, 2)]
    clist = [10.0, 100.0, 1000.0, 10000.0]

    maxscore = 0
    for idx, c in enumerate(clist):
        p_score = []
        for p in plist:
            print("=== Grid Search p={}, c={} ===".format(p, c))
            svr = SVRegressor(ker_type='-g', p=p, c=c, eps=0.1)
            score = cross_val_regression(X, y, svr, k=5, mth='R2')
            p_score.append(score)
        plt.plot(logplist, p_score, color=colors[idx], marker='o', label='c={}'.format(c))
        maxscore = max(maxscore, max(p_score))
    
    print("max R2 score: ", maxscore)
    
    plt.xlabel("parameter log(p)")
    plt.ylabel("R2 score")
    plt.legend()
    plt.show()
