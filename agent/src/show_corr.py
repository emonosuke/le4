from utils import cross_val_regression
from svr import SVRegressor
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from skopt import gp_minimize
import math
import seaborn as sns

def load_sanfrancisco(sample_size=0):
    """
    San Francisco-listings.csv を読み込み、前処理した結果を返す

    Parameters
    ----------
    sample_size(int): データセットからランダムにサンプリングする個数

    Returns
    ----------
    X(numpy.array(sample_size, 11)): 説明変数 
    y(numpy.array(sample_size, 1)): 目的変数
    """

    def dollartofloat(s):
        """
        '$1,300.00' 等の価格を表す str を float へ変換
        """
        s = s.replace('$', '')
        s = s.replace(',', '')
        return float(s)

    df = pd.read_csv('./data/San Francisco-listings.csv')

    # prices を float へ変換
    df['price'] = df['price'].map(dollartofloat)

    # prices > 500 のデータを削除
    df.drop(df.index[df.price > 500], inplace=True)

    # 特徴量を絞る
    df = df[['latitude', 'longitude', 'accommodates', 'number_of_reviews', 'review_scores_rating', 'price']]
    print("df size: ", len(df))

    # sample_size = 0 または sample_size が行数を超えている場合には、全て利用
    if sample_size == 0 or sample_size > len(df):
        print("load sanfrancisco: read {} data(all)".format(len(df)))
    else:
        print("load sanfrancisco: read {} / {} data".format(sample_size, len(df)))
        df = df.sample(n=sample_size)

    # y を作成
    y = np.array(df['price'].values)
    y = np.array([[yi] for yi in y])

    # 欠損値(review_scores_rating)を 0 へ変換
    df = df.fillna(0)

    # 数値データはそのまま X へ
    df_processed = df[['latitude', 'longitude', 'accommodates', 'number_of_reviews', 'review_scores_rating']]
    X = df_processed.values

    X = X.astype(np.float32)

    X = np.hstack((X, y))

    # 確認
    print(type(X))
    print(X)
    print("X: ", X.shape)

    return X

X = load_sanfrancisco()

corr_mat = np.corrcoef(X.transpose())

feature_names = ['lat', 'lon', 'accom', 'reviews', 'rating', 'price']

sns.heatmap(corr_mat, annot=True, xticklabels=feature_names, yticklabels=feature_names)

plt.show()
