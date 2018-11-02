import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from scaler import MinMaxScaler, StandardScaler
from extra_data import load_data_wine, load_data_iris
import math

def load_data(filepath):
    """
    @brief filepath にある sample_linear.dat 同様の形式のデータを X, y に読み込む
    """
    if filepath == 'WINE':
        return load_data_wine()
    if filepath == 'IRIS':
        return load_data_iris()
    
    res = []
    f = open(filepath)
    line = f.readline()
    while line:
        nums = [float(i) for i in line.split(', ')]
        res.append(nums)
        line = f.readline()
    arr = np.array(res)
    return arr[:, :-1], arr[:, -1]


def plot_decision_regions(x, y, model, resolution=0.1):
    """
    @brief
    x を2次元平面に plot する
    plot された点の色は y によって決まり、領域の色は model の予測結果により決まる
    @param[in] x, y, model, resolution resolution によりグリッドの幅を指定する
    @return なし
    """
    # x が2次元でない場合は終了
    if x.shape[1] != 2:
        print("plot_decision_regions: X must be 2-dimensional")
        return
    
    cmap = ListedColormap(('blue', 'red'))

    x1_min, x1_max = x[:, 0].min()-1, x[:, 0].max()+1
    x2_min, x2_max = x[:, 1].min()-1, x[:, 1].max()+1
    x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    # 各特徴量を1次元配列に変換してモデルにより Class を予測する
    z = model.predict(np.array([x1_mesh.ravel(), x2_mesh.ravel()]).T)
    z = z.reshape(x1_mesh.shape)

    # グリッドに対して等高線を plot
    plt.contourf(x1_mesh, x2_mesh, z, alpha=0.4, cmap=cmap)
    # 軸の範囲を指定
    plt.xlim(x1_mesh.min(), x1_mesh.max())
    plt.ylim(x2_mesh.min(), x2_mesh.max())

    # Class ごとに plot
    # Class=1
    plt.scatter(x=x[y > 0, 0],
                y=x[y > 0, 1],
                alpha=None,
                c='red',
                marker='o',
                label=1)
    # Class=-1
    plt.scatter(x=x[y <= 0, 0],
                y=x[y <= 0, 1],
                alpha=0.6,
                c='blue',
                marker='x',
                label=-1)
    
    plt.legend(loc='upper left')
    plt.show()

def cross_val_score(X, y, model, k=5):
    """
    k-fold の交差検証を行う
    """
    # 交差検証の前にX, yをシャッフルする
    random_mask = np.arange(len(y))
    np.random.shuffle(random_mask)
    X = X[random_mask]
    y = y[random_mask]

    part = len(y) // k
    accs = []
    for i in range(k):
        test_X = X[part * i : part * (i+1)]
        test_y = y[part * i : part * (i+1)]
        train_X = np.concatenate((X[0:part*i], X[part*(i+1):-1]), axis=0)
        train_y = np.concatenate((y[0:part*i], y[part*(i+1):-1]), axis=0)

        # 特徴量のスケーリング(正規化)
        sc = MinMaxScaler()
        train_X_std = sc.fit_transform(train_X)
        model.fit(train_X_std, train_y)
        test_X_std = sc.transform(test_X)
        pred = model.predict(test_X_std)
        n_correct = sum(pred == test_y)
        acc = n_correct / len(pred)
        print("fold {}/{} accuracy: ".format(i+1, k), acc)
        accs.append(acc)
    res = np.mean(np.array(accs))
    print("{}-fold average accuracy: ".format(k), res)
    return res

def cross_val_regression(X, y, svr, k=5, mth='MSE'):
    """
    SVR(回帰)について交差検証を行う
    """
    # 交差検証の前にX, yをシャッフルする
    random_mask = np.arange(len(y))
    np.random.shuffle(random_mask)
    X = X[random_mask]
    y = y[random_mask]

    k = 5
    part = len(y) // k
    scores = []
    for i in range(k):
        print("{} th cross validation...".format(i + 1))
        test_X = X[part * i : part * (i+1)]
        test_y = y[part * i : part * (i+1)]
        train_X = np.concatenate((X[0:part*i], X[part*(i+1):-1]), axis=0)
        train_y = np.concatenate((y[0:part*i], y[part*(i+1):-1]), axis=0)

        sc = StandardScaler()
        train_X_std = sc.fit_transform(train_X)
        test_X_std = sc.transform(test_X)

        svr.fit(train_X_std, train_y)
        score = svr.score(test_X_std, test_y, mth=mth)

        if math.isnan(score):
            print("nan detected: cross_val_regression terminated!!")
            return np.nan
        
        print("{} score: ".format(mth), score)
        scores.append(score)
    
    res = np.mean(np.array(scores))
    print("{}-fold average score: ".format(k), res)
    return res
