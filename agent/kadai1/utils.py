import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def load_data(filepath):
    """
    filepath にある sample_linear.dat の形式のデータを X, y に読み込む
    """
    res = []
    f = open(filepath)
    line = f.readline()
    while line:
        nums = [float(i) for i in line.split(', ')]
        res.append(nums)
        line = f.readline()
    arr = np.array(res)
    return arr[:, :-1], arr[:, -1]


def plot_decision_regions(x, y, model, resolution=0.01):
    markers = ('o', 'x')
    cmap = ListedColormap(('red', 'blue'))

    ## 2変数の入力データの最小値から最大値まで引数resolutionの幅でメッシュを描く
    x1_min, x1_max = x[:, 0].min()-1, x[:, 0].max()+1
    x2_min, x2_max = x[:, 1].min()-1, x[:, 1].max()+1
    x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    ## メッシュデータ全部を学習モデルで分類
    z = model.predict(np.array([x1_mesh.ravel(), x2_mesh.ravel()]).T)
    z = z.reshape(x1_mesh.shape)

    ## メッシュデータと分離クラスを使って決定境界を描いている
    plt.contourf(x1_mesh, x2_mesh, z, alpha=0.4, cmap=cmap)
    plt.xlim(x1_mesh.min(), x1_mesh.max())
    plt.ylim(x2_mesh.min(), x2_mesh.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0],
                    y=x[y == cl, 1],
                    alpha=0.6,
                    c=cmap(idx),
                    edgecolors='black',
                    marker=markers[idx],
                    label=cl)
    
    plt.show()
