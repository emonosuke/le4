import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def load_data(filepath):
    """
    @brief filepath にある sample_linear.dat 同様の形式のデータを X, y に読み込む
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
