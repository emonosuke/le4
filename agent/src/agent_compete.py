from utils import load_sanfrancisco
from svr import SVRegressor
import numpy as np
import sys
from scaler import StandardScaler

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

# p = 9.5, C = 27000, eps = 0.09
svr = SVRegressor(ker_type='-g', p=9.5, c=27000, eps=0.09)

print('start SVR fit...')
svr.fit(X_train_std, y_train)

print('start SVR predict...')
y_pred = svr.predict(X_test_std)

for per in range(70, 121, 5):
    print('percentage of prediction: ', per)
    y_offer = (per / 100.0) * y_pred

    # 利益
    # リストの価格の30%は原価とする
    revenue = 0

    cnt = 0
    for i in range(hf):
        # print(y_offer[i], y_test[i])

        if y_offer[i] < 0:
            y_offer[i] = 0
        if y_offer[i] < y_test[i]:
            cnt += 1
            revenue += (y_offer[i] - y_test[i] * 0.3)

    print("revenue by SVR agent: ", revenue)
    print("the agent wins {} times".format(cnt))
    print("efficiency: ", revenue / cnt)


### 単純に価格の平均値を提示し続けるエージェントと比較する
y_simp = np.mean(y_train)
print("mean of y_train: ", y_simp)

rev_simp = 0

cnt = 0
for i in range(hf):
    if y_simp < y_test[i]:
        cnt += 1
        rev_simp += (y_simp - y_test[i] * 0.3)

print("revenue of mean price offering: ", rev_simp)
print("the agent wins {} times".format(cnt))
print("efficiency: ", rev_simp / cnt)