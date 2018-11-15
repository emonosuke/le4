from utils import cross_val_regression
from svr import SVRegressor
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from skopt import gp_minimize
import math
from utils import load_sanfrancisco

INF = 1e12
X_pro, y_pro = load_sanfrancisco(500)
param_p = []
param_C = []
scores = []
maxparam = {'score': -INF, 'p': 0, 'C': 0, 'eps': 0}
num = 100

def bayes_opt(x):
    p = x[0]
    C = x[1]
    eps = x[2]

    print('===== bayes_opt p={}, C={}, eps={} ====='.format(p, C, eps))

    svr = SVRegressor(ker_type='-g', p=p, c=C, eps=eps)
    score = cross_val_regression(X_pro, y_pro, svr, k=5, mth='R2')

    if score > maxparam['score']:
        maxparam['score'] = score
        maxparam['p'] = p
        maxparam['C'] = C
        maxparam['eps'] = eps
    
    param_p.append(p)
    param_C.append(C)
    scores.append(score)
    
    if math.isnan(score):
        return INF

    return -1 * score

spaces = [
    (2**-1, 2**5), # p
    (2**-5, 2**15), # C
    (2**-10, 2**-3)  # eps
]
res = gp_minimize(bayes_opt, spaces, acq_func="EI", n_calls=num)
print(scores)

# グラフ描画
fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('sigma')
ax.set_ylabel('C')
ax = ax.scatter(param_p, param_C, c=scores, cmap='Blues')
fig.colorbar(ax)
fig.savefig('./data/bayes_opt.png')

print("max score parameter: ", maxparam)