#!/usr/bin/env python
"""
Usage:
    ./svrmain.py input_file [param c] [param eps] (-n | -p | -g | -s) ([param kernel])
        -n: kernel trick なし
        -p: Polynomial kernel [param]
        -g: Gauss kernel [param]
"""
from utils import load_data
from svr import SVRegressor
import sys
import numpy as np

def sysargv_mismatch():
    print("ERROR: command-line args must be following args")
    print("./main.py input_file [param c] [param eps] (-n | -p | -g) ([param kernel])")
    sys.exit(1)

if __name__ == '__main__':
    # TODO: argparse 導入
    c = 1.0
    eps = 0.1
    p = 2.0

    if len(sys.argv) < 5:
        sysargv_mismatch()
    
    # ./main3.py data/sample10.dat 1000.0 0.1 -n
    # ./main3.py data/sample10.dat 1000.0 0.1 -g 2.236
    input_file = sys.argv[1]
    c = float(sys.argv[2])
    eps = float(sys.argv[3])
    kernel_type = sys.argv[4]
    if len(sys.argv) > 5:
        p = float(sys.argv[5])
    

    X, y = load_data(input_file)

    svr = SVRegressor(kernel_type, p)

    svr.fit(X, y, c, eps)

    """
    preds = svr.predict(X)

    for pi, yi in zip(preds, y):
        print(pi, yi)

    # cross validation
    # 交差検証の前にX, yをシャッフルする
    random_mask = np.arange(len(y))
    np.random.shuffle(random_mask)
    X = X[random_mask]
    y = y[random_mask]

    k = 5
    part = len(y) // k
    scores = []
    for i in range(k):
        test_X = X[part * i : part * (i+1)]
        test_y = y[part * i : part * (i+1)]
        train_X = np.concatenate((X[0:part*i], X[part*(i+1):-1]), axis=0)
        train_y = np.concatenate((y[0:part*i], y[part*(i+1):-1]), axis=0)

        svr.fit(train_X, train_y, c, eps)
        score = svr.score(test_X, test_y, mth='MSE')
        print("MSE score: ", score)
        scores.append(score)
    res = np.mean(np.array(scores))
    print("{}-fold average score: ".format(k), res)
    """