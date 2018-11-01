#!/usr/bin/env python
"""
Usage:
    ./dat_main.py input_file [param c] [param eps] (-n | -p | -g | -s) ([param kernel])
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
    print("./dat_main.py input_file [param c] [param eps] (-n | -p | -g) ([param kernel])")
    sys.exit(1)

if __name__ == '__main__':
    c = 1.0
    eps = 0.1
    p = 2.0

    if len(sys.argv) < 5:
        sysargv_mismatch()
    
    input_file = sys.argv[1]
    c = float(sys.argv[2])
    eps = float(sys.argv[3])
    kernel_type = sys.argv[4]
    if len(sys.argv) > 5:
        p = float(sys.argv[5])
    

    X, y = load_data(input_file)

    svr = SVRegressor(kernel_type, p, c, eps)

    svr.fit(X, y)

    preds = svr.predict(X)

    print("prediction and correct y")
    for pi, yi in zip(preds, y):
        print(pi, yi)
