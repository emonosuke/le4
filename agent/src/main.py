#!/usr/bin/env python
"""
Usage:
    ./main.py input_file ( n | p | g | s ) [params]
        n: kernel trick なし [paramsなし]
        p: Polynomial kernel [params: p]
        g: Gauss kernel [params: p]
        s: Sigmoid kernel [params: p q]
"""
from utils import load_data, plot_decision_regions, cross_val_score
from svc import SVClassifier
from softsvc import SoftSVC
from scaler import MinMaxScaler
import sys
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 4 or sys.argv[3] not in ['n', 'p', 'g']:
        print("ERROR: command-line args must be following args")
        print("./main.py input_file [cvalue] ( n | p | g ) [param]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    c = float(sys.argv[2])
    kernel_type = sys.argv[3]
    # default 値で初期化
    p = 2.0
    if kernel_type ==  'p' or kernel_type == 'g':
        p = float(sys.argv[4])
    
    X, y = load_data(input_file)

    svm = SoftSVC(kernel_type, p, c)

    svm.fit(X, y)
    # cross_val_score(X, y, svm)

    plot_decision_regions(X, y, svm, resolution=0.1)