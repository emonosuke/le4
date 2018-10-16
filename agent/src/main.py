#!/usr/bin/env python
"""
Usage:
    ./main.py input_file ( n | p | g | s ) [params]
        n: kernel trick なし [paramsなし]
        p: Polynomial kernel [params: p]
        g: Gauss kernel [params: p]
        s: Sigmoid kernel [params: p q]
"""
from utils import load_data, plot_decision_regions
from svc import SVClassifier
import sys
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[2] not in ['n', 'p', 'g', 's']:
        print("ERROR: command-line args must be following args")
        print("./main.py input_file ( n | p | g | s ) [params]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    kernel_type = sys.argv[2]
    # default 値で初期化
    p = 2.0
    q = 2.0
    if kernel_type ==  'p' or kernel_type == 'g':
        p = float(sys.argv[3])
    elif kernel_type == 's':
        p = float(sys.argv[3])
        q = float(sys.argv[4])
    
    X, y = load_data(input_file)

    svm = SVClassifier(kernel_type, p, q)

    svm.fit(X, y)

    plot_decision_regions(X, y, svm, resolution=0.1)