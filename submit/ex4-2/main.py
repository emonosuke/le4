#!/usr/bin/env python
"""
Usage:
    ./main.py input_file (-h | -s) ([param c]) (-n | -p | -g | -s) ([param kernel])
        -h: ハードマージンSVM
        -s: ソフトマージンSVM[param]

        -n: kernel trick なし
        -p: Polynomial kernel [param]
        -g: Gauss kernel [param]
"""
from utils import load_data, plot_decision_regions, cross_val_score
from svc import HardSVC, SoftSVC
from scaler import MinMaxScaler
import sys
import numpy as np

def sysargv_mismatch():
    print("ERROR: command-line args must be following args")
    print("./main.py input_file (-h | -s) ([param c]) (-n | -p | -g) [param kernel]")
    sys.exit(1)

if __name__ == '__main__':
    # 以下で sys.argv より、input_file, isHard, c, kernel_type, p を決定する 
    # default 値は以下の通り
    c = 1.0
    p = 2.0

    if len(sys.argv) < 4:
        sysargv_mismatch()
    
    input_file = sys.argv[1]

    if sys.argv[2] not in ['-h', '-s']:
        sysargv_mismatch()
    
    isHard = (sys.argv[2] == '-h')

    if isHard:
        if sys.argv[3] not in ['-n', '-p', '-g']:
            sysargv_mismatch()
        else:
            kernel_type = sys.argv[3]
            if kernel_type in ['-p', '-g']:
                p = float(sys.argv[4])
    else:
        # sys.argv[2] == '-s'
        c = float(sys.argv[3])
        kernel_type = sys.argv[4]

        if sys.argv[4] not in ['-n', '-p', '-g']:
            sysargv_mismatch()
        else:
            kernel_type = sys.argv[4]
            if kernel_type in ['-p', '-g']:
                p = float(sys.argv[5])
    
    X, y = load_data(input_file)

    if isHard:
        svm = HardSVC(kernel_type, p)
    else: 
        svm = SoftSVC(kernel_type, p, c)

    # sc = MinMaxScaler()
    # X = sc.fit_transform(X)
    # svm.fit(X, y)
    cross_val_score(X, y, svm)

    # plot_decision_regions(X, y, svm, resolution=0.1)