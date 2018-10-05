#!/usr/bin/env python
'''
Usage:
    ### TODO: kernel のパラメータを与えられるようにする
    ./main.py input_file ( n | p | g | s )
        n: kernel trick なし
        p: Polynomial kernel(p = 2)
        g: Gauss kernel(p = 10)
        s: Sigmoid kernel(p1, p2)
'''
from utils import load_data, plot_decision_regions
from svclassifier import SVClassifier
import sys
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("./main.py input_file ( n | p | g | s )")
        sys.exit(1)
    
    input_file = sys.argv[1]
    kernel_type = sys.argv[2]

    X, y = load_data(input_file)

    svm = SVClassifier()

    svm.fit(X, y, kernel_type)

    # test = np.array([[1]])
    # pred = svm.predict(test)

    plot_decision_regions(X, y, svm)

    print(pred)
