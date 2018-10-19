#!/usr/bin/env python

from utils import load_data, plot_decision_regions, cross_val_score
from svc import HardSVC, SoftSVC
from scaler import MinMaxScaler
import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    input_file = sys.argv[1]
    X, y = load_data(input_file)
    kernel_type = '-g'

    pmin = -2
    pmax = 8
    p_list = [pow(2, i) for i in range(pmin, pmax+1, 1)]
    cmin = -3
    cmax = 3
    c_list = [pow(2, i) for i in range(cmin, cmax+1, 1)]
    logp_list = [i for i in range(pmin, pmax+1, 1)]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    max_acc = 0
    for idx, c in enumerate(c_list):
        p_acc = []
        for p in p_list:
            svm = SoftSVC(kernel_type, p, c)
            # svm = HardSVC(kernel_type, p)
            acc = cross_val_score(X, y, svm)
            p_acc.append(acc)
        max_acc = max(max_acc, max(p_acc))
        plt.plot(logp_list, p_acc, color=colors[idx], marker='o', label='c={}'.format(c))
    
    plt.xlabel("log_2(p)")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
    print("max accuracy: ", max_acc)