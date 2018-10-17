from utils import load_data, plot_decision_regions, cross_val_score
from svc import SVClassifier
from softsvc import SoftSVC
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

X = iris["data"][:, (2, 3)] # petal length, petal width
y = iris["target"]

# setosa or versicolor
X = X[(y == 0) | (y == 1)]
y = y[(y == 0) | (y == 1)]

print("X: ")
print(X)

def class_mapping(x):
    if x == 0:
        return -1
    else:
        return 1
y = list(map(class_mapping, y))
print("y: ")
print(y)

c = float(sys.argv[1])
kernel_type = sys.argv[2]
# default 値で初期化
p = 2.0
if kernel_type ==  'p' or kernel_type == 'g':
    p = float(sys.argv[4])

svm = SoftSVC(kernel_type, p, c)

svm.fit(X, y)
# cross_val_score(X, y, svm)

plot_decision_regions(X, y, svm, resolution=0.1)
