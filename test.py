# coding = utf-8

import os

path = os.path.curdir
path_1 = os.path.dirname(path)
path_2 = os.path.abspath(__file__)
print(path_1)
print(path_2)

import collections
s = " I love   China! "
ss = s.split()
sss = s.split(" ")
print(ss)
print(sss)
a = ["I", "love", "China", "China"]
b = collections.Counter(a).most_common()
print(b)

import numpy as np
data_list = [1, 3, 5, 7, 9]
data_ndarray = np.array(list(data_list))
print(data_ndarray)
print(type(data_ndarray))

for i in range(2):
    print("===")