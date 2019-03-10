#coding:gbk
import tkinter
import os,sys
import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

X = np.random.normal(size=(12, 2))
plt.scatter(X[:, 0], X[:, 1])
plt.plot(X[:, 0])
# create a new figure
plt.figure()
plt.plot(X[:, 0])
plt.show()
