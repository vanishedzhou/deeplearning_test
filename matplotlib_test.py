#coding:utf-8
import PyQt5
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import math
import sys

dz_rate = 0.579
if len(sys.argv) > 1:
    dz_rate = float(sys.argv[1])
x = np.arange(0, 1, 0.001)
y = dz_rate - (dz_rate - x) / (1 - x)
print("dz_rate:" + str(dz_rate))
plt.axis([0,0.1,0,0.08])
plt.plot(x, y)
plt.show()

#plt.figure()
#plt.plot(x,y,label="sin")
#plt.plot(x,y2,linestyle = "--", label="cos")
#plt.xlabel("x") # x轴标签
#plt.ylabel("y") # y轴标签
#plt.title('sin & cos') # 标题
#plt.legend()
#plt.show()

# load local pic
#img = imread("/Users/zhouzhiyong03/Pictures/bd_logo1.png")
#plt.imshow(img)
#plt.show()

# normal function
#x = np.arange(-5,5,0.1)
##x = np.array([1,2,3])
#y = 1/(1+np.exp(-x))
##print (y)
#plt.plot(x, y, label="func")
#plt.show()

# 阶跃函数
#def func(x):
#    return np.array(x > 0, dtype=np.int)
#x = np.arange(-5,5,0.1)
#y = func(x)
#print (func(x))
#plt.plot(x, y)
## 指定y轴范围
#plt.ylim(-0.1, 1.1)
#plt.show()

# rectified linear unit , ReLU
#x = np.arange(-5,5,0.1)
#y = np.maximum(0, x)
#plt.plot(x, y)
#plt.show()

# 导数
#x = np.arange(0,20,0.1)
#y = 0.01*x**2 + 0.1*x
#y = np.sum(x**2)
#plt.plot(x, y)
#plt.show()
