#coding:utf-8
import PyQt5
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import math

x = np.arange(0, 6, 0.1)
y = np.sin(x)
y2 = np.cos(x)

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

x = np.arange(-5,5,0.1)
y = np.maximum(0, x)
plt.plot(x, y)
plt.show()

