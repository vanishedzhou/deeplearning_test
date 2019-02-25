#coding:utf-8
import os,sys
import numpy as np

# array
arr = np.array([1,2,3])
print(arr)

# multi dimension array
arr = np.array([[1,2], [3,4]])
print (arr)
print (arr.shape)
print (arr.dtype)

# 数组乘法是相同位置元素相乘结果
# 元素形状不同也可乘，利用“广播”特性，即 brr变为 [10,20],[10,20]
arr = np.array([[1,2],[3,4]])
brr = np.array([10,20])
r = arr * brr
print (r)

# 压扁成一维数组
arr = arr.flatten()
print (arr)
# get index
print (arr[np.array([0,3])])
print (arr > 3)
print (arr[arr > 3])

# simple calc func
x = np.array([0.0,1.0])
w = np.array([0.5,0.5])
b = -0.7
print (np.sum(x*w)+b)

# array multiply
a = np.array([[1,2],[3,4]])
b = np.array([[10,20],[30,40]])
mul = np.dot(a, b)
print (mul)
