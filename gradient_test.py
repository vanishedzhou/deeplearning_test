# coding: utf-8
import sys, os
import numpy as np
import pickle
from dataset.mnist import load_mnist
from base_func import *

def f(x):
    return x[0]**2 + x[1]**2

"""
梯度下降求函数极小值（最小值）
"""
def gradient_descent():
    x = np.array([-3.0, -4.0])
    step_num = 100
    # learning rate 学习率, 也称为超参数,通常由人工设置，需要尝试多种值以便是学习更加顺利。过大容易使结果发散，过小还没学习多少即结束更新
    lr = 0.1
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        print (x)

if __name__ == "__main__":
    gradient_descent()
