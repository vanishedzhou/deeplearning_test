#coding:utf-8
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

"""
network flow
    F1 
x1  F2  S1  y1
x2  F3  S2  y2
"""
def init_network():
    network = {}
    # w1表示第一层隐层对应x1 x2的权重
    network['W1'] = np.array([[0.1, 0.2, 0.3], [0.1, 0.1, 0.1]])
    # b1表示第一层隐层各神经元的偏置值
    network['b1'] = np.array([0.5, 0.4, 0.5])
    # w2表示第二层隐层对应F1 F2 F3的权重
    network['W2'] = np.array([[0.1, 0.2], [0.1, 0.1], [0.2, 0.2]])
    # b2表示第二层隐层各神经元的偏置值
    network['b2'] = np.array([0.5, 0.5])
    # w2表示输出层对应S1 S2的权重
    network['W3'] = np.array([[0.1, 0.2], [0.1, 0.1]])
    # b2表示输出层各神经元的偏置值
    network['b3'] = np.array([0.4, 0.4])
    return network

def forward(network, x):
    W1, W2, W3 = [network['W1'], network['W2'], network['W3']]
    b1, b2, b3 = [network['b1'], network['b2'], network['b3']]
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    # y的激活函数此处直接为相等
    y = a3
    return y

def network_use():
    x = np.array([1,2])
    network = init_network()
    y = forward(network, x)
    print (y)

if __name__ == "__main__":
    network_use()
