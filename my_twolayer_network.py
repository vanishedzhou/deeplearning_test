#coding:utf-8
"""
手写数字识别，使用两层网络结构
"""
import os,sys
sys.path.append("./src")
import numpy as np
from src.ch04.two_layer_net import *
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

x = np.random.rand(100, 784)
t = np.random.rand(100, 10)
#print(x)
#print(t)
network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

#y = net.predict(x) 
##print(y)
#grads = net.gradient(x, t)
## 4
#print(len(grads))
## 784 * 100
#print(grads['W1'].shape)
## 100 * 10
#print(grads['W2'].shape)

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
#print(x_train.shape)
#print(t_train.shape)
#print(x_test.shape)
#print(t_test.shape)

iters_num = 20000  # 适当设定循环的次数
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # calc gradient
    grad = network.gradient(x_batch, t_batch)

    # update params
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print ("train_acc:%s, test_acc:%s" % (train_acc, test_acc))

# 绘制图形
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
