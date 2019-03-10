#coding:utf-8
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

# 由于e指数可能会很大，存在溢出可能
def simple_softmax(x):
    expa = np.exp(x)
    sum_expa = np.sum(expa)
    result = expa/sum_expa
    return result

# 改进方案，根据公式，加上/减去常数不影响最终结果
# softmax 输出值之和为1
def softmax(x):
    c = np.max(x)
    expa = np.exp(x - c)
    sum_expa = np.sum(expa)
    result = expa/sum_expa
    return result

# 均方误差
def mean_square_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# 交叉熵
def cross_entropy_error(y, t):
    # 防止y为0时值变为无穷
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

# 批处理的交叉熵计算
def batch_cross_entropy_error(y, t):
    # 防止y为0时值变为无穷
    delta = 1e-7
    if t.ndim == 1:
        # if y = np.array([1,2,3]) , change to y = np.array([[1,2,3]), 便于统一计算
        y = y.reshape(1, y.size)
        t = t.reshape(1,2 t.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size

if __name__ == "__main__":
    # test softmax
    #x = np.array([1,2])
    #result = softmax(x)
    #print (result)

    # test mean_square_error
    y = np.array([0.1, 0.6, 0.1, 0.2])
    t = np.array([0, 1, 0, 0])
    mse = mean_square_error(y, t)
    print (mse)
    # test cross_entropy_error
    cee = cross_entropy_error(y, t)
    print (cee)

