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

if __name__ == "__main__":
    x = np.array([1,2])
    result = softmax(x)
    print (result)

