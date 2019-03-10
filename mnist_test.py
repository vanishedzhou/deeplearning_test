#coding:utf-8
import sys,os
from dataset.mnist import load_mnist
from PIL import Image
import numpy as np

# show img by pixel data
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def test_show_img():
    # x is img data, t is label data
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    # 60000, 784
    print (x_train.shape)
    # 60000
    print (t_train.shape)
    print (x_test.shape)
    print (t_test.shape)
    
    # show the pic image
    img = x_train[2]
    label = t_train[0]
    print (label)
    print (img.shape)
    # reshape to 28 x 28 dimension array
    img = img.reshape(28,28)
    print (img.shape)
    img_show(img)

# 批处理
def test_batch_handle():
    (x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True, normalize=True)
    # 60000 * 784
    print (x_train.shape)
    # 60000 * 10, 值使用了one-hot表示
    print (t_train.shape)

    train_size = x_train.shape[0]
    # 60000
    print (train_size)
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    print (x_batch)

if __name__ == "__main__":
    test_batch_handle()
