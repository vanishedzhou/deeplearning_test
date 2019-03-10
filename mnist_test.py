#coding:utf-8
import sys,os
from dataset.mnist import load_mnist
from PIL import Image
import numpy as np

# show img by pixel data
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# x is img data, t is label data
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
print (x_train.shape)
print (t_train.shape)
print (x_test.shape)
print (t_test.shape)

img = x_train[2]
label = t_train[0]
print (label)
print (img.shape)
# reshape to 28 x 28 dimension array
img = img.reshape(28,28)
print (img.shape)

img_show(img)
