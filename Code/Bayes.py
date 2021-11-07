# -*- coding: UTF-8 -*-
# @time: 2021/10/29
# ====================================================================================================

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
import pylab

mnist = load_digits()
x,test_x,y,test_y = train_test_split(mnist.data,mnist.target,test_size=0.25,random_state=40)

model = MultinomialNB()
model.fit(x,y)
z=model.predict(test_x)
print('准确率：',np.sum(z==test_y)/z.size)

# 学习后识别1011到1016六张图片并给出预测
print(model.predict(mnist.data[1011:1017]))

# 实际的1011到1016代表的数
mnist.target[1011:1017]

# 显示1011到1016数字图片
fig = pylab.gcf()
fig.canvas.set_window_title('Bayes')
plt.subplot(321)
plt.imshow(mnist.images[1011],cmap=plt.cm.gray_r,interpolation='nearest')
plt.subplot(322)
plt.imshow(mnist.images[1012],cmap=plt.cm.gray_r,interpolation='nearest')
plt.subplot(323)
plt.imshow(mnist.images[1013],cmap=plt.cm.gray_r,interpolation='nearest')
plt.subplot(324)
plt.imshow(mnist.images[1014],cmap=plt.cm.gray_r,interpolation='nearest')
plt.subplot(325)
plt.imshow(mnist.images[1015],cmap=plt.cm.gray_r,interpolation='nearest')
plt.subplot(326)
plt.imshow(mnist.images[1016],cmap=plt.cm.gray_r,interpolation='nearest')
plt.show()
