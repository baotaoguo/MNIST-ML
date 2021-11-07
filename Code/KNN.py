# -*- coding: UTF-8 -*-
# @time: 2021/10/29
# ====================================================================================================

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pylab

mnist = load_digits()
x,test_x,y,test_y = train_test_split(mnist.data,mnist.target,test_size=0.25,random_state=40)

model = KNeighborsClassifier(n_neighbors=4)
model.fit(x,y)
z=model.predict(test_x)
print('准确率：',np.sum(z==test_y)/z.size)

# 学习后识别1660到1666六张图片并给出预测
print(model.predict(mnist.data[1650:1656]))

# 实际的1660到1666代表的数
mnist.target[1650:1656]

# 显示1660到1666数字图片
fig = pylab.gcf()
fig.canvas.set_window_title('KNN')
plt.subplot(321)
plt.imshow(mnist.images[1650],cmap=plt.cm.gray_r,interpolation='nearest')
plt.subplot(322)
plt.imshow(mnist.images[1651],cmap=plt.cm.gray_r,interpolation='nearest')
plt.subplot(323)
plt.imshow(mnist.images[1652],cmap=plt.cm.gray_r,interpolation='nearest')
plt.subplot(324)
plt.imshow(mnist.images[1653],cmap=plt.cm.gray_r,interpolation='nearest')
plt.subplot(325)
plt.imshow(mnist.images[1654],cmap=plt.cm.gray_r,interpolation='nearest')
plt.subplot(326)
plt.imshow(mnist.images[1655],cmap=plt.cm.gray_r,interpolation='nearest')
plt.show()


