# -*- coding: UTF-8 -*-
# @time: 2021/10/29
# ====================================================================================================

import numpy as np
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import _pickle as pickle
import matplotlib.pyplot as plt
import pylab

# 加载数据集
mnist = load_digits()
x,test_x,y,test_y = train_test_split(mnist.data,mnist.target,test_size=0.25,random_state=40)

# 选择分类器
model = svm.LinearSVC()
model.fit(x,y)

z=model.predict(test_x)
print('准确率：',np.sum(z==test_y)/z.size)

with open('../Dataset\\model.pkl','wb') as file:
        pickle.dump(model,file)

# 学习后识别520到525六张图片并给出预测
print(model.predict(mnist.data[500:506]))

# 实际的520到525代表的数
mnist.target[500:506]

# 显示520到525数字图片
fig = pylab.gcf()
fig.canvas.set_window_title('SVM')
plt.subplot(321)
plt.imshow(mnist.images[500],cmap=plt.cm.gray_r,interpolation='nearest')
plt.subplot(322)
plt.imshow(mnist.images[501],cmap=plt.cm.gray_r,interpolation='nearest')
plt.subplot(323)
plt.imshow(mnist.images[502],cmap=plt.cm.gray_r,interpolation='nearest')
plt.subplot(324)
plt.imshow(mnist.images[503],cmap=plt.cm.gray_r,interpolation='nearest')
plt.subplot(325)
plt.imshow(mnist.images[504],cmap=plt.cm.gray_r,interpolation='nearest')
plt.subplot(326)
plt.imshow(mnist.images[505],cmap=plt.cm.gray_r,interpolation='nearest')
plt.show()