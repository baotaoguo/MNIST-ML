# -*- coding: UTF-8 -*-
# @time: 2021/10/29
# ====================================================================================================

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pylab

mnist = load_digits()
x,test_x,y,test_y = train_test_split(mnist.data,mnist.target,test_size=0.25,random_state=40)

model = DecisionTreeClassifier(criterion="entropy")
model.fit(x,y)
z=model.predict(test_x)
print('准确率：',np.sum(z==test_y)/z.size)

# 学习后识别99到105六张图片并给出预测
print(model.predict(mnist.data[109:115]))


# 实际的99到105代表的数
var = mnist.target[109:115]

# 显示99到105数字图片
fig = pylab.gcf()
fig.canvas.set_window_title('decision tree')
plt.subplot(321)
plt.imshow(mnist.images[109],cmap=plt.cm.gray_r,interpolation='nearest')
plt.subplot(322)
plt.imshow(mnist.images[110],cmap=plt.cm.gray_r,interpolation='nearest')
plt.subplot(323)
plt.imshow(mnist.images[111],cmap=plt.cm.gray_r,interpolation='nearest')
plt.subplot(324)
plt.imshow(mnist.images[112],cmap=plt.cm.gray_r,interpolation='nearest')
plt.subplot(325)
plt.imshow(mnist.images[113],cmap=plt.cm.gray_r,interpolation='nearest')
plt.subplot(326)
plt.imshow(mnist.images[114],cmap=plt.cm.gray_r,interpolation='nearest')
plt.show()

import pandas as pd
x = pd.DataFrame(x)
with open("../Dataset\JueCetree.dot", 'w') as f:
     f = export_graphviz(model, feature_names = x.columns, out_file = f)
