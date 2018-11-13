#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
__author__ = 'YYF'
__mtime__ = '2018/11/13'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓   ┏┓
            ┏┛┻━━━┛┻┓
           ┃   ☃   ┃
           ┃ ┳┛ ┗┳ ┃
           ┃   ┻    ┃
            ┗━┓   ┏━┛
              ┃    ┗━━━┓
               ┃ 神兽保佑 ┣┓
               ┃ 永无BUG! ┏┛
                ┗┓┓┏ ━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
import tensorflow as tf
import numpy as np

# action = 1
# x = tf.one_hot(action, 2)
# print(x)
# #  将数据转换成one_hot形式
# x1 = tf.squeeze(x)
# print(x)
# with tf.Session() as sess:
#     print(sess.run(x))
#     print(sess.run(x1))

# 去除维度为1
# y = tf.multiply(x, self.pre_statu)
# # 将当前动作和当前状态进行对应相乘
# y = tf.reduce_sum(y, axis=1)
# # 将输出y进行相加
# y = tf.expand_dims(y, axis=1)
# # 将输出y进行扩维输出
x = np.random.randint(1, 2, size=(2))
print(x)
x2 = np.random.randint(0, 2, size=(5, 2))
print(x2)
x3 = np.multiply(x, x2)
print(x3)
x4 = np.sum(x3)
print(x4)