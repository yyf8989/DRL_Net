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

a = tf.constant([0, 1, 1])
b = tf.constant([[3., 4.], [1., 2.], [5., 6.]])
c = tf.one_hot(a, 2)
d = c*b
e = tf.reduce_sum(d, axis=1)
f = tf.expand_dims(e, axis=1)

with tf.Session() as sess:
    print(sess.run(c))
    # 转换成one_hot
    print(sess.run(d))
    # 相乘查看对应数值相乘结果
    print(sess.run(e))
    # 进行d的axis=1的相加
    print(sess.run(f))
    # 再扩展维度

