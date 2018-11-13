#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
__author__ = 'YYF'
__mtime__ = '2018/11/12'
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
import numpy as np

# 定义reward关系
rewards = [
    [-1, -1, -1, -1, -0, -1],
    [-1, -1, -1, -0, -1, 100],
    [-1, -1, -1, -0, -1, -1],
    [-1, -0, -0, -1, -0, -1],
    [-0, -1, -1, -0, -1, 100],
    [-1, -0, -1, -1, -0, 100]
]
# 定义折扣率
discount = 0.8


class Walk_out:
    def __init__(self, discount, rewards):
        self.discount = discount
        self.rewards = rewards

    def create_value(self):
        # 先生成价值0矩阵
        values = np.zeros([6, 6])
        # print(values)

        # values (-> action_status)与reward的关系式为:
        #  values(i) = reward + discount * max(value of action_status)

        # 第一种方法，随机生成训练
        for _ in range(10000):
            # 随机生成状态行
            statu = np.random.randint(0, 6)
            # print(statu)
            # 随机生成动作列
            action = np.random.randint(0, 6)
            # print(action)
            # 找到对应动作状态行的最大值
            get_max = max(values[action])
            # print(get_max)
            # 找到reward的对应值
            get_reward = self.rewards[statu][action]
            # print(get_reward)
            # 初始化get_reward为-1--> 0，保证价值为0,其余保持为原计算数据
            if get_reward == -1:
                values[statu][action] = 0
            else:
                values[statu][action] = get_reward + self.discount * get_max

        # 第一种方法，连续生成训练
        # for i in range(1000):
        #     for statu in range(6):
        #         for action in range(6):
        #             get_max = max(values[action])
        #             # print(get_max)
        #             get_reward = self.rewards[statu][action]
        #             # print(get_reward)
        #             if get_reward == -1:
        #                 values[statu][action] = 0
        #             else:
        #                 values[statu][action] = get_reward + self.discount * get_max

        values = values / 5
        # for i in range(6):
        #     for j in range(6):
        #         values[i][j] = int(values[i][j])

        # print(values)
        return values

    def go_max_values(self, values_status, start_statu, end_statu):
        Q_value = []
        actions = []
        start = start_statu
        end = end_statu
        actions.append(start)
        values = values_status
        while start != end:
            # for i in values[start]:
            #     if values[start][i] != 0:
            q_value = max(values[start])
            Q_value.append(q_value)
            start = np.argmax(values[start])
            actions.append(start)
        # print(Q_value, actions)
        return Q_value, actions


if __name__ == '__main__':
    walk = Walk_out(discount, rewards)
    values = walk.create_value()
    print('得到的q价值为:\n', values)
    print('----------------------------------------------------')
    # print('请输入起点(0-5):', end='')
    # start_palce = int(input())
    # print('请输入终点(起终点不能为同一个)(0-5):', end='')
    # end_palce = int(input())
    q_values, Actions = walk.go_max_values(values, 2, 5)
    ttl_values = np.sum(q_values)
    print('总价值为：{}'.format(ttl_values))
    print('动作路径:', end='')
    for node in Actions:
        if node < len(Actions):
            print('{} -->'.format(node), end='')
        else:
            print(node)
    print('----------------------------------------------------')