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
import tensorflow as tf
import gym
import numpy as np
import time


# QNet进行全连接输出
class QNet:
    def __init__(self):
        self.w1 = tf.Variable(tf.truncated_normal([4, 30], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([30]))

        self.w2 = tf.Variable(tf.truncated_normal([30, 30], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([30]))

        self.w3 = tf.Variable(tf.truncated_normal([30, 2], stddev=0.1))
        self.b3 = tf.Variable(tf.zeros([2]))

    def forward(self, observation):
        y = tf.nn.relu(tf.matmul(observation, self.w1) + self.b1)
        y = tf.nn.relu(tf.matmul(y, self.w2) + self.b2)
        y = tf.matmul(y, self.w3) + self.b3

        return y


# 复制QNet的结构进行训练
class TargetQNet:
    def __init__(self):
        self.w1 = tf.Variable(tf.truncated_normal([4, 30], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([30]))

        self.w2 = tf.Variable(tf.truncated_normal([30, 30], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([30]))

        self.w3 = tf.Variable(tf.truncated_normal([30, 2], stddev=0.1))
        self.b3 = tf.Variable(tf.zeros([2]))

    def forward(self, observation):
        y = tf.nn.relu(tf.matmul(observation, self.w1) + self.b1)
        y = tf.nn.relu(tf.matmul(y, self.w2) + self.b2)
        y = tf.matmul(y, self.w3) + self.b3

        return y


class Sample_game:
    def __init__(self):
        # 创建环境
        self.env = gym.make('CartPole-v0')
        # 定义用于训练的经验池
        self.experience_pool = []
        # 定义观察的状态
        self.observation = self.env.reset()

        # 创建经验池，要收集 St, Reward, action, St+1
        for i in range(10000):
            # 对动作采样,动作为1或者0
            action = self.env.action_space.sample()
            # 使用openai内嵌的功能模块取出对应的值 下一个采样，回报值为1， done=False和信息值
            next_observation, reward, done, info = self.env.step(action)
            # 将数据保存在经验池中，保存的为S(t), Reward, Action, S(t+1), Done(True/False)
            self.experience_pool.append([self.observation, reward, action, next_observation, done])
            # 依照工作状态进行状态的更新
            if done:
                self.observation = self.env.reset()
            else:
                self.observation = next_observation

    # 随机从经验池中选取对应的经验batch
    def get_sample(self, batch_size):
        experiences = []
        idxs = []
        for _ in range(batch_size):
            # 定位取出经验的序号索引，以便于后续进行更新
            # 随机取出对应索引的经验池
            idx = np.random.randint(0, len(self.experience_pool))
            # 保存索引以便于更新
            idxs.append(idx)
            # 保存在经验池
            experiences.append(self.experience_pool[idx])
        return idxs, experiences

    # 定义重置整体
    def reset(self):
        return self.env.reset()

    # 定义显示用函数
    def render(self):
        self.env.render()


class DRLNet:
    def __init__(self):
        # 定义采样情况，因为此有四个状态，故状态维度为4
        self.observation = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='Observation')
        # 定义动作的私有变量，动作只能为1个
        self.action = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='Action')
        # 定义回报的私有变量，回报只能为1个
        self.reward = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='Reward')
        # 同理，定义下一步采样情况，因为此有四个状态，故状态维度为4
        self.next_observation = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='Next_Observation')
        # 定义完成状态显示
        self.done = tf.placeholder(dtype=tf.bool, shape=[None], name='Done')

        # 实例化QNet
        self.qNet = QNet()
        # 实例化目标网络
        self.targetQNet = TargetQNet()

    def forward(self, discount):
        # 根据采样输入QNet，然后输出状态
        self.pre_statu = self.qNet.forward(self.observation)
        # 选择当前状态对应的Q值
        # 数据处理过程
        # 1, x = tf.one_hot(action,2), 将数据转换成one_hot形式
        # 2.x = tf.squeeze(x)，去除维度为1,形状为(1,2)
        # 3. y = tf.multiply(x, self.pre_statu), 将当前动作和当前状态进行对应相乘
        # 4. y = tf.reduce_sum(y, axis=1) 将输出y进行相加,行相加
        # 5. y = tf.expand_dims(y, axis=1) 将输出y进行扩维输出
        self.pre_q = tf.expand_dims(
            tf.reduce_sum(tf.multiply(tf.squeeze(tf.one_hot(self.action, 2), axis=1), self.pre_statu), axis=1), axis=1)

        # 根据下一个状态输出下一个Q t+1的状态
        self.next_statu = self.targetQNet.forward(self.next_observation)
        # 同理求出对应状态的Qt+1的值
        self.next_q = tf.expand_dims(tf.reduce_max(self.next_statu, axis=1), axis=1)

        # 得到目标Q值，加上判定条件，如果是最后一步，只用奖励，否则奖励Q(t) = r(t) + dis * maxQ(t+1)
        self.target_q = tf.where(self.done, self.reward, self.reward + discount * self.next_q)

    def play(self):
        # 输出当前状态
        self.statu = self.qNet.forward(self.observation)
        # 去除最大Q值对应的索引就是对应的动作
        return tf.argmax(self.statu, axis=1)

    def backward(self):
        # 使用均方差误差计算损失
        self.loss = tf.reduce_mean((self.target_q - self.pre_q) ** 2)
        # 使用Adm优化器
        self.optimizer = tf.train.RMSPropOptimizer(0.01).minimize(self.loss)

    def copy_params(self):
        # 复制权重，便于更新网络权重，将self.qNet的值复制给self.Target
        return [
            tf.assign(self.targetQNet.w1, self.qNet.w1),
            tf.assign(self.targetQNet.w2, self.qNet.w2),
            tf.assign(self.targetQNet.w3, self.qNet.w3),
            tf.assign(self.targetQNet.b1, self.qNet.b1),
            tf.assign(self.targetQNet.b2, self.qNet.b2),
            tf.assign(self.targetQNet.b3, self.qNet.b3),
        ]


if __name__ == '__main__':
    sample = Sample_game()
    drlnet = DRLNet()
    # 赋值discount因子为0.9
    drlnet.forward(0.9)
    drlnet.backward()
    copy_op = drlnet.copy_params()
    run_action_op = drlnet.play()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        batch_size = 200
        # 定义探索值为0.1
        explore = 0.1
        for k in range(100000):
            # 取出batch_size大小的数据组，生成对应的idxs列表和对应的experience列表组
            idxs, experiences = sample.get_sample(batch_size)
            # 重新定义对应参数的空列表，便于收集对应数据
            observations = []
            rewards = []
            actions = []
            next_observations = []
            dones = []

            # 将之前输出的经验值的经验取出，然后分别添加到对应的列表中
            for experience in experiences:
                observations.append(experience[0])
                rewards.append([experience[1]])
                actions.append([experience[2]])
                next_observations.append(experience[3])
                dones.append(experience[4])

            # 每10次复制一次参数
            if k % 10 == 0:
                print('-------------------copy_params---------------------')
                sess.run(copy_op)

            # 每次进行训练的loss和opt，输入对应的参数
            _loss, _ = sess.run([drlnet.loss, drlnet.optimizer], feed_dict={
                drlnet.observation: observations,
                drlnet.action: actions,
                drlnet.reward: rewards,
                drlnet.next_observation: next_observations,
                drlnet.done: dones
            })

            # 定义探索率为0.0001
            explore -= 0.0001
            if explore < 0.0001:
                explore = 0.0001

            # 输出损失值
            print('*********************', _loss, '************************')

            # 定义每次训练中失败的次数count
            count = 0
            # 重置采样
            run_observation = sample.reset()

            for idx in idxs:
                # 从500次后开始显示
                if k > 500:
                    sample.render()

                # 如果随机值小于探索值，就随机选一个动作作为探索
                if np.random.rand() < explore:
                    run_action = np.random.randint(0, 2)
                else:
                    # 否则就选择Q值最大的那个动作进行
                    run_action = sess.run(run_action_op, feed_dict={
                        drlnet.observation: [run_observation]
                    })[0]

                run_next_observation, run_reward, run_done, run_info = sample.env.step(run_action)

                sample.experience_pool[idx] = [run_observation, run_reward, run_action, run_next_observation, run_done]
                if run_done:
                    run_observation = sample.reset()
                    count += 1
                else:
                    run_observation = run_next_observation
            print('done---------------------------------------------', count)
