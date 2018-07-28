#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/11 16:01
# @Author  : SuperDan
# @File    : PSO.py
# @Software: PyCharm

"""
    基本的粒子群算法（PSO）实现
"""
import numpy as np
import numpy.random as random
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
import matplotlib.pyplot as plt
import copy

class PSO(object):
    #----------------------PSO算法参数设置----------------------
    def __init__(self, Np, Tp, Dim):
        self.wmax = 0.9                                                          # 惯性权重
        self.wmin = 0.4
        self.w = 0
        self.c1 = 1.5                                                            # 学习因子
        self.c2 = 1.5
        self.Np = Np                                                             # 群体粒子个数
        self.Tp = Tp                                                             # 最大迭代个数
        self.Dim = Dim                                                           # 搜索维度

        self.Xmax = 20                                                           # 位置最大值
        self.Xmin = -20
        self.Vmax = 1                                                            # 位置改变速度最大值
        self.Vmin = -1

        # 所有粒子的位置和速度
        self.xp = np.zeros((self.Np, self.Dim))
        self.vp = np.zeros((self.Np, self.Dim))

        # 个体最佳位置和全局最佳位置
        self.p = np.zeros((self.Np, self.Dim))
        self.g = np.zeros((1, self.Dim))

        # 个体最佳适应值和全局最佳适应值
        self.pbest = np.zeros(self.Np)
        self.Gbest = np.inf

    # ----------------------目标函数----------------------
    @staticmethod
    def func(x):
        result = np.sum(np.power(x,2))
        return result

    # ----------------------初始化种群----------------------
    def init_population(self):
        # 初始化种群个体（限制速度和位置）
        self.xp = random.rand(self.Np, self.Dim) * (self.Xmax - self.Xmin) + self.Xmin
        self.vp = random.rand(self.Np, self.Dim) * (self.Vmax - self.Vmin) + self.Vmin

        # 初始化个体最优位置和最优值
        # self.p = self.xp   注意这样会使两者内存地址一样，一个变量的值变化，另一个变量的值也会随之变化
        self.p = copy.deepcopy(self.xp)
        for i in range(self.Np):
            self.pbest[i] = self.func(self.p[i, :])

        # 初始化全局最优位置和最优值
        for i in range(self.Np):
            if self.pbest[i] < self.Gbest:
                self.Gbest = self.pbest[i]
                self.g = self.p[i, :]

    # ------------------按照公式依次迭代直到满足精度或者迭代次数------------------
    def iterator(self):
        gb = np.zeros(self.Tp)
        for i in range(self.Tp):
            fitness = np.zeros(self.Np)
            for j in range(self.Np):
                fitness[j] = self.func(self.xp[j, :])
                # 更新个体最优位置和最优值
                if fitness[j] < self.pbest[j]:
                    self.p[j, :] = self.xp[j, :]
                    self.pbest[j] = fitness[j]
                # 更新全局最优位置和最优值
                if self.pbest[j] < self.Gbest:
                    self.g = self.p[j, :]
                    self.Gbest = self.pbest[j]
                # 更新位置和速度
                self.w = self.wmax - (self.wmax - self.wmin) * i / self.Tp
                self.vp[j, :] = self.w * self.vp[j, :] + self.c1 * random.rand() * (self.p[j, :] - self.xp[j, :]) + self.c2 * random.rand() * (self.g - self.xp[j, :])

                self.xp[j, :] = self.xp[j, :] + self.vp[j, :]
                # 边界条件处理
                for k in  range(self.Dim):
                    if (self.vp[j, k] > self.Vmax) or (self.vp[j, k] < self.Vmin):
                        self.vp[j, k] = random.rand() * (self.Vmax - self.Vmin) + self.Vmin
                    if self.xp[j, k] > self.Xmax or self.xp[j, k] < self.Xmin:
                        self.xp[j, k] = random.rand() * (self.Xmax - self.Xmin) + self.Xmin
            # 记录历代全局最优值
            gb[i] = self.Gbest
        return gb
# ------------------程序运行------------------
My_PSO = PSO(Np=100, Tp=200, Dim=20)
My_PSO.init_population()
fit = My_PSO.iterator()

# ------------------程序运行------------------
print('适应度最优值：%.2e' % fit[-1])
plt.figure()
plt.title('适应度进化函数')
plt.xlabel('迭代次数')
plt.ylabel('适应度值')

plt.plot(fit, color='g', linewidth=2)
plt.show()

