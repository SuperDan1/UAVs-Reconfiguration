#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/27 15:49
# @Author  : SuperDan
# @File    : HPSOGA1.py
# @Software: PyCharm
"""
HPSOGA算法实现
"""
import numpy as np
import numpy.random as random
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
import matplotlib.pyplot as plt
import copy

class HPSOGA(object):
    #----------------------Step1:算法参数设置及初始化----------------------
    def __init__(self, Np, Tp, Dim):
        self.Np = Np                                           # 群体粒子个数
        self.Tp = Tp                                           # 最大迭代个数
        self.Dim = Dim                                         # 搜索维度

        # PSO算法参数设置
        self.wmax = 0.9                                        # 惯性权重
        self.wmin = 0.4
        self.w = 0

        self.c1 = 1.5                                          # 学习因子
        self.c2 = 1.5

        # GA算法参数设置
        self.Pc = 0.8                                          # 交叉概率
        self.Pm = 0.2                                          # 变异概率

        # benchmark函数范围
        self.Xmax = 20                                         # 变量范围
        self.Xmin = -20

        self.Vpmax = 1                                         # PSO算法中速度大小
        self.Vpmin = -1

        self.pe = 1                                            # 混合概率
        self.N_PSO = int(self.Np * self.pe)                    # PSO算法子群的个数
        self.N_GA = self.Np - self.N_PSO                       # GA算法子群的个数

        # 初始化所有粒子的位置
        self.xp = np.zeros((self.Np, self.Dim))
        self.fit = np.zeros(self.Np)                          # 所有粒子的适应度值
        self.g = np.zeros((1, self.Dim))                      # 全局最佳位置
        self.Gbest = np.inf                                   # 全局最佳适应度值

        # 初始化PSO算法子群的位置和速度
        self.pop_PSO = np.zeros((self.N_PSO, self.Dim))
        self.vp = np.zeros((self.N_PSO, self.Dim))
        self.pbest_PSO = np.zeros(self.N_PSO)                 # PSO算法子群的个体最佳适应度值
        self.p_PSO = np.zeros((self.N_PSO, self.Dim))         # PSO算法子群的个体最佳位置
        self.Gbest_PSO = np.inf                               # PSO算法子群的最佳适应度值
        self.g_PSO = np.zeros((1, self.Dim))                  # PSO算法子群的全局最佳位置

        # 初始化GA算法子群的位置
        self.pop_GA  = np.zeros((self.N_GA, self.Dim))
        self.Gbest_GA = np.inf                                # GA算法子群的最佳适应度值
        self.g_GA = np.zeros((1, self.Dim))                   # GA算法子群的全局最佳位置
        self.nf = np.zeros((self.N_GA, self.Dim))             # 新种群

        self.trace = np.zeros(self.Tp)                        # 记录每代最佳适应度值

    # ----------------------目标函数----------------------
    @staticmethod
    def func(x):
        result = np.sum(np.power(x,2))
        return result

    # ----------------------初始化种群,计算适应度值----------------------
    def init_population(self):
        # 初始化种群个体（限制速度和位置）
        self.xp = random.rand(self.Np, self.Dim) * (self.Xmax - self.Xmin) + self.Xmin
        for i in range(self.Np):
            self.fit[i] = self.func(self.xp[i, :])
        self.Gbest = min(self.fit)
        index = np.argmin(self.fit)
        self.g = self.xp[index, :]

        # PSO算法子群
        self.pop_PSO = self.xp[:self.N_PSO]
        self.vp = random.rand(self.Np, self.Dim) * (self.Vpmax - self.Vpmin) + self.Vpmin
        self.p_PSO = copy.deepcopy(self.pop_PSO)
        for i in range(self.N_PSO):
            self.pbest_PSO[i] = self.func(self.pop_PSO[i, :])
        self.Gbest_PSO = min(self.pbest_PSO)
        index = np.argmin(self.pbest_PSO)
        self.g_PSO = self.pop_PSO[index, :]

        # GA算法子群
        if self.N_GA != 0:
            self.pop_GA = self.xp[self.N_PSO:]
            fitness = np.zeros(self.N_GA)
            for i in range(self.N_GA):
                fitness[i] = self.func(self.pop_GA[i, :])
            self.Gbest_GA = min(fitness)
            index = np.argmin(fitness)
            self.g_GA = self.pop_GA[index, :]

    # ------------------按照公式依次迭代直到满足精度或者迭代次数------------------
    def iterator(self):
        for i in range(self.Tp):
            # -------------------Step2:计算微粒代价函数并保留最优微粒的位置和适应度值-------------------
            for j in range(self.Np):
                self.fit[j] = self.func(self.xp[j, :])
            if min(self.fit) < self.Gbest:
                self.Gbest = min(self.fit)
                index = np.argmin(self.fit)
                self.g = self.xp[index, :]

            # -------------------Step3:分离种群为PSO算法子群和GA算法子群-------------------
            # random.shuffle(self.xp)
            self.pop_PSO = self.xp[:self.N_PSO]
            self.pop_GA = self.xp[self.N_PSO:]

            # -------------------Step4:对子群使用PSO算法-------------------
            if self.N_PSO != 0:
                fitness = np.zeros(self.N_PSO)
                for j in range(self.N_PSO):
                    fitness[j] = self.func(self.pop_PSO[j, :])
                    # 更新个体最优位置和最优值
                    if fitness[j] < self.pbest_PSO[j]:
                        self.p_PSO[j, :] = self.pop_PSO[j, :]
                        self.pbest_PSO[j] = fitness[j]
                    # 更新全局最优位置和最优值
                    if self.pbest_PSO[j] < self.Gbest_PSO:
                        self.g_PSO = self.p_PSO[j, :]
                        self.Gbest_PSO = self.pbest_PSO[j]
                    # 更新位置和速度
                    self.w = self.wmax - (self.wmax - self.wmin) * i / self.Tp
                    self.vp[j, :] = self.w * self.vp[j, :] + self.c1 * random.rand() * (self.p_PSO[j, :] - self.pop_PSO[j, :]) + self.c2 * random.rand() * (self.g_PSO - self.pop_PSO[j, :])

                    self.pop_PSO[j, :] = self.pop_PSO[j, :] + self.vp[j, :]
                    # 边界条件处理
                    for k in  range(self.Dim):
                        if (self.vp[j, k] > self.Vpmax) or (self.vp[j, k] < self.Vpmin):
                            self.vp[j, k] = random.rand() * (self.Vpmax - self.Vpmin) + self.Vpmin
                        if self.pop_PSO[j, k] > self.Xmax or self.pop_PSO[j, k] < self.Xmin:
                            self.pop_PSO[j, k] = random.rand() * (self.Xmax - self.Xmin) + self.Xmin

            # -------------------Step5:对子群使用GA算法-------------------
            if self.N_GA != 0:
                pass
            # -------------------Step6:PSO子群和GA子群进行信息共享-------------------
            if self.Gbest_PSO < self.Gbest_GA:
                self.Gbest_GA = copy.deepcopy(self.Gbest_PSO)
                self.g_GA = copy.deepcopy(self.g_PSO)
            else:
                self.Gbest_PSO = copy.deepcopy(self.Gbest_GA)
                self.g_PSO = copy.deepcopy(self.g_GA)
            self.xp = np.vstack((self.pop_PSO, self.pop_GA))

            # 记录历代全局最优值
            self.trace[i] = self.Gbest
        return self.trace

# -------------------------程序运行-------------------------
My_HPSOGA = HPSOGA(Np=100, Tp=200, Dim=20)
My_HPSOGA.init_population()
fit = My_HPSOGA.iterator()
print('适应度最优值：%.2e' % fit[-1])
plt.figure()
plt.title('适应度进化函数')
plt.xlabel('迭代次数')
plt.ylabel('适应度值')

plt.plot(fit, color='G', linewidth=2)
plt.show()
