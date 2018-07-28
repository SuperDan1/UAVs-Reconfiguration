#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/26 16:36
# @Author  : SuperDan
# @File    : HPSOGA.py
# @Software: PyCharm
"""
HPSOGA算法实现
"""
import numpy as np
from numpy import random
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
import matplotlib.pyplot as plt
import copy

class HPSOGA(object):
    #--------------------Step1：HPSOGA算法参数设置--------------------
    def __init__(self, Np, Tp, Dim):
        self.Np = Np                                                   # 群体粒子个数
        self.Tp = Tp                                                   # 最大迭代次数
        self.Dim = Dim                                                 # 解的维度

        # PSO算法参数设置
        self.wmax = 0.9                                                # 惯性权重
        self.wmin = 0.4
        self.w = 0
        self.c1 = 1.5                                                  # 学习因子
        self.c2 = 1.5

        # GA算法参数设置
        self.Pc = 0.8                                                  # 交叉概率
        self.Pm = 0.2                                                  # 变异概率

        self.Xmax = 20                                                 # 位置范围
        self.Xmin = -20
        self.Vpmax = 1                                                 # 位置改变速度范围
        self.Vpmin = -1

        self.pe = 1                                                  # 混合概率
        self.N_PSO = int(self.Np * self.pe)                            # PSO算法子群的个数
        self.N_GA = self.Np - self.N_PSO                               # GA算法子群的个数

        # 所有粒子的位置
        self.xp = np.zeros((self.Np, self.Dim))

        # 粒子的位置和速度
        self.pop_PSO = np.zeros((self.N_PSO, self.Dim))
        self.pop_GA = np.zeros((self.N_GA, self.Dim))
        self.vp = np.zeros((self.N_PSO, self.Dim))

        # 个体最佳位置和全局最佳位置
        self.p_PSO = np.zeros((self.N_PSO, self.Dim))

        self.G = np.zeros((1, self.Dim))
        self.G_PSO = np.zeros((1,self.Dim))
        self.G_GA = np.zeros((1,self.Dim))

        # 个体最佳适应值和全局最佳适应值
        self.pbest_PSO = np.zeros(self.N_PSO)
        self.Gbest = np.inf
        self.Gbest_PSO = np.inf
        self.Gbest_GA = np.inf
        self.fit = np.zeros(self.Np)
        self.trace = np.zeros(self.Tp)

        self.nf = np.zeros((self.N_GA, self.Dim))  # 新种群

    # -----------------------目标函数-----------------------
    @staticmethod
    def func(x):
        result = np.sum(np.power(x, 2))
        return result

    # ----------------------Step2：初始化种群，计算适应度值----------------------
    def init_population(self):
        # 初始化种群个体（限制速度和位置）
        self.xp = random.rand(self.Np, self.Dim) * (self.Xmax - self.Xmin) + self.Xmin
        self.pop_PSO = self.xp[:self.N_PSO]
        self.pop_GA = self.xp[self.N_PSO:]

        self.vp = random.rand(self.N_PSO, self.Dim) * (self.Vpmax - self.Vpmin) + self.Vpmin
        self.p_PSO = copy.deepcopy(self.pop_PSO)
        for i in range(self.N_PSO):
            self.pbest_PSO[i] = self.func(self.p_PSO[i])
        # 初始化全局最优位置和最优值
        for j in range(self.Np):
            self.fit[j] = self.func(self.xp[j, :])
        if min(self.fit) < self.Gbest:
            self.Gbest = min(self.fit)
            index = np.argmin(self.fit)
            self.G = self.xp[index, :]


    # ------------------按照公式依次迭代直到满足精度或者迭代次数------------------
    def iterator(self):
        for i in range(self.Tp):
            # 计算微粒代价函数并保留最优微粒的位置和代价函数
            for j in range(self.Np):
                self.fit[j] = self.func(self.xp[j, :])
            if min(self.fit) < self.Gbest:
                self.Gbest = min(self.fit)
                index = np.argmin(self.fit)
                self.G = self.xp[index, :]
            # 用混合概率pe将微粒群分为两个子群：一个子群为粒子群，另一个子群为染色体子群
            # random.shuffle(self.xp)
            self.pop_PSO = self.xp[:self.N_PSO]
            self.pop_GA = self.xp[self.N_PSO:]

            # PSO算法
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
                        self.G_PSO = self.p_PSO[j, :]
                        self.Gbest_PSO = self.pbest_PSO[j]
                    # 更新位置和速度
                    self.w = self.wmax - (self.wmax - self.wmin) * i / self.Tp
                    self.vp[j, :] = self.w * self.vp[j, :] + self.c1 * random.rand() * (
                                self.p_PSO[j, :] - self.pop_PSO[j, :]) + self.c2 * random.rand() * (self.G - self.pop_PSO[j, :])

                    self.pop_PSO[j, :] = self.pop_PSO[j, :] + self.vp[j, :]
                    # 边界条件处理
                    for k in range(self.Dim):
                        if (self.vp[j, k] > self.Vpmax) or (self.vp[j, k] < self.Vpmin):
                            self.vp[j, k] = random.rand() * (self.Vpmax - self.Vpmin) + self.Vpmin
                        if self.pop_PSO[j, k] > self.Xmax or self.pop_PSO[j, k] < self.Xmin:
                            self.pop_PSO[j, k] = random.rand() * (self.Xmax - self.Xmin) + self.Xmin

            # GA算法
            if self.N_GA != 0:

                self.selection()
                self.crossover()
                self.mutation()
                self.nf[0, :] = self.G_GA
                # 组合粒子群和染色体群
                self.pop_GA = copy.deepcopy(self.nf)
                # 计算微粒代价函数并保留最优微粒的位置和代价函数
                fitness = np.zeros(self.N_GA)
                for j in range(self.N_GA):
                    fitness[j] = self.func(self.pop_GA[j, :])
                if min(fitness) < self.Gbest_GA:
                    self.Gbest_GA = min(fitness)
                    index = np.argmin(fitness)
                    self.G_GA = self.pop_GA[index, :]

            if self. Gbest_PSO < self.Gbest_GA:
                self.Gbest_GA = copy.deepcopy(self.Gbest_PSO)
                self.G_GA = copy.deepcopy(self.G_PSO)
            else:
                self.Gbest_PSO = copy.deepcopy(self.Gbest_GA)
                self.G_PSO = copy.deepcopy(self.G_GA)
            self.xp = np.vstack((self.pop_PSO, self.pop_GA))

            self.trace[i] = self.Gbest
        return self.trace

# ----------------------基于轮盘赌的复制操作----------------------
    def selection(self):
        fitness = np.zeros(self.N_GA)
        for i in range(self.N_GA):
            fitness[i] = -self.func(self.pop_GA[i, :])                 # 适应度值
        maxFit = np.max(fitness)                                       # 适应度的最大值
        minFit = np.min(fitness)                                       # 适应度的最小值
        Fit_norm = (fitness - minFit) / (maxFit - minFit)              # 归一化适应度值
        sum_Fit = np.sum(Fit_norm)
        fitvalue = [f / sum_Fit for f in Fit_norm]                 # 依适应度的概率值
        fitvalue = np.cumsum(fitvalue)                             # 累加和
        ms = np.sort(random.rand(self.N_GA))
        fiti = 0
        newi = 0
        while newi < self.N_GA:
            if ms[newi] < fitvalue[fiti]:
                self.nf[newi, :] = self.pop_GA[fiti, :]
                newi += 1
            else:
                fiti += 1

    # ----------------------基于概率的交叉操作----------------------
    def crossover(self):
        for j in range(self.Dim):
            for i in range(0, self.N_GA, 2):
                p = random.rand()
                if p < self.Pc:
                    self.nf[i+1, j] = self.Pc * self.nf[i+1, j] + (1 - self.Pc) * self.nf[i, j]
                    self.nf[i, j] = self.Pc * self.nf[i, j] + (1 - self.Pc) * self.nf[i+1, j]

    # ----------------------基于概率的变异操作----------------------
    def mutation(self):
        for i in range(self.N_GA):
            for j in range(self.Dim):
                if random.random() < self.Pm:
                    temp = random.randint(0,2)
                    if temp == 0:
                        self.nf[i, j] = self.nf[i, j] + 0.8 * (self.G[j] - self.nf[i, j]) * np.abs(random.randn())
                    else:
                        self.nf[i, j] = self.nf[i, j] - 0.8 * (self.G[j] - self.nf[i, j]) * np.abs(random.randn())

My_HPSOGA = HPSOGA(Np=100, Tp=200, Dim=10)
My_HPSOGA.init_population()
fit = My_HPSOGA.iterator()
print('适应度最优值：%.2e' % fit[-1])
plt.figure()
plt.title('适应度进化函数')
plt.xlabel('迭代次数')
plt.ylabel('适应度值')

plt.plot(fit, color='G', linewidth=2)
plt.show()