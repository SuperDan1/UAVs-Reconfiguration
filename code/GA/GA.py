#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/15 21:54
# @Author  : SuperDan
# @File    : GA.py
# @Software: PyCharm
"""
基本的遗传（GA）算法实现
"""
import numpy as np
import numpy.random as random
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
import matplotlib.pyplot as plt
import copy

class GA(object):

    # ----------------------GA算法参数设置----------------------
    def __init__(self, Np, Tp, Dim):
        self.Np = Np                                               # 蚁群数量
        self.Tp = Tp                                               # 迭代次数
        self.Dim = Dim                                             # 搜索维度

        self.Pc = 0.8                                              # 交叉概率
        self.Pm = 0.2                                              # 变异概率
        self.Xmax = 20                                             # 位置最大值
        self.Xmin = -20

        # 初始化种群
        self.xp = np.zeros((self.Np, self.Dim))
        self.nf = np.zeros((self.Np, self.Dim))
        self.xBest = np.zeros(self.Dim)
        self.maxFit = 0

    # ----------------------目标函数----------------------
    @staticmethod
    def func(x):
        result = np.sum(np.power(x, 2))
        return result

    # ----------------------实施算法----------------------
    def solution(self):
        self.xp = random.rand(self.Np, self.Dim) * (self.Xmax - self.Xmin) + self.Xmin
        trace = np.zeros(self.Tp)
        for i in range(self.Tp):
            self.selection()
            self.crossover()
            self.mutation()
            # Fit = np.zeros(self.Np)
            # for j in range(self.Np):
            #     Fit[j] = self.func(self.nf[j, :])  # 适应度值
            # maxFit = np.max(Fit)
            self.xp = copy.deepcopy(self.nf)
            self.xp[0, :] = self.xBest                             # 保留最优个体在新种群内
            trace[i] = -self.maxFit                                # 历代最优适应度值
        return trace

            # ----------------------基于轮盘赌的复制操作----------------------
    def selection(self):
        Fit = np.zeros(self.Np)
        for i in range(self.Np):
            Fit[i] = -self.func(self.xp[i, :])                      # 适应度值
        self.maxFit = np.max(Fit)                                       # 适应度的最大值
        minFit = np.min(Fit)                                       # 适应度的最小值
        index = np.argmax(Fit)                                     # 得到适应度值最大的索引
        self.xBest = self.xp[index, :]                             # 最优个体
        Fit_norm = (Fit - minFit) / (self.maxFit - minFit)              # 归一化适应度值
        self.nf = np.zeros((self.Np, self.Dim))                    # 新种群

        sum_Fit = np.sum(Fit_norm)
        fitvalue = [f / sum_Fit for f in Fit_norm]                 # 依适应度的概率值
        fitvalue = np.cumsum(fitvalue)                             # 累加和
        ms = np.sort(random.rand(self.Np))
        fiti = 0
        newi = 0
        while newi < self.Np:
            if ms[newi] < fitvalue[fiti]:
                self.nf[newi, :] = self.xp[fiti, :]
                newi += 1
            else:
                fiti += 1

    # ----------------------基于概率的交叉操作----------------------
    def crossover(self):
        for j in range(self.Dim):
            for i in range(0, self.Np, 2):
                p = random.rand()
                if p < self.Pc:
                    self.nf[i+1, j] = self.Pc * self.nf[i+1, j] + (1 - self.Pc) * self.nf[i, j]
                    self.nf[i, j] = self.Pc * self.nf[i, j] + (1 - self.Pc) * self.nf[i+1, j]

    # ----------------------基于概率的变异操作----------------------
    def mutation(self):
        for i in range(self.Np):
            for j in range(self.Dim):
                if random.rand() < self.Pm:
                    temp = random.randint(0,2)
                    if temp == 0:
                        self.nf[i, j] = self.nf[i, j] + 0.8 * (self.xBest[j] - self.nf[i, j]) * np.abs(random.randn())
                    else:
                        self.nf[i, j] = self.nf[i, j] - 0.8 * (self.xBest[j] - self.nf[i, j]) * np.abs(random.randn())

# ----------------------运行程序----------------------
My_GA = GA(Np=100, Tp=200, Dim=10)
fit = My_GA.solution()
print('适应度最优值：%.2e' % fit[-1])
plt.figure()
plt.title('适应度进化函数')
plt.xlabel('迭代次数')
plt.ylabel('适应度值')

plt.plot(fit, color='g', linewidth=2)
plt.show()
