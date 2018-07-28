#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/15 21:54
# @Author  : SuperDan
# @File    : GA.py
# @Software: PyCharm
"""
基本的遗传（GA）算法实现
"""
import UAV
import numpy as np
import numpy.random as random
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
import matplotlib.pyplot as plt
import copy
import time

N=5                                                # 无人机的数量
ns=5                                               # 将终端时间T离散成5个部分
Dsafe=5                                            # 安全距离
Dcomm=45                                           # 通信距离

Tmax = 1e5                                         # 无人机动力范围
Tmin = 0
nmax = 20                                         # 无人机负载范围
nmin = -20
phimax = 1                                         # 无人机俯仰角范围
phimin = -1
deltatmax = 200                                    # 无人机终端时间T范围
deltatmin = 0

class GA(object):

    # ----------------------GA算法参数设置----------------------
    def __init__(self, Np, Tp, Dim):
        self.Np = Np                                               # 蚁群数量
        self.Tp = Tp                                               # 迭代次数
        self.Dim = Dim                                             # 搜索维度

        self.Pc = 0.8                                              # 交叉概率
        self.Pm = 0.1                                              # 变异概率
        self.rou = 0.8                                             # 学习常数
        self.beata = 0.2                                           # 惯性常数

        self.deltap = np.zeros((self.Np, self.Dim))                # 迭代间的适应度值的差
        self.sp = np.zeros((self.Np, self.Dim))                    # 进化趋势
        self.acc = np.zeros((self.Np, self.Dim))

        # 范围限制
        T_max = np.ones(N * ns) * Tmax
        T_min = np.ones(N * ns) * Tmin
        n_max = np.ones(N * ns) * nmax
        n_min = np.ones(N * ns) * nmin
        phi_max = np.ones(N * ns) * phimax
        phi_min = np.ones(N * ns) * phimin
        deltat_max = np.ones(1) * deltatmax
        deltat_min = np.ones(1) * deltatmin
        self.Xmax = np.hstack((T_max, n_max, phi_max, deltat_max)) # 位置最大值
        self.Xmin = np.hstack((T_min, n_min, phi_min, deltat_min))

        # 初始化种群
        self.xp = np.zeros((self.Np, self.Dim))
        self.nf = np.zeros((self.Np, self.Dim))
        self.xBest = np.zeros(self.Dim)
        self.maxFit = 0
        self.Fit = np.zeros(self.Np)

        self.trace = np.zeros(self.Tp)
    # ----------------------目标函数----------------------
    @staticmethod
    def func(x):
        My_UAV = UAV.UAV(N=5, ns=5, Dsafe=5, Dcomm=45, sigma=1.8e7)
        My_UAV.init_state()
        My_UAV.UAV_state(x[0:N * ns], x[N * ns:2 * N * ns], x[2 * N * ns:3 * N * ns], x[-1])
        J = My_UAV.UAV_fitness(x[-1])
        return -J[0]

    # ----------------------实施算法----------------------
    def solution(self):
        self.xp = random.rand(self.Np, self.Dim) * (self.Xmax - self.Xmin) + self.Xmin
        for i in range(self.Np):
            self.Fit[i] = self.func(self.xp[i, :])                 # 适应度值
        self.maxFit = np.max(self.Fit)                                       # 适应度的最大值
        index = np.argmax(self.Fit)  # 得到适应度值最大的索引
        self.xBest = self.xp[index, :]  # 最优个体
        for i in range(self.Tp):
            self.selection()
            self.crossover()
            self.mutation()
            self.xp = copy.deepcopy(self.nf)

            for j in range(self.Np):
                self.Fit[j] = self.func(self.xp[j, :])  # 适应度值
            if np.max(self.Fit) > self.maxFit:
                self.maxFit = np.max(self.Fit)
                index = np.argmax(self.Fit)  # 得到适应度值最大的索引
                self.xBest = self.xp[index, :]  # 最优个体
            self.trace[i] = -self.maxFit                           # 历代最优适应度值
            print('第%d次迭代'% i,'适应度值为%2e'%self.trace[i])

        return self.trace

            # ----------------------基于轮盘赌的复制操作----------------------
    def selection(self):

        minFit = np.min(self.Fit)                                       # 适应度的最小值

        Fit_norm = (self.Fit - minFit) / (self.maxFit - minFit)              # 归一化适应度值
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
        for i in range(0, self.Np, 2):
            p = random.rand()
            if p < self.Pc:
                r = random.rand()
                self.nf[i+1, :] = r * self.nf[i+1, :] + (1 - r) * self.nf[i, :]
                self.nf[i, :] = r * self.nf[i, :] + (1 - r) * self.nf[i+1, :]

    # ----------------------基于概率的变异操作----------------------
    # def mutation(self):
    #     for i in range(self.Np):
    #         for j in range(self.Dim):
    #             if random.rand() < self.Pm:
    #                 temp = random.randint(0,2)
    #                 if temp == 0:
    #                     self.nf[i, j] = self.nf[i, j] + 0.8 * (self.xBest[j] - self.nf[i, j]) * np.abs(random.randn())
    #                 else:
    #                     self.nf[i, j] = self.nf[i, j] - 0.8 * (self.xBest[j] - self.nf[i, j]) * np.abs(random.randn())
    def mutation(self):
        for i in range(self.Np):
            for j in range(self.Dim):
                if random.rand() < self.Pm:
                    self.nf[i, j] = self.nf[i, j] + self.beata * self.deltap[i, j] + self.rou * self.sp[i, j]
                    self.deltap[i, j] = (self.xBest[j] - self.nf[i, j]) * np.abs(random.randn())
                    if self.func(self.nf[i, :]) > self.Fit[i]:
                        self.acc[i, j] = 1
                    else:
                        self.acc[i, j] = 0
                    self.sp[i, j] = self.beata * self.acc[i, j] * self.deltap[i, j] + self.rou * self.sp[i, j]

    # ------------------进行绘图------------------
    def UAV_plot(self):

        T = self.xBest[0:N * ns]
        n = self.xBest[N * ns:2 * N * ns]
        phi = self.xBest[2 * N * ns:3 * N * ns]
        deltat = self.xBest[-1]

        T = np.reshape(T, (N, ns))
        n = np.reshape(n, (N, ns))
        phi = np.reshape(phi, (N, ns))

        # 绘制迭代次数-适应度值
        plt.figure()
        plt.title('适应度进化函数')
        plt.xlabel('迭代次数')
        plt.ylabel('适应度值')
        plt.plot(self.trace, color='g', linewidth=2)

        # 绘制最终的无人机位置的二维图
        plt.figure()
        plt.title('二维图')
        My_UAV = UAV.UAV(N=5, ns=5, Dsafe=5, Dcomm=45, sigma=1.8e7)
        My_UAV.init_state()
        My_UAV.UAV_state(T, n, phi, deltat)
        J = My_UAV.UAV_fitness(self.xBest[-1])
        x = J[1]
        y = J[2]
        plt.annotate('中心无人机', xy=(x[2, -1], y[2, -1]))
        plt.plot(x[:, -1], y[:, -1], '-o', markersize=7)
        plt.show()
# ------------------程序运行------------------
def main():
    start = time.clock()
    Dim = N * ns * 3 + 1
    My_GA = GA(Np=200, Tp=200, Dim=Dim)
    fit = My_GA.solution()
    end = time.clock()
    second = end - start
    print('运行花费时间%fs' % second)
    print('适应度最优值：%.2e' % fit[-1])
    My_GA.UAV_plot()

if __name__ == '__main__':
    main()

