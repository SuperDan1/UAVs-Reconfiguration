#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/11 16:01
# @Author  : SuperDan
# @File    : PSO.py
# @Software: PyCharm

"""
    基本的粒子群算法（PSO）实现
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

        # 范围限制
        T_max = np.ones(N * ns) * Tmax
        T_min = np.ones(N * ns) * Tmin
        n_max = np.ones(N * ns) * nmax
        n_min = np.ones(N * ns) * nmin
        phi_max = np.ones(N * ns) * phimax
        phi_min = np.ones(N * ns) * phimin
        deltat_max = np.ones(1) * deltatmax
        deltat_min = np.ones(1) * deltatmin
        self.Xmax = np.hstack((T_max, n_max, phi_max, deltat_max))                                                           # 位置最大值
        self.Xmin = np.hstack((T_min, n_min, phi_min, deltat_min))
        self.Vmax = 0.01 * (self.Xmax -  self.Xmin)                              # 位置改变速度最大值
        self.Vmin = -self.Vmax

        # 所有粒子的位置和速度
        self.xp = np.zeros((self.Np, self.Dim))
        self.vp = np.zeros((self.Np, self.Dim))

        # 个体最佳位置和全局最佳位置
        self.p = np.zeros((self.Np, self.Dim))
        self.g = np.zeros((1, self.Dim))

        # 个体最佳适应值和全局最佳适应值
        self.pbest = np.zeros(self.Np)
        self.Gbest = np.inf

        # 记录每代的最佳适应度值
        self.gb = np.zeros(self.Tp)
    # ----------------------目标函数----------------------
    @staticmethod
    def func(x):
        My_UAV = UAV.UAV(N=5, ns=5, Dsafe=5, Dcomm=45, sigma=1.8e7)
        My_UAV.init_state()
        My_UAV.UAV_state(x[0:N*ns], x[N*ns:2*N*ns], x[2*N*ns:3*N*ns], x[-1])
        J = My_UAV.UAV_fitness(x[-1])
        return J[0]

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
                    if (self.vp[j, k] > self.Vmax[k]) or (self.vp[j, k] < self.Vmin[k]):
                        self.vp[j, k] = random.rand() * (self.Vmax[k] - self.Vmin[k]) + self.Vmin[k]
                    if self.xp[j, k] > self.Xmax[k] or self.xp[j, k] < self.Xmin[k]:
                        self.xp[j, k] = random.rand() * (self.Xmax[k] - self.Xmin[k]) + self.Xmin[k]
            # 记录历代全局最优值
            self.gb[i] = self.Gbest
        return self.gb

    # ------------------进行绘图------------------
    def UAV_plot(self):

        # 绘制迭代次数-适应度值
        plt.figure()
        plt.title('适应度进化函数')
        plt.xlabel('迭代次数')
        plt.ylabel('适应度值')
        plt.plot(self.gb, color='g', linewidth=2)


        # 绘制最终的无人机位置的二维图
        plt.figure()
        plt.title('二维图')
        My_UAV = UAV.UAV(N=5, ns=5, Dsafe=5, Dcomm=45, sigma=1.8e7)
        My_UAV.init_state()
        My_UAV.UAV_state(self.g[0:N * ns], self.g[N * ns:2 * N * ns], self.g[2 * N * ns:3 * N * ns], self.g[-1])
        J = My_UAV.UAV_fitness(self.g[-1])
        x = J[1]
        y = J[2]
        plt.annotate('中心无人机',xy=(x[2, -1], y[2,-1]))
        plt.plot(x[:, -1], y[:, -1], '-o',markersize=7)
        plt.show()

# ------------------程序运行------------------
def main():
    start = time.clock()
    Dim = N * ns * 3 + 1
    My_PSO = PSO(Np=400, Tp=370, Dim=Dim)
    My_PSO.init_population()
    fit = My_PSO.iterator()

    end = time.clock()
    second = end - start
    print('运行花费时间%fs' % second)
    print('适应度最优值：%.2e' % fit[-1])
    My_PSO.UAV_plot()
if __name__ == '__main__':
    main()

