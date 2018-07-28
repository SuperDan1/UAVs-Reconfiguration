#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/13 10:49
# @Author  : SuperDan
# @File    : UAV_fitness.py
# @Software: PyCharm
"""
对无人机编队的建模。
"""
import numpy as np
import numpy.random as random
import copy

class UAV(object):
    # ----------------------对实例属性进行初始化----------------------
    def __init__(self, N=5, ns=5, Dsafe=5, Dcomm=45, sigma=1.8e7):
        """
        :param N:    无人机群个数
        :param ns:   将终端时间T离散成5个部分
        :param Dsafe:安全距离,单位：Km
        :param Dcomm:通信距离单位：Km
        :param sigma:惩罚系数
        """
        self.N = N
        self.ns = ns
        self.Dsafe = Dsafe
        self.Dcomm = Dcomm
        self.sigma = sigma

        # 变量范围限制
        self.Lmax = 20                                             # UAV初始化位置范围，单位：Km
        self.Lmin = -20
        self.Vmax = 340                                            # UAV的速度范围，单位：m/s
        self.Vmin = 0
        self.gamma_max = 1                                         # 航迹角的范围，单位：弧度
        self.gamma_min = -1
        self.ka_max = np.pi                                        # 航迹角的范围，单位：弧度
        self.ka_min = -np.pi

        # 状态变量
        self.x0 = np.zeros(self.N)                                 # 无人机群的初始位置
        self.y0 = np.zeros(self.N)
        self.z0 = np.zeros(self.N)
        self.v0 = np.zeros(self.N)                                 # 无人机群的初始速度
        self.D0 = np.zeros(self.N)                                 # 无人机群的初始气动阻力，单位：N
        self.W  = np.zeros(1)                                      # 无人机群的重量，单位：N
        self.gamma0 = np.zeros(self.N)
        self.ka0 = np.zeros(self.N)

        # 无人机编队第i架无人机相对于中心无人机期望的相对坐标值
        self.xe = [-20, -10, 0, -10, -20]
        self.ye = [20, 10, 0, -10, -20]
        self.ze = [0, 0, 0, 0, 0]

        # 记录每个时间段的状态变量
        self.x = np.zeros((self.N, self.ns))
        self.y = np.zeros((self.N, self.ns))
        self.z = np.zeros((self.N, self.ns))
        self.v = np.zeros((self.N, self.ns))
        self.D = np.zeros((self.N, self.ns))
        self.gamma = np.zeros((self.N, self.ns))
        self.ka = np.zeros((self.N, self.ns))

    # ----------------------对无人机群的状态进行随机初始化----------------------
    def init_state(self):
        self.x0 = random.rand(self.N) * (self.Lmax - self.Lmin) + self.Lmin
        self.y0 = random.rand(self.N) * (self.Lmax - self.Lmin) + self.Lmin
        self.z0 = random.rand(self.N) * (self.Lmax - self.Lmin) + self.Lmin
        self.v0 = random.rand(self.N) * (self.Vmax - self.Vmin) + self.Vmin
        self.D0 = 1 / 2 * 0.08 * 1.29 * np.power(self.v0, 2) * 20
        self.W = 1e5
        self.gamma0 = random.rand(self.N) * (self.gamma_max - self.gamma_min) + self.gamma_min
        self.ka0 = random.rand(self.N) * (self.ka_max - self.ka_min) + self.ka_min

    # ----------------------计算两架无人机之间的距离----------------------
    @staticmethod
    def distance(x1, y1, z1, x2, y2, z2):
        d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        return d

    # ----------------------根据无人机运动方程进行迭代，获取最终状态----------------------
    def UAV_state(self, T, n, phi, deltat):
        """
        :param T: 无人机群的动力，单位：N
        :param n: 无人机群的负载，单位，N
        :param phi: 俯仰角，单位：弧度
        :param deltat: 编队终端时间，单位，s
        :return: 无人机群不同时间段的位置x，y，z，速度v，距离d，自由终端约束f
        """
        g = 9.8                                                    # 重力加速度

        T = np.reshape(T, (self.N,self.ns))
        n = np.reshape(n, (self.N, self.ns))
        phi = np.reshape(phi, (self.N, self.ns))

        for i in range(self.ns):
            for j in range(self.N):
                self.x0[j] = self.x0[j] + self.v0[j] * np.cos(self.gamma0[j]) * np.cos(self.ka0[j]) * deltat / 1000
                self.y0[j] = self.y0[j] + self.v0[j] * np.cos(self.gamma0[j]) * np.sin(self.ka0[j]) * deltat / 1000
                self.z0[j] = self.z0[j] - self.v0[j] * np.sin(self.gamma0[j]) * deltat / 1000
                self.v0[j] = self.v0[j] + g * ((T[j, i] - self.D0[j]) / self.W - np.sin(self.gamma0[j]))* deltat
                self.D0[j] = self.D0[j] + 1/2 * 0.08 * 1.29 * (self.v0[j] ** 2) * 20 * deltat
                self.gamma0[j] = self.gamma0[j] + g / self.v0[j] * (n[j, i] * np.cos(phi[j, i]) - np.cos(self.ka0[j]))
                self.ka0[j] = self.ka0[j] + g * n[j, i] * np.sin(phi[j, i]) / self.v0[j] / np.cos(self.gamma0[j]) * deltat
            self.x[:, i] = self.x0
            self.y[:, i] = self.y0
            self.z[:, i] = self.z0
            self.v[:, i] = self.v0
            self.D[:, i] = self.D0
            self.gamma[:, i] = self.gamma0
            self.ka[:, i] = self.ka0

    # ----------------------计算无人机群编队的适应度值----------------------
    def UAV_fitness(self, deltat):

        # 选择编号为3的无人机作为中心无人机
        xcenter = self.x[3, -1]
        ycenter = self.y[3, -1]
        zcenter = self.z[3, -1]

        # 自由终端约束
        f = 0
        for i in range(self.N):
            f = f + (self.x[i, -1] - xcenter - self.xe[i]) ** 2 + (self.y[i, -1] - ycenter - self.ye[i]) ** 2  \
                + (self.z[i, -1] - zcenter - self.ze[i]) ** 2


        # 计算无人机间的距离及惩罚值
        de = 0
        d = np.zeros((self.N, self.N, self.ns))
        for k in range(self.ns):
            for i in range(self.N - 1):
                for j in range(1, self.N):
                    d[i, j , k] = self.distance(self.x[i,k], self.y[i,k], self.y[i,k],
                                                self.x[j,k], self.y[j,k], self.z[j,k])
                    d[j, i, k] =  copy.deepcopy(d[i, j, k])
                    de = de + max(0, self.Dsafe - d[i, j , k]) + max(0,  d[i, j , k] - self.Dcomm)

        df = 0
        for i in range(self.N):
            for j in range(self.ns):
                df = df + max(0, self.Vmin - self.v[i, j])  + max(0, self.v[i, j] - self.Vmax) + \
                          max(0, self.gamma_min - self.gamma[i, j]) + max(0, self.gamma[i, j] - self.gamma_max) + \
                          max(0, self.ka_min - self.ka[i, j]) + max(0, self.ka[i, j] - self.ka_max)

        # 适应度值
        J = self.ns * deltat + self.sigma * (f + de + df)
        return J, self.x, self.y, self.z, self.v, self.D, self.gamma, self.ka

# ------------------程序运行------------------
def main():
    T = random.rand(5,5) * (1e4 - 0) + 0
    n = random.rand(5, 5)* (200+200) - 200
    phi = random.rand(5, 5)
    My_UAV = UAV(N=5, ns=5, Dsafe=5, Dcomm=45, sigma=1.8e7)
    My_UAV.init_state()
    My_UAV.UAV_state(T, n, phi, 1)
    J = My_UAV.UAV_fitness(1)
    print('适应度值：%e' % J[0])

if __name__ == '__main__':
    main()