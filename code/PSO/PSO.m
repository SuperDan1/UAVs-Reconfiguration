%%%%%%%%%%%%%%%%%%%%%粒子群算法求函数极值%%%%%%%%%%%%%%%%%%%%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%初始化%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;                                                %清除所有变量
close all;                                                %清图
clc;                                                      %清屏
N = 100;                                                  %群体粒子个数
D = 10;                                                   %粒子维数
T = 200;                                                  %最大迭代次数
c1 = 1.5;                                                 %学习因子1
c2 =1.5;                                                  %学习因子2
w = 0.8;                                                  %惯性权重
Xmax = 20;                                                %位置最大值
Xmin = -20;                                               %位置最小值
Vmax = 10;                                                %速度最大值
Vmin = -10;                                               %速度最小值
%%
%%%%%%%%%%%%%%%%%%%%%初始化种群个体（限制位置和速度）%%%%%%%%%%%%%%%%%%%%%
x = rand(N,D) * (Xmax - Xmin) + Xmin;                     %随机产生位置
v = rand(N,D) * (Vmax - Vmin) + Vmin;                     %随机产生速度
%%
%%%%%%%%%%%%%%%%%%%%%初始化个体最优位置和最优值%%%%%%%%%%%%%%%%%%%%%
p = x;                                                    %粒子现在的位置
pbest = ones(N,1);                                        %初始化个体最优值
for i = 1:N
    pbest(i) = func(x(i,:));
end
%%
%%%%%%%%%%%%%%%%%%%%%初始化全局最优位置和最优值%%%%%%%%%%%%%%%%%%%%%
g = ones(1,D);                                            %全局最优位置
gbest = inf;
for i = 1:N
    if(pbest(i) < gbest)
        g = p(i,:);
        gbest = pbest(i);
    end
end
gb = ones(1,T);                                           %保存每次迭代的最优位置
%%
%%%%%%%%%%%%%%%%%%按照公式依次迭代直到满足精度或者迭代次数%%%%%%%%%%%%%%%%%%
for i = 1:T
    for j = 1:N
        %%%%%%%%%%更新个体最优位置和最优值%%%%%%%%%%
        if(func(x(j,:)) < pbest(j))
            p(j,:) = x(j,:);
            pbest(j) = func(x(j,:));
        end
        %%%%%%%%%%更新全局最优位置和最优值%%%%%%%%%%
        if(pbest(j) < gbest)
            g = p(j,:);
            gbest = pbest(j);
        end
        %%%%%%%%%%更新位置和速度%%%%%%%%%%
        v(j,:) = w*v(j,:) + c1*rand*(p(j,:) - x(j,:)) + c2*rand*(g - x(j,:));
        x(j,:) = x(j,:) + v(j,:);
        %%%%%%%%%%边界条件处理%%%%%%%%%%
        for ii = 1:D
            if (v(j,ii) > Vmax) || (x(j,ii) < Vmin)
                v(j,ii) = rand*(Vmax - Vmin) + Vmin;
            end
            if (x(j,ii) > Xmax) || (x(j,ii) < Xmin)
                x(j,ii) = rand*(Xmax - Xmin) + Xmin;
            end
        end
    end
    %%%%%%%%%%记录历代全局最优值%%%%%%%%%%
    gb(i) = gbest;
end
g;                                                        %最优个体
gb(end)                                                   %最优值
figure;
plot(gb);
xlabel('迭代次数')
ylabel('适应度值')
title('适应度进化函数')