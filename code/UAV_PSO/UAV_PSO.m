%%%%%%%%%%%%%%%%%%%%%粒子群算法求函数极值%%%%%%%%%%%%%%%%%%%%%
%%
close all;                                                    %清图
clc;                                                          %清屏
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%初始化%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%无人机状态初始化%%%%%%%%%%
N = 5;                                             %5架无人机编队
np = 5;                                            %将终端时间T离散成5个部分

Dsafe = 5;                                                 %安全距离
Dcomm = 45;                                                %通信距离
Lmin = -20;                                        %x,y,z坐标的最小值
Lmax = 20;                                         %x,y,z坐标的最大值

x0 = rand(N,1) * (Lmax - Lmin) + Lmin;
y0 = rand(N,1) * (Lmax - Lmin) + Lmin;

phimin = -pi;
phimax = pi;
phi = rand(N,1) * (phimax - phimin) + phimin;   %俯仰角

                                        %航迹角


% x0 = zeros(1,N);
% y0 = zeros(1,N);
% v0 = zeros(N,1);
% gamma = zeros(N,1);
% chi = zeros(N,1);
% D = 1/2*0.08*1.29*(v0/3.6).^2*30;             %气动阻力
% W=2000;                                             %无人机重量
%%%%%%%%%%PSO算法参数设置%%%%%%%%%%
Np = 400;                                           %群体粒子个数
Tp = 600;                                          %最大迭代次数

Vmax = 4.4;                                        %无人机最大速度
Vmin = 0;                                          %无人机飞行最小速度
v = rand(Np,N*np) * (Vmax - Vmin) + Vmin;             %N架无人机的速度

gamma_max = 53;
gamma_min = -53;
gamma = rand(Np,N*np) * (gamma_max - gamma_min) + gamma_min; 

deltat_max = 100;
deltat_min = 0;
deltat = rand(Np,1) * (gamma_max - gamma_min) + gamma_min; 

c1 = 2;                                           %学习因子1
c2 = 2;                                            %学习因子2
%w = 0.8;                                            %惯性权重
Xpmax = [Vmax*ones(1,N*np),gamma_max*ones(1,N*np),deltat_max];
Xpmin = [Vmin*ones(1,N*np),gamma_min*ones(1,N*np),deltat_min];
Vpmax = (Xpmax-Xpmin);                                        %速度最大值
% Vpmax = 0.01*ones(1,76);
Vpmin = -Vpmax;                                       %速度最小值
Wmax = 0.8;
Wmin = 0.4;
%%
%%%%%%%%%%%%%%%%%%%%%初始化种群个体（限制位置和速度）%%%%%%%%%%%%%%%%%%%%%
xp = [v,gamma,deltat];
[Xim,Dim] = size(xp);

vp = zeros(Np,Dim);
for i = 1:Dim
    vp(:,i) = rand(Np,1) * (Vpmax(i) - Vpmin(i)) + Vpmin(i);                    %随机产生速度
end
%%
%%%%%%%%%%%%%%%%%%%%%初始化个体最优位置和最优值%%%%%%%%%%%%%%%%%%%%%
p = xp;                                                   %粒子现在的位置
pbest = ones(Np,1);                                       %初始化个体最优值
for i = 1:Np
    pbest(i) = UAV_fitness(xp(i,1:25), xp(i,26:50),xp(end),x0,y0,phi);
end
%%
%%%%%%%%%%%%%%%%%%%%%初始化全局最优位置和最优值%%%%%%%%%%%%%%%%%%%%%
g = ones(1,N*np+1);                                            %全局最优位置
gbest = inf;
for i = 1:Np
    if(pbest(i) < gbest)
        g = p(i,:);
        gbest = pbest(i);
    end
end
gb = ones(1,Tp);                                           %保存每次迭代的最优位置
%%
%%%%%%%%%%%%%%%%%%按照公式依次迭代直到满足精度或者迭代次数%%%%%%%%%%%%%%%%%%
for i = 1:Tp
    for j = 1:Np
        %%%%%%%%%%更新个体最优位置和最优值%%%%%%%%%%
        if(UAV_fitness(xp(j,1:25), xp(j,26:50),xp(end),x0,y0,phi) < pbest(j))
            p(j,:) = xp(j,:);
            pbest(j) = UAV_fitness(xp(j,1:25), xp(j,26:50),xp(end),x0,y0,phi);
        end
        %%%%%%%%%%更新全局最优位置和最优值%%%%%%%%%%
        if(pbest(j) < gbest)
            g = p(j,:);
            gbest = pbest(j);
        end
        w = Wmax - (Wmax-Wmin)*i/Tp;
        %%%%%%%%%%更新位置和速度%%%%%%%%%%
        vp(j,:) = w*vp(j,:) + c1*rand*(p(j,:) - xp(j,:)) + c2*rand*(g - xp(j,:));
        xp(j,:) = xp(j,:) + vp(j,:);
        %%%%%%%%%%边界条件处理%%%%%%%%%%
        for k = 1:Dim
            if (vp(j,k) > Vpmax(k)) || (vp(j,k) < Vpmin(k))
                vp(j,k) = rand*(Vpmax(k) - Vpmin(k)) + Vpmin(k);
            end
            if (xp(j,k) > Xpmax(k)) || (xp(j,k) < Xpmin(k))
                xp(j,k) = rand*(Xpmax(k) - Xpmin(k)) + Xpmin(k);
            end
        end
    end
    %%%%%%%%%%记录历代全局最优值%%%%%%%%%%
    gb(i) = gbest;
end
%%
%画图
g                                                       %最优个体
gb(end)                                                  %最优值

[ x,y,phi0] = get_state(g(1:25), g(26:50), g(end),x0,y0,phi);
fitness = UAV_fitness(g(1:25), g(26:50), g(end),x0,y0,phi);
%for i = 1:N
%    plot3(x(i,:),y(i,:),z(i,:),'-','linewidth',2);
%    hold on;
%end

figure(2);
plot(x(:,end),y(:,end),'--*','markersize',5);
text(x(3,end),y(3,end),'中心无人机','FontSize',10)
grid on;


figure(5);
plot(gb);
grid on;
xlabel('迭代次数')
ylabel('适应度值')
title('适应度进化函数')