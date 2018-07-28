%%%%%%%%%%%%%%%%%%%%%����Ⱥ�㷨������ֵ%%%%%%%%%%%%%%%%%%%%%
%%
close all;                                                    %��ͼ
clc;                                                          %����
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%��ʼ��%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%���˻�״̬��ʼ��%%%%%%%%%%
N = 5;                                             %5�����˻����
np = 5;                                            %���ն�ʱ��T��ɢ��5������

Dsafe = 5;                                                 %��ȫ����
Dcomm = 45;                                                %ͨ�ž���
Lmin = -20;                                        %x,y,z�������Сֵ
Lmax = 20;                                         %x,y,z��������ֵ

x0 = rand(N,1) * (Lmax - Lmin) + Lmin;
y0 = rand(N,1) * (Lmax - Lmin) + Lmin;

phimin = -pi;
phimax = pi;
phi = rand(N,1) * (phimax - phimin) + phimin;   %������

                                        %������


% x0 = zeros(1,N);
% y0 = zeros(1,N);
% v0 = zeros(N,1);
% gamma = zeros(N,1);
% chi = zeros(N,1);
% D = 1/2*0.08*1.29*(v0/3.6).^2*30;             %��������
% W=2000;                                             %���˻�����
%%%%%%%%%%PSO�㷨��������%%%%%%%%%%
Np = 400;                                           %Ⱥ�����Ӹ���
Tp = 600;                                          %����������

Vmax = 4.4;                                        %���˻�����ٶ�
Vmin = 0;                                          %���˻�������С�ٶ�
v = rand(Np,N*np) * (Vmax - Vmin) + Vmin;             %N�����˻����ٶ�

gamma_max = 53;
gamma_min = -53;
gamma = rand(Np,N*np) * (gamma_max - gamma_min) + gamma_min; 

deltat_max = 100;
deltat_min = 0;
deltat = rand(Np,1) * (gamma_max - gamma_min) + gamma_min; 

c1 = 2;                                           %ѧϰ����1
c2 = 2;                                            %ѧϰ����2
%w = 0.8;                                            %����Ȩ��
Xpmax = [Vmax*ones(1,N*np),gamma_max*ones(1,N*np),deltat_max];
Xpmin = [Vmin*ones(1,N*np),gamma_min*ones(1,N*np),deltat_min];
Vpmax = (Xpmax-Xpmin);                                        %�ٶ����ֵ
% Vpmax = 0.01*ones(1,76);
Vpmin = -Vpmax;                                       %�ٶ���Сֵ
Wmax = 0.8;
Wmin = 0.4;
%%
%%%%%%%%%%%%%%%%%%%%%��ʼ����Ⱥ���壨����λ�ú��ٶȣ�%%%%%%%%%%%%%%%%%%%%%
xp = [v,gamma,deltat];
[Xim,Dim] = size(xp);

vp = zeros(Np,Dim);
for i = 1:Dim
    vp(:,i) = rand(Np,1) * (Vpmax(i) - Vpmin(i)) + Vpmin(i);                    %��������ٶ�
end
%%
%%%%%%%%%%%%%%%%%%%%%��ʼ����������λ�ú�����ֵ%%%%%%%%%%%%%%%%%%%%%
p = xp;                                                   %�������ڵ�λ��
pbest = ones(Np,1);                                       %��ʼ����������ֵ
for i = 1:Np
    pbest(i) = UAV_fitness(xp(i,1:25), xp(i,26:50),xp(end),x0,y0,phi);
end
%%
%%%%%%%%%%%%%%%%%%%%%��ʼ��ȫ������λ�ú�����ֵ%%%%%%%%%%%%%%%%%%%%%
g = ones(1,N*np+1);                                            %ȫ������λ��
gbest = inf;
for i = 1:Np
    if(pbest(i) < gbest)
        g = p(i,:);
        gbest = pbest(i);
    end
end
gb = ones(1,Tp);                                           %����ÿ�ε���������λ��
%%
%%%%%%%%%%%%%%%%%%���չ�ʽ���ε���ֱ�����㾫�Ȼ��ߵ�������%%%%%%%%%%%%%%%%%%
for i = 1:Tp
    for j = 1:Np
        %%%%%%%%%%���¸�������λ�ú�����ֵ%%%%%%%%%%
        if(UAV_fitness(xp(j,1:25), xp(j,26:50),xp(end),x0,y0,phi) < pbest(j))
            p(j,:) = xp(j,:);
            pbest(j) = UAV_fitness(xp(j,1:25), xp(j,26:50),xp(end),x0,y0,phi);
        end
        %%%%%%%%%%����ȫ������λ�ú�����ֵ%%%%%%%%%%
        if(pbest(j) < gbest)
            g = p(j,:);
            gbest = pbest(j);
        end
        w = Wmax - (Wmax-Wmin)*i/Tp;
        %%%%%%%%%%����λ�ú��ٶ�%%%%%%%%%%
        vp(j,:) = w*vp(j,:) + c1*rand*(p(j,:) - xp(j,:)) + c2*rand*(g - xp(j,:));
        xp(j,:) = xp(j,:) + vp(j,:);
        %%%%%%%%%%�߽���������%%%%%%%%%%
        for k = 1:Dim
            if (vp(j,k) > Vpmax(k)) || (vp(j,k) < Vpmin(k))
                vp(j,k) = rand*(Vpmax(k) - Vpmin(k)) + Vpmin(k);
            end
            if (xp(j,k) > Xpmax(k)) || (xp(j,k) < Xpmin(k))
                xp(j,k) = rand*(Xpmax(k) - Xpmin(k)) + Xpmin(k);
            end
        end
    end
    %%%%%%%%%%��¼����ȫ������ֵ%%%%%%%%%%
    gb(i) = gbest;
end
%%
%��ͼ
g                                                       %���Ÿ���
gb(end)                                                  %����ֵ

[ x,y,phi0] = get_state(g(1:25), g(26:50), g(end),x0,y0,phi);
fitness = UAV_fitness(g(1:25), g(26:50), g(end),x0,y0,phi);
%for i = 1:N
%    plot3(x(i,:),y(i,:),z(i,:),'-','linewidth',2);
%    hold on;
%end

figure(2);
plot(x(:,end),y(:,end),'--*','markersize',5);
text(x(3,end),y(3,end),'�������˻�','FontSize',10)
grid on;


figure(5);
plot(gb);
grid on;
xlabel('��������')
ylabel('��Ӧ��ֵ')
title('��Ӧ�Ƚ�������')