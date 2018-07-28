function [ x,y,z,d,f] = get_state( T,n,phi,deltat,x0,y0,z0,v0,D,W,gamma,chi)

N = 5;
np = 5;                                    %���ն�ʱ��T��ɢ��5������
g = 9.8;
d = zeros(N,N,np);
T = reshape(T,N,np);

n = reshape(n,N,np);

phi = reshape(phi,N,np);

v0 = v0./3.6;
sigma = 1.8e7;                                            %�ͷ�ϵ��
x = zeros(N,np);
y = zeros(N,np);
z = zeros(N,np);
v = zeros(N,np);
for i = 1:np 
    for j = 1:N
        x0(j) = x0(j)+v0(j) * cos(gamma(j)) * cos(chi(j)) * deltat/1000;
        y0(j) = y0(j)+v0(j) * cos(gamma(j)) * sin(chi(j)) * deltat/1000;
        z0(j) = z0(j)-v0(j) * sin(gamma(j)) * deltat/1000;
        v0(j) = v0(j) + g * (((T(j,i) - D(j))/W)-sin(gamma(j))) * deltat;
        D(j) = 1/2*0.08*1.29*(v0(j)).^2*30;
        gamma(j) = gamma(j) + (g/v0(j)) * (n(j,i) * cos(phi(j,i)) - cos(gamma(j))) * deltat;
        chi(j) = chi(j) + g * n(j,i) * sin(phi(j,i)) / v0(j) / cos(gamma(j)) * deltat;
    end
    v(:,i) = v0;
    x(:,i) = x0;
    y(:,i) = y0;
    z(:,i) = z0;
end

%���˻�֮��ľ���
for k = 1:np
    for i = 1:N
        for j = (i+1):N
            d(i,j,k) = distance(x(i,k),y(i,k),z(i,k),x(j,k),y(j,k),z(j,k));
            d(j,i,k) = d(i,j,k);
        end
    end
end

%ѡ����Ϊ3�����˻���Ϊ�������˻�
xcenter = x(3,end);
ycenter = y(3,end);
zcenter = z(3,end);

%�ն�Tʱ�̱���ڵ�i�����˻�������������˻��������������ֵ
xe = [-20, -10, 0, -10, -20];
ye = [20, 10, 0, -10, -20]; 
ze = [0,0,0,0,0];



%�����ն�Լ��
f = 0;
for i = 1:N
%     f = f + sigma*(x(i,end) - xcenter - xe(i))^2 +sigma*(y(i,end) - ycenter - ye(i))^2 +1.8e4*(z(i,end) - zcenter - ze(i))^2;
    f = f + sigma*(x(i,end) - xcenter - xe(i))^2 +sigma*(y(i,end) - ycenter - ye(i))^2;
end
end

function d = distance( x1,y1,z1,x2,y2,z2 )
%UNTITLED3 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
d = sqrt((x1 - x2)^2 +(y1 - y2)^2 + (z1 - z2)^2); 
end

