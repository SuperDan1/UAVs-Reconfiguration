%%%%%%%%%%适应度函数%%%%%%%%%%
function fitness = UAV_fitness(T,n,phi,deltat,x,y,z,v,D,W,gamma,chi)
%计算终端时间T时的状态
N = 5;
np = 5;                                    %将终端时间T离散成5个部分
Dsafe = 5;                                                 %安全距离
Dcomm = 45;                                                %通信距离
sigma = 1.8e7;                                            %惩罚系数

[ ~,~,~,d,f] = get_state( T,n,phi,deltat,x,y,z,v,D,W,gamma,chi);

de = zeros(N,N);
for k = 1:np
    for i = 1:(N-1)
        for j = (i+1):N
            de(i,j) = max(0,Dsafe - d(i,j,np)) + max(0,d(i,j,np)-Dcomm);
        end
    end
end
de = sum(de(:));
Jextend = np*deltat + f + sigma*de;
fitness = Jextend;
end


