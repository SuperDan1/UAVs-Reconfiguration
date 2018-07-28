%%%%%%%%%%��Ӧ�Ⱥ���%%%%%%%%%%
function fitness = UAV_fitness(T,n,phi,deltat,x,y,z,v,D,W,gamma,chi)
%�����ն�ʱ��Tʱ��״̬
N = 5;
np = 5;                                    %���ն�ʱ��T��ɢ��5������
Dsafe = 5;                                                 %��ȫ����
Dcomm = 45;                                                %ͨ�ž���
sigma = 1.8e7;                                            %�ͷ�ϵ��

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


