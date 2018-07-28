%%%%%%%%%%%%%%%%%%%%��׼�Ŵ��㷨������ֵ%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%��ʼ������%%%%%%%%%%%%%%%%%%%%
clear all;                                               %������б���
close all;                                               %��ͼ
clc;                                                     %����
Np = 100;                                                %��Ⱥ����

Pc = 0.8;                                                %������
Pm = 0.1;                                                %������
G = 200;                                                 %����Ŵ�����
Dim = 10;                                                %ά��
Xs = 20;                                                 %����
Xx = -20;                                                %����
x = rand(Np,Dim) * (Xs - Xx) + Xx;                             %�����ó�ʼ��Ⱥ
%%%%%%%%%%%%%%%%%%%%�Ŵ��㷨ѭ��%%%%%%%%%%%%%%%%%%%%
trace = ones(1,G);                                       %��¼ÿ�ε��������Ÿ���
                                       
Fit = ones(1,Np);
deltap = zeros(Np,Dim);                                  %���������Ӧ��ֵ�Ĳ�
sp = zeros(Np,Dim);                                      %��������
rou = 1.45;                                              %ѧϰ����
beata = 0.85;                                            %���Գ���
acc = zeros(1,Dim);
for k =1:G             
   %%%%%%%%%%%%%%%%%%%%�������ƽ���Ϊ������Χ��ʮ����%%%%%%%%%%%%%%%%%%%%
   for i = 1:Np
       Fit(i) = func(x(i,:));                            %��Ӧ��ֵ
   end
   maxFit = max(Fit);                                    %��Ӧ�ȵ����ֵ
   minFit = min(Fit);                                    %��Ӧ�ȵ���Сֵ
   rr = find(Fit == maxFit);                             %�õ���Ӧ��ֵ��С������
   xBest = x(rr(1),:);                                   %�������Ÿ����ʮ����
   Fit = (Fit - minFit) / (maxFit - minFit);             %��һ����Ӧ��ֵ
   nf = zeros(Np,Dim);                                   %��ʼ������Ⱥ
   %%%%%%%%%%%%%%%%%%%%�������̶ĵĸ��Ʋ���%%%%%%%%%%%%%%%%%%%%
   sum_Fit = sum(Fit);
   fitvalue = Fit./sum_Fit;                              %����Ӧ�ȵĸ���ֵ
   fitvalue = cumsum(fitvalue);                          %�ۼӺ�
   ms = sort(rand(Np,1));
   fiti = 1;
   newi = 1;
   while newi <= Np
      if(ms(newi) < fitvalue(fiti)) 
         nf(newi,:) =x(fiti,:);
         newi = newi + 1; 
      else
          fiti = fiti + 1;
      end
   end
   %%%%%%%%%%%%%%%%%%%%���ڸ��ʵĽ������%%%%%%%%%%%%%%%%%%%%
   for i = 1:Dim
      for j = 1:2:Np
          p = rand;
          if p < Pc                                          %��������pС�ڽ�����ʣ�����н���
             nf(j+1,i) = Pc*nf(j+1,i) + (1-Pc)*nf(j,i);
             nf(j,i) = Pc*nf(j,i) + (1-Pc)*nf(j+1,i);      
          end
      end
   end
   
   %%%%%%%%%%%%%%%%%%%%���ڸ��ʵı������%%%%%%%%%%%%%%%%%%%%
   
   
   
   
   for j = 1:Dim
      for i = 1:Np
         if(rand < Pm)
%             mpoint = round(rand*Dim);                         %����Ļ���λ�� 
%             if mpoint <= 0
%                mpoint = 1; 
%             end
%             nf(i,j) = nf(i,j) + beata*deltap(i,j) + rou*sp(i,j);
%             deltap(i,j) = (xBest(j) - nf(i,j)) * abs(randn());
%             if func(nf(i,:)) > Fit(i)
%                acc(j) = 1;
%             else
%                 acc(j) = 0;
%             end
%             sp(i,j) = beata*acc(j)*deltap(i,j)+2*sp(i,j);
              temp = randi([0,10],1)/2;
              if temp ==0
                 nf(i,j) = nf(i,j) + 0.8*(xBest(j) - nf(i,j))*abs(randn()); 
              else
                 nf(i,j) = nf(i,j) - 0.8*(nf(i,j)-xBest(j))*abs(randn()); 
              end
         end
      end
   end
   
   x = nf;
   x(1,:) = xBest;                                       %�������Ÿ���������Ⱥ��
   trace(k) = -maxFit;                                   %����������Ӧ��
end
%%%%%%%%%%%%%%%%%%%%��ͼ%%%%%%%%%%%%%%%%%%%%
xBest                                                    %���Ÿ���
trace(end)
figure;
plot(trace);
xlabel('��������');
ylabel('Ŀ�꺯��ֵ');
title('��Ӧ�Ƚ�������');