%%%%%%%%%%%%%%%%%%%%标准遗传算法求函数极值%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%初始化参数%%%%%%%%%%%%%%%%%%%%
clear all;                                               %清除所有变量
close all;                                               %清图
clc;                                                     %清屏
Np = 100;                                                %种群数量

Pc = 0.8;                                                %交叉率
Pm = 0.1;                                                %变异率
G = 200;                                                 %最大遗传代数
Dim = 10;                                                %维度
Xs = 20;                                                 %上限
Xx = -20;                                                %下限
x = rand(Np,Dim) * (Xs - Xx) + Xx;                             %随机获得初始种群
%%%%%%%%%%%%%%%%%%%%遗传算法循环%%%%%%%%%%%%%%%%%%%%
trace = ones(1,G);                                       %记录每次迭代的最优个体
                                       
Fit = ones(1,Np);
deltap = zeros(Np,Dim);                                  %迭代间的适应度值的差
sp = zeros(Np,Dim);                                      %进化趋势
rou = 1.45;                                              %学习速率
beata = 0.85;                                            %惯性常数
acc = zeros(1,Dim);
for k =1:G             
   %%%%%%%%%%%%%%%%%%%%将二进制解码为定义域范围内十进制%%%%%%%%%%%%%%%%%%%%
   for i = 1:Np
       Fit(i) = func(x(i,:));                            %适应度值
   end
   maxFit = max(Fit);                                    %适应度的最大值
   minFit = min(Fit);                                    %适应度的最小值
   rr = find(Fit == maxFit);                             %得到适应度值最小的索引
   xBest = x(rr(1),:);                                   %历代最优个体的十进制
   Fit = (Fit - minFit) / (maxFit - minFit);             %归一化适应度值
   nf = zeros(Np,Dim);                                   %初始化新种群
   %%%%%%%%%%%%%%%%%%%%基于轮盘赌的复制操作%%%%%%%%%%%%%%%%%%%%
   sum_Fit = sum(Fit);
   fitvalue = Fit./sum_Fit;                              %依适应度的概率值
   fitvalue = cumsum(fitvalue);                          %累加和
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
   %%%%%%%%%%%%%%%%%%%%基于概率的交叉操作%%%%%%%%%%%%%%%%%%%%
   for i = 1:Dim
      for j = 1:2:Np
          p = rand;
          if p < Pc                                          %如果随机数p小于交叉概率，则进行交叉
             nf(j+1,i) = Pc*nf(j+1,i) + (1-Pc)*nf(j,i);
             nf(j,i) = Pc*nf(j,i) + (1-Pc)*nf(j+1,i);      
          end
      end
   end
   
   %%%%%%%%%%%%%%%%%%%%基于概率的变异操作%%%%%%%%%%%%%%%%%%%%
   
   
   
   
   for j = 1:Dim
      for i = 1:Np
         if(rand < Pm)
%             mpoint = round(rand*Dim);                         %变异的基因位置 
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
   x(1,:) = xBest;                                       %保留最优个体在新种群中
   trace(k) = -maxFit;                                   %历代最优适应度
end
%%%%%%%%%%%%%%%%%%%%画图%%%%%%%%%%%%%%%%%%%%
xBest                                                    %最优个体
trace(end)
figure;
plot(trace);
xlabel('迭代次数');
ylabel('目标函数值');
title('适应度进化曲线');