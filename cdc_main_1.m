
clear all 
clc
clf 
format short

p = [0.1;0.05;0.05;0.15;0.2;0.05;0.03;0.07;0.25;0.05];
q = [0.05;0.1;0.05;0.2;0.07;0.15;0.05;0.25;0.03;0.05];

Q = [1  1  1  0  1  0  1  0  1  1;
     1  0  1  0  0  0  1  0  0  1;
     1  1  0  0  1  0  1  1  0  0;
     0  0  0  0  1  0  0  1  0  0;
     1  0  1  1  1  1  0  0  0  0;
     0  0  0  0  1  0  1  0  1  0;
     1  1  1  0  0  1  1  0  0  0;
     0  0  1  1  0  0  0  0  1  1;
     1  0  0  0  0  1  0  1  0  1;
     1  1  0  0  0  0  0  1  1  0];
% 
Q1 = [1 -1  1  0  1  0 1  0  1  1;
     -1  0  1  0  0  0  1  0  0  1;
      1  1  0  0  1  0  1  1  0  0;
      0  0  0  0  1  0  0  1  0  0;
      1  0  1  1  1  1  0  0  0  0;
      0  0  0  0  1  0  1  0  1  0;
      1   1  1  0  0  1  1  0  0  0;
      0  0  1  1  0  0  0  0  1  1;
      1  0  0  0  0  1  0  1  0  1;
      1  1  0  0  0  0  0  1  1  0];

Q2 = [1  1  -1  0  1  0  1  0  -1  1;
      1  0  1  0  0  0  1  0  0  1;
      -1  1  0  0  1  0  1  1  0  0;
      0  0  0  0  1  0  0  1  0  0;
      1  0  1  1  1  1  0  0  0  0;
      0  0  0  0  1  0  1  0  1  0;
      1  1  1  0  0  1  1  0  0  0;
      0  0  1  1  0  0  0  0  1  1;
      -1  0  0  0  0  1  0  1  0  -1;
      1  1  0  0  0  0  0  1  -1  0];

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% p = [0.2,0.3,0.1,0.4]';
% q = [0.1,0.1,0.4,0.4]';
% % 
% Q = [0,1,1,1; 
%      1,0,1,1; 
%      1,1,0,1; 
%      1,1,1,0];
% 
% Q1 = [0,-1,1,1; 
%       -1,0,1,1; 
%       1,1,0,1; 
%       1,1,1,0];
% 
% Q2 = [0,1,-1,1; 
%       1,0,1,1; 
%      -1,1,0,1; 
%       1,1,1,0];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% p = [0.2,0.3,0.5]';
% q = [0.2,0.3,0.5]';
% % 
% Q = [1,1,0; 
%      1,1,1; 
%      1,1,1];
% 
% Q1 = [1,-1,0; 
%       1,0,-1; 
%       -1,1,1];
% 
% Q2 = [1,-1,0; 
%       1,0,-1; 
%       -1,1,1];



X = sign(Q1);
bX = abs(Q1);
X_pos = (bX + Q1)./2;
X_neg = (bX - Q1)./2;

Y = sign(Q2);
bY = abs(Q2);
Y_pos = (bX + Q2)./2;
Y_neg = (bX - Q2)./2;

% Initialization
[n,~] = size(Q1);
alpha1 = ones(n,1);
alpha2 = ones(n,1);

T = 100;
cost = [];

Err_1 = [];
Err_2 = [];

Cpp = ((X_pos + Y_pos) == 2).*Q.*exp(-1);
Cmm = ((X_neg + Y_neg) == 2).*Q.*exp(-1);
Cpm = ((X_pos + Y_neg) == 2).*Q.*exp(-1);
Cmp = ((X_neg + Y_pos) == 2).*Q.*exp(-1);

%% Iteration
for t = 1:T

%%%%%%%%%%%%%%%%%%
% update of alpha1 - p - X
a_1 = Cpp'*alpha2 + Cpm'*(1./alpha2);
b_1 = Cmp'*alpha2 + Cmm'*(1./alpha2);

% location indicator
alpha1_neg_loc = (b_1~=0);
alpha1_pos_loc = ~alpha1_neg_loc;
% two cases 
alpha1_var = zeros(n,1);
alpha1_var(alpha1_neg_loc) = Gsolver(a_1(alpha1_neg_loc),b_1(alpha1_neg_loc),p(alpha1_neg_loc));
alpha1_var(alpha1_pos_loc) = p(alpha1_pos_loc)./a_1(alpha1_pos_loc);
alpha1 = alpha1_var;


%%%%%
bP = diag(alpha1)*Cpp*diag(alpha2) + diag(1./alpha1)*Cmp*diag(alpha2) + diag(alpha1)*Cpm*diag(1./alpha2) + diag(1./alpha1)*Cmm*diag(1./alpha2);
Err_1(end+1) = norm(sum(Y.*bP,1)'-q)/norm(q);

%%
%%%%%%%%%%%%%%%%%%
% update of alpha2 - q - Y
a_2 = Cpp*alpha1 + Cmp*(1./alpha1);
b_2 = Cpm*alpha1 + Cmm*(1./alpha1);

% location indicator
alpha2_neg_loc = (b_2~=0);
alpha2_pos_loc = ~alpha2_neg_loc;

% two cases 
alpha2_var = zeros(n,1);
alpha2_var(alpha2_neg_loc) = Gsolver(a_2(alpha2_neg_loc),b_2(alpha2_neg_loc),q(alpha2_neg_loc));
alpha2_var(alpha2_pos_loc) = q(alpha2_pos_loc)./a_2(alpha2_pos_loc);
alpha2 = alpha2_var;

%%
% convergence 
bP = diag(alpha1)*Cpp*diag(alpha2) + diag(1./alpha1)*Cmp*diag(alpha2) + diag(alpha1)*Cpm*diag(1./alpha2) + diag(1./alpha1)*Cmm*diag(1./alpha2);

obj = bP.*log(bP);
obj(isnan(obj)) = 0;
obj_value = sum(real(obj),'all');
cost(end+1) = obj_value;

% Error storage
Err_2(end+1) = norm(sum(X.*bP,2)-p)/norm(p);
end


%% Result checking -- prelimaries
disp('objective value:')
sum(obj_value,'all')

disp('final error 1:')
disp(sum(Y.*bP,1)'-q)

disp('final error 2:')
disp(sum(X.*bP,2)-p)
%%
figure(1);
subplot(3,1,1)
plot(1:T,cost,'Color',[0 0.4470 0.7410],'LineWidth',2);
axis tight;
ylabel('Objective') 
title('Convergence')

subplot(3,1,2)
plot(1:T,log10(Err_1),'Color',[0.4660 0.6740 0.1880],'LineWidth',2);
ylabel('Marginal 1') 
axis tight; 
title('log|| (YP)^T 1 - q||');

subplot(3,1,3)
plot(1:T,log(Err_2),'Color',[0.6350 0.0780 0.1840],'LineWidth',2);
ylabel('Marginal 2') 
xlabel('Iteration')  
axis tight; 
title('log|| (XP) 1 - p||');

%%

figure(2)

pos_point = X.*bP;
pos_point(pos_point<0) = 0;

neg_point = X.*bP;
neg_point(neg_point>0) = 0;


list1 = find (pos_point>0);
[I1,J1] = ind2sub(size(pos_point),list1);

list2 = find (neg_point<0);
[I2,J2] = ind2sub(size(neg_point),list2);

c_1 = [0.1;0.05;0.05;0.15;0.2;0.05;0.03;0.07;0.25;0.05];
c_2 = [0.05;0.1;0.05;0.2;0.07;0.15;0.05;0.25;0.03;0.05];


hold on

for i=1:n
     marginal1 = plot(i,-1,'o','MarkerSize',100.*c_1(i),'LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
end

for i=1:n
     marginal2 = plot(-1,i,'o','MarkerSize',100.*c_2(i),'LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
end


for i=1:length(I1)
     pt_pos = plot(I1(i),J1(i),'o','MarkerSize',...
     300*pos_point(I1(i),J1(i)),'MarkerFaceColor','b');
end

for i=1:length(I2)
     pt_neg = plot(I2(i),J2(i),'o','MarkerSize',...
     300*abs(neg_point(I2(i),J2(i))),'MarkerFaceColor','r');
end

hold off

grid on
xticks([1:10])
yticks([1:10])
xlim([-2,10])
ylim([-2,10])
set(gca,"FontSize",20)
% xlim([-1 10])
% ylim([-1 10])
% axis off

%%
figure(3)

pos_point = Y.*bP;
pos_point(pos_point<0) = 0;

neg_point = Y.*bP;
neg_point(neg_point>0) = 0;


list1 = find (pos_point>0);
[I1,J1] = ind2sub(size(pos_point),list1);

list2 = find (neg_point<0);
[I2,J2] = ind2sub(size(neg_point),list2);

c_1 = [0.1;0.05;0.05;0.15;0.2;0.05;0.03;0.07;0.25;0.05];
c_2 = [0.05;0.1;0.05;0.2;0.07;0.15;0.05;0.25;0.03;0.05];


hold on

for i=1:n
     marginal1 = plot(i,-1,'o','MarkerSize',100.*c_1(i),'LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
end

for i=1:n
     marginal2 = plot(-1,i,'o','MarkerSize',100.*c_2(i),'LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
end


for i=1:length(I1)
     pt_pos = plot(I1(i),J1(i),'o','MarkerSize',...
     300*pos_point(I1(i),J1(i)),'MarkerFaceColor','b');
end

for i=1:length(I2)
     pt_neg = plot(I2(i),J2(i),'o','MarkerSize',...
     300*abs(neg_point(I2(i),J2(i))),'MarkerFaceColor','r');
end

hold off

grid on
xticks([1:10])
yticks([1:10])
xlim([-2,10])
ylim([-2,10])
set(gca,"FontSize",20)

%%

figure(4)

Q = Q./sum(Q,'all');

Qpos_point = Q;

Qpos_point(Qpos_point<0) = 0;

Qneg_point = Q;
Qneg_point(Qneg_point>0) = 0;


list1 = find (Qpos_point>0);
[I1,J1] = ind2sub(size(Qpos_point),list1);

list2 = find (Qneg_point<0);
[I2,J2] = ind2sub(size(Qneg_point),list2);

q_1 = sum(Q,2);
q_2 = sum(Q,1)';


hold on

for i=1:n
    marginal1 = plot(i,-1,'o','MarkerSize',100.*q_1(i),'LineWidth',1.5,'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
end

for i=1:n
     marginal2 = plot(-1,i,'o','MarkerSize',100.*q_2(i),'LineWidth',1.5,'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
end


for i=1:length(I1)
     pt_pos = plot(I1(i),J1(i),'o','MarkerSize',...
     400*Qpos_point(I1(i),J1(i)),'MarkerFaceColor','b');
end

for i=1:length(I2)
     pt_neg = plot(I2(i),J2(i),'o','MarkerSize',...
     400*abs(Qneg_point(I2(i),J2(i))),'MarkerFaceColor','r');
end

hold off

grid on
xticks([1:10])
yticks([1:10])
xlim([-2,10])
ylim([-2,10])
set(gca,"FontSize",20)


%%
function x = Gsolver(a,b,c) 
x = (c + sqrt(c.^2 + 4.*a.*b))./(2.*a);
end