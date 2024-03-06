clear all
clc
clf 
addpath('function')



% %% Generate Q+ and Q-
n = 3;

%%
% X_pos_rd(:,:,1) = [1, 1, 0; 1, 0, 1; 1, 0, 1];
% X_pos_rd(:,:,2) = [0, 0, 0; 1, 1, 0; 1, 1, 0];
% X_pos_rd(:,:,3) = [0, 1, 1; 1, 0, 1; 0, 0, 1];
% 
% X_neg_rd(:,:,1) = [0, 0, 1; 0, 0, 0; 0, 0, 0];
% X_neg_rd(:,:,2) = [1, 0, 0; 0, 0, 0; 0, 0, 0];
% X_neg_rd(:,:,3) = [0, 0, 0; 0, 0, 0; 0, 1, 0];
% 
% Y_pos_rd(:,:,1) = [1, 1, 0; 1, 0, 1; 1, 0, 1];
% Y_pos_rd(:,:,2) = [0, 0, 0; 1, 1, 0; 1, 1, 0];
% Y_pos_rd(:,:,3) = [0, 1, 1; 1, 0, 1; 0, 0, 1];
% 
% Y_neg_rd(:,:,1) = [0, 0, 1; 0, 0, 0; 0, 0, 0];
% Y_neg_rd(:,:,2) = [1, 0, 0; 0, 0, 0; 0, 0, 0];
% Y_neg_rd(:,:,3) = [0, 0, 0; 0, 0, 0; 0, 1, 0];
% 
% Z_pos_rd(:,:,1) = [1, 1, 0; 1, 0, 1; 1, 0, 1];
% Z_pos_rd(:,:,2) = [0, 0, 0; 1, 1, 0; 1, 1, 0];
% Z_pos_rd(:,:,3) = [0, 1, 1; 1, 0, 1; 0, 0, 1];
% 
% Z_neg_rd(:,:,1) = [0, 0, 1; 0, 0, 0; 0, 0, 0];
% Z_neg_rd(:,:,2) = [1, 0, 0; 0, 0, 0; 0, 0, 0];
% Z_neg_rd(:,:,3) = [0, 0, 0; 0, 0, 0; 0, 1, 0];

% Q_pos_rd(:,:,1) = [1, 1, 0; 1, 0, 1; 1, 0, 1];
% Q_pos_rd(:,:,2) = [0, 0, 0; 1, 1, 0; 1, 1, 0];
% Q_pos_rd(:,:,3) = [0, 1, 1; 1, 0, 1; 0, 0, 1];

%%
X_pos_rd(:,:,1) = [1, 1, 0; 1, 0, 1; 1, 0, 1];
X_pos_rd(:,:,2) = [0, 0, 0; 1, 0, 0; 1, 1, 0];
X_pos_rd(:,:,3) = [0, 1, 1; 1, 0, 1; 0, 0, 1];

X_neg_rd(:,:,1) = [0, 0, 1; 0, 0, 0; 0, 0, 0];
X_neg_rd(:,:,2) = [1, 0, 0; 0, 1, 0; 0, 0, 0];
X_neg_rd(:,:,3) = [0, 0, 0; 0, 0, 0; 0, 1, 0];


Y_pos_rd(:,:,1) = [0, 1, 1; 0, 0, 1; 1, 0, 1];
Y_pos_rd(:,:,2) = [1, 0, 0; 1, 1, 0; 1, 1, 0];
Y_pos_rd(:,:,3) = [0, 1, 0; 1, 0, 1; 0, 0, 1];

Y_neg_rd(:,:,1) = [1, 0, 0; 1, 0, 0; 0, 0, 0];
Y_neg_rd(:,:,2) = [0, 0, 0; 0, 0, 0; 0, 0, 0];
Y_neg_rd(:,:,3) = [0, 0, 1; 0, 0, 0; 0, 1, 0];


Z_pos_rd(:,:,1) = [1, 1, 1; 1, 0, 1; 1, 0, 1];
Z_pos_rd(:,:,2) = [0, 0, 0; 1, 0, 0; 1, 1, 0];
Z_pos_rd(:,:,3) = [0, 1, 1; 1, 0, 1; 0, 0, 1];

Z_neg_rd(:,:,1) = [0, 0, 1; 0, 0, 0; 0, 0, 0];
Z_neg_rd(:,:,2) = [1, 0, 0; 0, 1, 0; 0, 0, 0];
Z_neg_rd(:,:,3) = [0, 0, 0; 0, 0, 0; 0, 1, 0];


%%
% X_pos_rd(:,:,1) = [1, 1, 1; 1, 1, 1; 1, 1, 1];
% X_pos_rd(:,:,2) = [1, 1, 1; 1, 1, 1; 1, 1, 1];
% X_pos_rd(:,:,3) = [1, 1, 1; 1, 1, 1; 1, 1, 1];
% 
% X_neg_rd(:,:,1) = [0, 0, 0; 0, 0, 0; 0, 0, 1];
% X_neg_rd(:,:,2) = [0, 0, 0; 0, 1, 0; 0, 0, 0];
% X_neg_rd(:,:,3) = [0, 1, 0; 0, 0, 0; 0, 0, 0];
% 
% 
% Y_pos_rd(:,:,1) = [1, 1, 1; 1, 1, 1; 1, 1, 1];
% Y_pos_rd(:,:,2) = [1, 1, 1; 1, 1, 1; 1, 1, 1];
% Y_pos_rd(:,:,3) = [1, 1, 1; 1, 1, 1; 1, 1, 1];
% 
% Y_neg_rd(:,:,1) = [1, 0, 0; 0, 0, 0; 0, 0, 0];
% Y_neg_rd(:,:,2) = [0, 0, 0; 0, 1, 0; 0, 0, 0];
% Y_neg_rd(:,:,3) = [0, 0, 0; 0, 0, 0; 0, 0, 1];
% 
% 
% Z_pos_rd(:,:,1) = [1, 1, 1; 1, 1, 1; 1, 1, 1];
% Z_pos_rd(:,:,2) = [1, 1, 1; 1, 1, 1; 1, 1, 1];
% Z_pos_rd(:,:,3) = [1, 1, 1; 1, 1, 1; 1, 1, 1];
% 
% Z_neg_rd(:,:,1) = [0, 1, 0; 0, 0, 0; 0, 0, 0];
% Z_neg_rd(:,:,2) = [0, 0, 0; 0, 0, 1; 0, 0, 0];
% Z_neg_rd(:,:,3) = [0, 0, 0; 0, 0, 0; 1, 0, 0];


% c_1 = [0.2;0.3;0.5];
% c_2 = [0.4;0.4;0.2];
% c_3 = [0.1;0.6;0.3];

c_1 = [0.5;0.3;0.2];
c_2 = [0.4;0.4;0.2];
c_3 = [0.2;0.6;0.2];

% % %%
% n = 10;   % works for the 3-D case
% x1 = 1;  % index for generating posotive elements 
% x2 = 0.2; % index for generating negative elements
% 
% 
% Q_pos_rd = Q_generator(n,x1);
% Q_neg_rd = Q_neg_generator(n,x2);


% Three marginals
% x = (0:n-1)'/n-1;
% 
% Gaussian = @(x,t0,sigma)exp( -(x-t0).^2/(2*sigma^2) );
% normalize = @(p)p/sum(p(:));
% sigma = .06;
% 
% c_1 = Gaussian(x, .2, sigma); 
% c_2 = Gaussian(x, .5, sigma);
% c_3 = Gaussian(x, .8, sigma);
% 
% c_1 = normalize(c_1);
% c_2 = normalize(c_2);
% c_3 = normalize(c_3);

%%
[num_node,~,~] = size(X_pos_rd);

[X_pos, X_neg, X] = sign_matrix(X_pos_rd,X_neg_rd);
[Y_pos, Y_neg, Y] = sign_matrix(Y_pos_rd,Y_neg_rd);
[Z_pos, Z_neg, Z] = sign_matrix(Z_pos_rd,Z_neg_rd);

bQ = X_pos_rd + X_neg_rd;
bQ(bQ>=1) = 1;
Q = bQ;


%% Algorithm initialization

% intial value of variable \mu and \nu, set as all-ones vectors.
alpha_1 = ones(num_node,1);
alpha_2 = ones(num_node,1);
alpha_3 = ones(num_node,1);

% element scaling by exp(-1). % 3D tensor 
Cppp = ((X_pos + Y_pos + Z_pos) == 3).*Q.*exp(-1);
Cppm = ((X_pos + Y_pos + Z_neg) == 3).*Q.*exp(-1);
Cpmp = ((X_pos + Y_neg + Z_pos) == 3).*Q.*exp(-1);
Cpmm = ((X_pos + Y_neg + Z_neg) == 3).*Q.*exp(-1);
Cmpp = ((X_neg + Y_pos + Z_pos) == 3).*Q.*exp(-1);
Cmpm = ((X_neg + Y_pos + Z_neg) == 3).*Q.*exp(-1);
Cmmp = ((X_neg + Y_neg + Z_pos) == 3).*Q.*exp(-1);
Cmmm = ((X_neg + Y_neg + Z_neg) == 3).*Q.*exp(-1);

% empty vector for iteration tracking.
cost = [];
Err_1 = [];
Err_2 = [];
Err_3 = [];

% iteration
iter = 100;

%% forward-backward (sinkhorn-like) iteration
for t = 1:iter


%% 1. update of alpha1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute a,b,c for alpha_1 - c_1 - X
a_p2p3 = tensorprod(alpha_2,alpha_3,2);
a_p2m3 = tensorprod(alpha_2,1./alpha_3,2);

Sp_1 = tensorprod(Cppp,a_p2p3,[2,3],[1,2])+tensorprod(Cppm,a_p2m3,[2,3],[1,2])+tensorprod(Cpmp,1./a_p2m3,[2,3],[1,2])+tensorprod(Cpmm,1./a_p2p3,[2,3],[1,2]);
Sm_1 = tensorprod(Cmpp,a_p2p3,[2,3],[1,2])+tensorprod(Cmpm,a_p2m3,[2,3],[1,2])+tensorprod(Cmmp,1./a_p2m3,[2,3],[1,2])+tensorprod(Cmmm,1./a_p2p3,[2,3],[1,2]);

% address the positive & negative elements, using location indicator
alpha_neg_loc_1 = (Sm_1~=0);
alpha_pos_loc_1 = ~alpha_neg_loc_1;

% update \nu_i in two cases, i.e.,
alpha_var_1 = zeros(num_node,1);
% 1) when a_{ij} < 0, compute \nu_i using function 'FB_solver'
alpha_var_1(alpha_neg_loc_1) = Gsolver(Sp_1(alpha_neg_loc_1),Sm_1(alpha_neg_loc_1),c_1(alpha_neg_loc_1));
% 2) when a_{ij} > 0, compute \nu_i using sinkhorn iteration
alpha_var_1(alpha_pos_loc_1) = c_1(alpha_pos_loc_1)./Sp_1(alpha_pos_loc_1); %%??
% 3) update \nu from the two cases
alpha_1 = alpha_var_1;

% a12  = tensorprod(alpha_1,alpha_2,2);
% a12(isnan(a12)) = 0;
% a_123 = tensorprod(a12,alpha_3);
% a_123(isnan(a_123)) = 0;
% P  = Qe_pos.*a_123 + Qe_neg.*(1./a_123);
P = P_maker(Cppp,Cppm,Cpmp,Cpmm,Cmpp,Cmpm,Cmmp,Cmmm,alpha_1,alpha_2,alpha_3);

Err_3(end+1) = norm(squeeze(sum(sum(Z.*P)))-c_3)/norm(c_3);


%% 2. update of alpha2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute a,b,c for alpha_2 - c_2 - Y
a_p1p3 = tensorprod(alpha_1,alpha_3,2);
a_p1m3 = tensorprod(alpha_1,1./alpha_3,2);

Sp_2 = tensorprod(Cppp,a_p1p3,[1,3],[1,2])+tensorprod(Cppm,a_p1m3,[1,3],[1,2])+tensorprod(Cmpp,1./a_p1m3,[1,3],[1,2])+tensorprod(Cmpm,1./a_p1p3,[1,3],[1,2]);
Sm_2 = tensorprod(Cpmp,a_p1p3,[1,3],[1,2])+tensorprod(Cpmm,a_p1m3,[1,3],[1,2])+tensorprod(Cmmp,1./a_p1m3,[1,3],[1,2])+tensorprod(Cmmm,1./a_p1p3,[1,3],[1,2]);

% address the positive & negative elements, using location indicator
alpha_neg_loc_2 = (Sm_2~=0);
alpha_pos_loc_2 = ~alpha_neg_loc_2;

% update \mu_j in two cases, i.e.,
alpha_var_2 = zeros(num_node,1);
% 1) when a_{ij} < 0, compute \mu_j using function 'FB_solver'
alpha_var_2(alpha_neg_loc_2) = Gsolver(Sp_2(alpha_neg_loc_2),Sm_2(alpha_neg_loc_2),c_2(alpha_neg_loc_2));
% 2) when a_{ij} > 0, compute \mu_j using sinkhorn iteration
alpha_var_2(alpha_pos_loc_2) = c_2(alpha_pos_loc_2)./Sp_2(alpha_pos_loc_2);
% 3) update \nu from the two cases
alpha_2 = alpha_var_2;


%% 3. update of alpha3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute a,b,c for alpha_3 - c_3 - Z
a_p1p2 = tensorprod(alpha_1,alpha_2,2);
a_p1m2 = tensorprod(alpha_1,1./alpha_2,2);

Sp_3 = tensorprod(Cppp,a_p1p2,[1,2],[1,2])+tensorprod(Cpmp,a_p1m2,[1,2],[1,2])+tensorprod(Cmpp,1./a_p1m2,[1,2],[1,2])+tensorprod(Cmmp,1./a_p1p2,[1,2],[1,2]);
Sm_3 = tensorprod(Cppm,a_p1p2,[1,2],[1,2])+tensorprod(Cpmm,a_p1m2,[1,2],[1,2])+tensorprod(Cmpm,1./a_p1m2,[1,2],[1,2])+tensorprod(Cmmm,1./a_p1p2,[1,2],[1,2]);

% address the positive & negative elements, using location indicator
alpha_neg_loc_3 = (Sm_3~=0);
alpha_pos_loc_3 = ~alpha_neg_loc_3;

% update \mu_j in two cases, i.e.,
alpha_var_3 = zeros(num_node,1);
% 1) when a_{ij} < 0, compute \mu_j using function 'FB_solver'
alpha_var_3(alpha_neg_loc_3) = Gsolver(Sp_3(alpha_neg_loc_3),Sm_3(alpha_neg_loc_3),c_3(alpha_neg_loc_3));
% 2) when a_{ij} > 0, compute \mu_j using sinkhorn iteration
alpha_var_3(alpha_pos_loc_3) = c_3(alpha_pos_loc_3)./Sp_3(alpha_pos_loc_3);
% 3) update \nu from the two cases
alpha_3 = alpha_var_3;

%% 4. convergence %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% update \P according to the element-wise closed-form expression. (to be changed)
% P = diag(exp(-mu))*A_pos*diag(exp(-alpha_1)) + diag(exp(mu))*A_neg*diag(exp(alpha_1));

% a12  = tensorprod(alpha_1,alpha_2,2);
% a12(isnan(a12)) = 0;
% a_123 = tensorprod(a12,alpha_3);
% a_123(isnan(a_123)) = 0;
% P  = Qe_pos.*a_123 + Qe_neg.*(1./a_123);
P = P_maker(Cppp,Cppm,Cpmp,Cpmm,Cmpp,Cmpm,Cmmp,Cmmm,alpha_1,alpha_2,alpha_3);

% compute and save the value of the objective function.
obj = P.*log(P./bQ);
obj(isnan(obj)) = 0;
obj_value = sum(real(obj),'all');
cost(end+1) = obj_value;


%% 5.Error storage
% compuate and save the violation/error of the two marginals (marginal constraints).
sign_P = X.*P;

Err_1(end+1) = norm(squeeze(sum(sum(permute(X.*P,[2,3,1])))) - c_1)/norm(c_1);
Err_2(end+1) = norm(squeeze(sum(sum(permute(Y.*P,[3,1,2])))) - c_2)/norm(c_2);

end

squeeze(sum(sum(permute(X.*P,[2,3,1])))) - c_1
squeeze(sum(sum(permute(Y.*P,[3,1,2])))) - c_2
squeeze(sum(sum(Z.*P)))-c_3


%%
figure(1)
subplot(4,1,1);
plot(1:iter,cost,'Color',[0.6350 0.0780 0.1840],'LineWidth',3);
axis tight;
ylabel('Objective') 
title('Convergence')

subplot(4,1,2);
plot(log10(Err_1),'LineWidth',3,'Color',"#0072BD"); 
axis tight; 
title('log||\Sigma XP - c_1||');

subplot(4,1,3);
plot(log10(Err_2),'LineWidth',3,'Color',"#D95319"); 
axis tight; 
title('log||\Sigma YP  - c_2||');

subplot(4,1,4);
plot(log10(Err_3),'LineWidth',3,'Color',"#77AC30"); 
axis tight; 
xlabel('iteration')
title('log||\Sigma ZP - c_3||');

%%
figure(2)
pos_point = Y.*P;
pos_point(pos_point<0) = 0;

neg_point = sign_P;
neg_point(neg_point>0) = 0;


list1 = find (pos_point>0);
[I1,J1,K1] = ind2sub(size(pos_point),list1);

list2 = find (neg_point<0);
[I2,J2,K2] = ind2sub(size(neg_point),list2);

hold on

for i=1:n
     marginal1 = scatter3(i,0.5,0.5,300.*c_1(i),'filled','LineWidth',1.5,'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
end
% 
for i=1:n
     marginal2 = scatter3(0.5,i,0.5,300.*c_2(i),'filled','LineWidth',1.5,'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
end
% 
for i=1:n
     marginal3 = scatter3(0.5,0.5,i,300.*c_3(i),'filled','LineWidth',1.5,'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
end

C = ones(length(I1),1)*[0 0 1];
scatter3sph(I1,J1,K1,'size',(1/3).*sqrt(pos_point(list1)),'color',C,'trans',0.6);

C = [1 0 0];
scatter3sph(I2,J2,K2,'size',(1/3).*sqrt(abs(neg_point(list2))),'color',C,'trans',0.6);


hold off
axis tight
axis equal;
grid on

xticks([1:3])
yticks([1:3])
zticks([1:3])
view([66 16])

% %%
% figure(3)
% pos_point = Y.*P;
% pos_point(pos_point<0) = 0;
% 
% neg_point = sign_P;
% neg_point(neg_point>0) = 0;
% 
% 
% list1 = find (pos_point>0);
% [I1,J1,K1] = ind2sub(size(pos_point),list1);
% 
% list2 = find (neg_point<0);
% [I2,J2,K2] = ind2sub(size(neg_point),list2);
% 
% hold on
% 
% for i=1:n
%      marginal1 = scatter3(i,0.5,0.5,300.*c_1(i),'filled','LineWidth',1.5,'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
% end
% % 
% for i=1:n
%      marginal2 = scatter3(0.5,i,0.5,300.*c_2(i),'filled','LineWidth',1.5,'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
% end
% % 
% for i=1:n
%      marginal3 = scatter3(0.5,0.5,i,300.*c_3(i),'filled','LineWidth',1.5,'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
% end
% 
% C = ones(length(I1),1)*[0 0 1];
% scatter3sph(I1,J1,K1,'size',(1/3).*sqrt(pos_point(list1)),'color',C,'trans',0.6);
% 
% C = [1 0 0];
% scatter3sph(I2,J2,K2,'size',(1/3).*sqrt(abs(neg_point(list2))),'color',C,'trans',0.6);
% 
% 
% hold off
% axis tight
% axis equal;
% grid on
% 
% xticks([1:3])
% yticks([1:3])
% zticks([1:3])
% view([66 16])
% 
% 
% %%
% figure(4)
% pos_point = Z.*P;
% pos_point(pos_point<0) = 0;
% 
% neg_point = sign_P;
% neg_point(neg_point>0) = 0;
% 
% 
% list1 = find (pos_point>0);
% [I1,J1,K1] = ind2sub(size(pos_point),list1);
% 
% list2 = find (neg_point<0);
% [I2,J2,K2] = ind2sub(size(neg_point),list2);
% 
% hold on
% 
% for i=1:n
%      marginal1 = scatter3(i,0.5,0.5,300.*c_1(i),'filled','LineWidth',1.5,'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
% end
% % 
% for i=1:n
%      marginal2 = scatter3(0.5,i,0.5,300.*c_2(i),'filled','LineWidth',1.5,'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
% end
% % 
% for i=1:n
%      marginal3 = scatter3(0.5,0.5,i,300.*c_3(i),'filled','LineWidth',1.5,'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
% end
% 
% C = ones(length(I1),1)*[0 0 1];
% scatter3sph(I1,J1,K1,'size',(1/3).*sqrt(pos_point(list1)),'color',C,'trans',0.6);
% 
% C = [1 0 0];
% scatter3sph(I2,J2,K2,'size',(1/3).*sqrt(abs(neg_point(list2))),'color',C,'trans',0.6);
% 
% 
% hold off
% axis tight
% axis equal;
% grid on
% 
% xticks([1:3])
% yticks([1:3])
% zticks([1:3])
% view([66 16])
% 
% figure(3)
% pos_point = Y.*P;
% pos_point(pos_point<0) = 0;
% 
% neg_point = sign_P;
% neg_point(neg_point>0) = 0;
% 
% 
% list1 = find (pos_point>0);
% [I1,J1,K1] = ind2sub(size(pos_point),list1);
% 
% list2 = find (neg_point<0);
% [I2,J2,K2] = ind2sub(size(neg_point),list2);
% 
% hold on
% 
% for i=1:n
%      marginal1 = scatter3(i,0.5,0.5,300.*c_1(i),'filled','LineWidth',1.5,'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
% end
% % 
% for i=1:n
%      marginal2 = scatter3(0.5,i,0.5,300.*c_2(i),'filled','LineWidth',1.5,'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
% end
% % 
% for i=1:n
%      marginal3 = scatter3(0.5,0.5,i,300.*c_3(i),'filled','LineWidth',1.5,'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
% end
% 
% C = ones(length(I1),1)*[0 0 1];
% scatter3sph(I1,J1,K1,'size',(1/3).*sqrt(pos_point(list1)),'color',C,'trans',0.6);
% 
% C = [1 0 0];
% scatter3sph(I2,J2,K2,'size',(1/3).*sqrt(abs(neg_point(list2))),'color',C,'trans',0.6);
% 
% 
% hold off
% axis tight
% axis equal;
% grid on
% 
% xticks([1:3])
% yticks([1:3])
% zticks([1:3])
% view([66 16])
%%
function P = P_maker(Cppp,Cppm,Cpmp,Cpmm,Cmpp,Cmpm,Cmmp,Cmmm,alpha_1,alpha_2,alpha_3)
p1p2  = tensorprod(alpha_1,alpha_2,2);
p1m2  = tensorprod(alpha_1,1./alpha_2,2);

p1p2(isnan(p1p2)) = 0;
p1m2(isnan(p1m2)) = 0;

p1p2p3 = tensorprod(p1p2,alpha_3);
p1p2m3 = tensorprod(p1p2,1./alpha_3);
p1m2p3 = tensorprod(p1m2,alpha_3);
p1m2m3 = tensorprod(p1m2,alpha_3);

p1p2p3(isnan(p1p2p3)) = 0;
p1p2m3(isnan(p1p2m3)) = 0;
p1m2p3(isnan(p1m2p3)) = 0;
p1m2m3(isnan(p1m2m3)) = 0;

P  = Cppp.*p1p2p3+Cmmm.*(1./p1p2p3)+Cppm.*p1p2m3+Cmmp.*(1./p1p2m3)+Cpmp.*(p1m2p3)+Cmpm.*(1./p1m2p3)+Cpmm.*(p1m2m3)+Cmpp.*(1./p1m2m3);
end 


function [X_plus,X_minu,X] = sign_matrix(Q_pos_rd,Q_neg_rd)

bQ = Q_pos_rd + Q_neg_rd;
bQ(bQ>=1) = 1;
Q = bQ;
Q(Q_neg_rd~=0) = -1;

Q_pos = (bQ + Q)./2;
Q_neg = (bQ - Q)./2;


X_plus = Q_pos;
X_minu = Q_neg;

X = X_plus - X_minu;

end




% figure(2)
% pos_point = sign_P;
% pos_point(pos_point<0) = 0;
% 
% neg_point = sign_P;
% neg_point(neg_point>0) = 0;
% 
% 
% list1 = find (pos_point>0);
% [I1,J1,K1] = ind2sub(size(pos_point),list1);
% 
% list2 = find (neg_point<0);
% [I2,J2,K2] = ind2sub(size(neg_point),list2);
% 
% hold on
% 
% for i=1:n
%      marginal1 = scatter3(i,0.5,0.5,300.*c_1(i),'filled','LineWidth',1.5,'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
% end
% % 
% for i=1:n
%      marginal2 = scatter3(0.5,i,0.5,300.*c_2(i),'filled','LineWidth',1.5,'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
% end
% % 
% for i=1:n
%      marginal3 = scatter3(0.5,0.5,i,300.*c_3(i),'filled','LineWidth',1.5,'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
% end
% 
% C = ones(length(I1),1)*[0 0 1];
% scatter3sph(I1,J1,K1,'size',(1/3).*sqrt(pos_point(list1)),'color',C,'trans',0.6);
% 
% C = [1 0 0];
% scatter3sph(I2,J2,K2,'size',(1/3).*sqrt(abs(neg_point(list2))),'color',C,'trans',0.6);
% 
% 
% hold off
% axis tight
% axis equal;
% grid on
% 
% xticks([1:3])
% yticks([1:3])
% zticks([1:3])
% view([66 16])
% 
% 
% %%
% figure(3)
% 
% Q = Q./sum(Q,"all");
% q_1 = squeeze(sum(sum(permute(Q,[2,3,1]))));
% q_2 = squeeze(sum(sum(permute(Q,[3,1,2]))));
% q_3 = squeeze(sum(sum(Q)));
% 
% 
% 
% Qpos_point = Q;
% Qpos_point(Qpos_point<0) = 0;
% 
% Qneg_point = Q;
% Qneg_point(Qneg_point>0) = 0;
% 
% 
% list1 = find (Qpos_point>0);
% [I1,J1,K1] = ind2sub(size(Qpos_point),list1);
% 
% list2 = find (Qneg_point<0);
% [I2,J2,K2] = ind2sub(size(Qneg_point),list2);
% 
% hold on
% 
% for i=1:n
%      marginal1 = scatter3(i,0.5,0.5,300.*q_1(i),'filled','LineWidth',1.5,'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
% end
% % 
% for i=1:n
%      marginal2 = scatter3(0.5,i,0.5,300.*q_2(i),'filled','LineWidth',1.5,'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
% end
% % 
% for i=1:n
%      marginal3 = scatter3(0.5,0.5,i,300.*q_3(i),'filled','LineWidth',1.5,'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
% end
% 
% 
% C = ones(length(I1),1)*[0 0 1];
% scatter3sph(I1,J1,K1,'size',(1/3).*sqrt(Qpos_point(list1)),'color',C,'trans',0.6);
% 
% C = [1 0 0];
% scatter3sph(I2,J2,K2,'size',(1/3).*sqrt(abs(Qneg_point(list2))),'color',C,'trans',0.6);
% 
% 
% hold off
% axis tight
% grid on
% xticks([1:3])
% yticks([1:3])
% zticks([1:3])
% view([66 16])