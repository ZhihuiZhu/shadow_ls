%%%
%Estimate rank-1 PSD matrix X = uu^T from rank-1 POVMs {U_i}
%We generate each POVM U = {u_j} as a random orthonormal basis: the
%corresponding POVM measurments are p(j) = (u_j^T u)^2
addpath(genpath('./'));
clear all; close all;
randn('seed',1); rand('seed',1)
%% generate the low-rank matrix
n = 5; % number of bits
d = 2^n;
r = 1;
%u_true = randn(d,r);u_true = u_true/norm(u_true,'fro');
u_true = zeros(d,1); u_true(1) = 1;
Xo = u_true*u_true';

%% observables to be estimated

u_1 = zeros(d,1); u_1(1) = 1;
u_2 = [1/sqrt(2);1/sqrt(2*(d-1))*ones(d-1,1)];
u_3 = zeros(d,1); u_3(2) = 1;
Obs_mtx{1} = u_1; Obs_mtx{2} = u_2; Obs_mtx{3} = u_3;
Obs_fun = @(u,X) trace(u'*X*u);
for i = 1:50
    Obs_mtx{i+3} = normc(randn(d,1));
end
Obs_val_true = [];
for i = 1:length(Obs_mtx)
    Obs_val_true(i) = Obs_fun(Obs_mtx{i},Xo);
end

Obs_val_rls = [];State_rls = [];

%% generate rank-1 PSD probability measure
mX = 2.^([2:11]);
L = 1; %number of shots for each POVM setting
N_exp = 50;
Lambda = logspace(-3,1,10);
length_Lambda = length(Lambda);lengthX = length(mX);lengthObs = length(Obs_mtx);

for i = 1:lengthX
    i
    m = mX(i);
    parfor exp = 1:N_exp
        shadow = zeros(d);
        %%% generate measurements
        AtA = zeros(d^2,d^2);
        A = [];
        y = [];
        Aty = zeros(d^2,1);
        for t = 1:m
            [U] = orth(randn(d) + 1i*randn(d));
            %%% store A^T A matrix
            indx_temp = 1:d+1:d^2;
            temp = kron(U,conj(U)); % this step can be accelerated
            temp = temp(:,indx_temp);
            AtA = AtA + temp*temp';
            probs = sum((abs(U'*u_true)).^2, 2);
            % Counts for each outcome can be generated using the |histc| function.
            counts = histcounts(rand(L,1), [0; cumsum(probs)]);

            %%% stroe A^T y: V1 for L = 1
            indx = find(counts ==1);
            shadow = shadow + ( (d+1)*U(:,indx)*U(:,indx)' - eye(d));
            Aty = Aty + kron(U(:,indx),conj(U(:,indx)));

            %%% V2 for L>1
            %   temp = U*diag(sqrt(counts/N));
            %   shadow = shadow + ( (d+1)*temp*temp' - eye(d));
            %   y = [y counts/L]; % measurements

        end


        for tt = 1:length_Lambda
            lambda = Lambda(tt);
            %%%regularized least-squares
            X_rls = pinv(AtA + lambda*eye(size(AtA)))*Aty; % accurate but
            % could be slow
            %X_rls = lsqr(AtA + lambda*eye(size(AtA)),Aty); % a fast method
            X_rls = reshape(X_rls,d,d);
            %%%estimating the quantities
            State_rls(i,tt,exp) = norm(X_rls - Xo,'fro');

            for j = 1:lengthObs
                Obs_val_rls(i,j,tt,exp) = Obs_fun(Obs_mtx{j},X_rls);
            end
        end

    end

end
%%
save rls_diff_lambda.mat mX Obs_val_true Obs_val_rls State_rls eig_pos_ls eig_neg_ls

%% plot the results

load rls_diff_lambda.mat
temp_state = State_rls(:,end,:);
temp_Obs_val = real(Obs_val_rls);
fontsize = 30;
Lambda = [1e-3 1e-2 1e-1 1 10];
plotStyle = {'b','k-.','r:','g--','c:','k',''};
Obs_val_rls= real(Obs_val_rls);
lengthObs = 3;
figure
for i = 1:2:length(Lambda)
    temp = State_rls(:,i,:);
    loglog(mX,mean(temp,3),plotStyle{(i+1)/2},'linewidth',4);hold on
end
loglog(mX,mean(temp_state,3),plotStyle{5},'linewidth',4);
xlim([min(mX),max(mX)])
    ylim([0.5,40])
    xticks([10 100 1000])
    legendStrings = "\mu = " + string(Lambda);
    legend(legendStrings)
    xlabel('$M$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
    ylabel('$\|\widehat \rho - \rho\|_F$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex')
    set(gca,'LineWidth', 2, 'FontSize', fontsize, 'FontName', 'Times New Roman','Color'      , 'white'                 );
    set(gcf, 'Color', 'white');
fig_name = strcat('shadow_vs_ls/rls_diff_lambda_state');
 export_fig(fig_name, '-pdf', '-nocrop')
 

figure
j = 1;
for i = 1:2:length(Lambda)
    temp = mean((Obs_val_rls - Obs_val_true(j)).^2,4);
    loglog(mX,temp(:,j,i),plotStyle{(i+1)/2},'linewidth',4);hold on
end
 temp = mean((temp_Obs_val - Obs_val_true(j)).^2,4);
  loglog(mX,temp(:,j,end),plotStyle{5},'linewidth',4);hold on
legendStrings = "\mu = " + string(Lambda);
% legend(legendStrings)
xlim([min(mX),max(mX)])
xticks([10 100 1000])
ylim([4e-4, 1])
yticks([1e-3 1e-2 1e-1 1])
xlabel('$M$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
ylabel('MSE','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex')
set(gca,'LineWidth', 2, 'FontSize', fontsize, 'FontName', 'Times New Roman','Color'      , 'white'                 );
set(gcf, 'Color', 'white');
fig_name = strcat('shadow_vs_ls/rls_diff_lambda_obs_MSE',num2str(j-1));
 export_fig(fig_name, '-pdf', '-nocrop')

figure
j = 2;
for i = 1:2:length(Lambda)
    temp = mean((Obs_val_rls - Obs_val_true(j)).^2,4);
    loglog(mX,temp(:,j,i),plotStyle{(i+1)/2},'linewidth',4);hold on
end
 temp = mean((temp_Obs_val - Obs_val_true(j)).^2,4);
  loglog(mX,temp(:,j,end),plotStyle{5},'linewidth',4);hold on
legendStrings = "\mu = " + string(Lambda);
%    legend(legendStrings)
xlim([min(mX),max(mX)])
xticks([10 100 1000])
ylim([4e-4, 1])
yticks([1e-3 1e-2 1e-1 1])
xlabel('$M$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
set(gca,'LineWidth', 2, 'FontSize', fontsize, 'FontName', 'Times New Roman','Color'      , 'white'                 );
set(gcf, 'Color', 'white');
fig_name = strcat('shadow_vs_ls/rls_diff_lambda_obs_MSE',num2str(j-1));
 export_fig(fig_name, '-pdf', '-nocrop')

figure
j = 3;
for i = 1:2:length(Lambda)
    temp = mean((Obs_val_rls - Obs_val_true(j)).^2,4);
    loglog(mX,temp(:,j,i),plotStyle{(i+1)/2},'linewidth',4);hold on
end
 temp = mean((temp_Obs_val - Obs_val_true(j)).^2,4);
  loglog(mX,temp(:,j,end),plotStyle{5},'linewidth',4);hold on
legendStrings = "\mu = " + string(Lambda);
%    legend(legendStrings)
xlim([min(mX),max(mX)])
xticks([10 100 1000])
ylim([4e-4, 1])
yticks([1e-3 1e-2 1e-1 1])
xlabel('$M$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
set(gca,'LineWidth', 2, 'FontSize', fontsize, 'FontName', 'Times New Roman','Color'      , 'white'                 );
set(gcf, 'Color', 'white');
fig_name = strcat('shadow_vs_ls/rls_diff_lambda_obs_MSE',num2str(j-1));
 export_fig(fig_name, '-pdf', '-nocrop')

