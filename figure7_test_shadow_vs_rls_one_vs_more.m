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
for i = 1:50
    Obs_mtx{i+3} = normc(randn(d,1));
end
Obs_fun = @(u,X) trace(u'*X*u);

Obs_val_true = [];
for i = 1:length(Obs_mtx)
    Obs_val_true(i) = Obs_fun(Obs_mtx{i},Xo);
end

Obs_val_shadow = []; Obs_val_rls = [];
State_shadow = []; State_rls = [];
eig_pos_shadow = []; eig_neg_shadow = [];
eig_pos_rls = []; eig_neg_rls = [];
%% generate rank-1 PSD probability measure
mX = 2.^([4:11]);
N_exp = 50;
lambda = 0.01;lengthX = length(mX);lengthObs = length(Obs_mtx);
Lall = 2.^([0:3:11]); %number of shots for each POVM setting
lengthL = length(Lall);
for i = 1:lengthX
    i
    parfor exp = 1:N_exp
        for ll = 1:lengthL
            L = Lall(ll);
            m = mX(i)/L;
            if m<1

                for j = 1:lengthObs
                    Obs_val_shadow(i,ll,j,exp) = NaN;
                    Obs_val_rls(i,ll,j,exp) = NaN;
                end
            else

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
                    %%% store A^T y and shaodw
                    Aty = Aty + temp*counts'/L;
                    temp = U*diag(sqrt(counts/L));
                    shadow = shadow + ( (d+1)*temp*temp' - eye(d));
                end

                %%% shadow
                X_shadow = shadow/m;

                %%%regularized least-squares
                X_rls = pinv(AtA + lambda*eye(size(AtA)))*Aty; % accurate but
                % could be slow
                %X_rls = lsqr(AtA + lambda*eye(size(AtA)),Aty); % a fast method
                X_rls = reshape(X_rls,d,d);
                %%%estimating the quantities
                State_shadow(i,ll,exp) = norm(X_shadow - Xo,'fro');
                State_rls(i,ll,exp) = norm(X_rls - Xo,'fro');
                for j = 1:lengthObs
                    Obs_val_shadow(i,ll,j,exp) = Obs_fun(Obs_mtx{j},X_shadow);
                    Obs_val_rls(i,ll,j,exp) = Obs_fun(Obs_mtx{j},X_rls);
                end
            end
        end
    end
end
%%
save shadow_vs_rls_one_vs_more.mat mX Obs_val_true Obs_val_shadow Obs_val_rls...
    State_shadow State_rls mX Lall
%Obs_val_shadow = mean(Obs_val_shadow,4); Obs_val_rls = mean(Obs_val_rls,4);
%%
load shadow_vs_rls_one_vs_more.mat
Obs_val_shadow = real(Obs_val_shadow); Obs_val_rls = real(Obs_val_rls);
fontsize = 30;linewidth = 4;
%plotStyle = {'b','k-.','r:','g--','p:','k',''};
plotStyle = {'b','k','r','g','p','m'};
%for j = 1:size(Obs_val_shadow,3)
for j = 1:3
    figure
    for i = 1:length(Lall)
        temp = mean((Obs_val_shadow - Obs_val_true(j)).^2,4);
        loglog(mX,temp(:,i,j),plotStyle{i},'linewidth',linewidth);hold on
    end
    xlim([min(mX),max(mX)])
    ylim([4e-4, 1])
    yticks([1e-3 1e-2 1e-1 1])
    %loglog(mX,MSE_shadow(:,2),plotStyle{2},'linewidth',3); hold on
    xlabel('$ML$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
    if j ==1
        legendInfo{1} = ['$L = 1$'];
        legendInfo{2} = ['$L = 8$'];
        legendInfo{3} = ['$L = 64$'];
        legendInfo{4} = ['$L = 512$'];
        legend(legendInfo,'Interpreter','LaTex','Location','Southwest')
        ylabel('MSE by CS','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex')
    end
    set(gca,'LineWidth', 2, 'FontSize', fontsize, 'FontName', 'Times New Roman','Color'      , 'white'                 );
    set(gcf, 'Color', 'white');
    fig_name = strcat('shadow_vs_ls/shadow_one_more_MSE_',num2str(j-1));
    export_fig(fig_name, '-pdf', '-nocrop')

    figure
    for i = 1:length(Lall)
        temp = mean((Obs_val_rls - Obs_val_true(j)).^2,4);
        loglog(mX,temp(:,i,j),plotStyle{i},'linewidth',linewidth);hold on
    end
    xlim([min(mX),max(mX)])
    ylim([4e-4, 1])
    yticks([1e-3 1e-2 1e-1 1])
    xlabel('$ML$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
    if j ==1
        legendInfo{1} = ['$L = 1$'];
        legendInfo{2} = ['$L = 8$'];
        legendInfo{3} = ['$L = 64$'];
        legendInfo{4} = ['$L = 512$'];
        legend(legendInfo,'Interpreter','LaTex','Location','Southwest')
        ylabel('MSE by RLS','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex')
    end
    set(gca,'LineWidth', 2, 'FontSize', fontsize, 'FontName', 'Times New Roman','Color'      , 'white'                 );
    set(gcf, 'Color', 'white');
    fig_name = strcat('shadow_vs_ls/rls_one_more_MSE_',num2str(j-1));
    export_fig(fig_name, '-pdf', '-nocrop')
end



%%% plot the average MSE for random linear observables
figure
for i = 1:length(Lall)
    mse_shadow = zeros(size(Obs_val_shadow,1),1);
    for j = 4:53
        temp = mean((Obs_val_shadow - Obs_val_true(j)).^2,4);
        mse_shadow = mse_shadow +  temp(:,i,j);
    end
    mse_shadow = mse_shadow/50;
    loglog(mX,mse_shadow,plotStyle{i},'linewidth',linewidth);hold on
end
xlim([min(mX),max(mX)])
ylim([4e-4, 1])
yticks([1e-3 1e-2 1e-1 1])
%loglog(mX,MSE_shadow(:,2),plotStyle{2},'linewidth',3); hold on
xlabel('$ML$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
set(gca,'LineWidth', 2, 'FontSize', fontsize, 'FontName', 'Times New Roman','Color'      , 'white'                 );
set(gcf, 'Color', 'white');
fig_name = strcat('shadow_vs_ls/shadow_one_more_MSE_randn');
export_fig(fig_name, '-pdf', '-nocrop')

figure
for i = 1:length(Lall)
    mse_rls = zeros(size(Obs_val_rls,1),1);
    for j = 4:53
        temp = mean((Obs_val_rls - Obs_val_true(j)).^2,4);
        mse_rls = mse_rls +  temp(:,i,j);
    end
    mse_rls = mse_rls/50;
    loglog(mX,mse_rls,plotStyle{i},'linewidth',linewidth);hold on
end
xlim([min(mX),max(mX)])
ylim([4e-4, 1])
yticks([1e-3 1e-2 1e-1 1])
%loglog(mX,MSE_shadow(:,2),plotStyle{2},'linewidth',3); hold on
xlabel('$ML$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');

set(gca,'LineWidth', 2, 'FontSize', fontsize, 'FontName', 'Times New Roman','Color'      , 'white'                 );
set(gcf, 'Color', 'white');
fig_name = strcat('shadow_vs_ls/rls_one_more_MSE_randn');
export_fig(fig_name, '-pdf', '-nocrop')