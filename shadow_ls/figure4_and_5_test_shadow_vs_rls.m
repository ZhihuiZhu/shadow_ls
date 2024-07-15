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

Obs_val_shadow = []; Obs_val_rls = [];
State_shadow = []; State_rls = [];
eig_pos_shadow = []; eig_neg_shadow = [];
eig_pos_rls = []; eig_neg_rls = [];
%% generate rank-1 PSD probability measure
mX = 2.^([2:11]);
L = 1; %number of shots for each POVM setting
N_exp = 50;
lambda = 0.1;lengthX = length(mX);lengthObs = length(Obs_mtx);
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

            %%% V2 for N>1
            %   temp = U*diag(sqrt(counts/N));
            %   shadow = shadow + ( (d+1)*temp*temp' - eye(d));
            %   y = [y counts/L]; % measurements

        end

        %%% shadow
        X_shadow = shadow/m;

        %%%regularized least-squares
        X_rls = pinv(AtA + lambda*eye(size(AtA)))*Aty; % accurate but
        % could be slow
        %X_rls = lsqr(AtA + lambda*eye(size(AtA)),Aty); % a fast method
        X_rls = reshape(X_rls,d,d);
        %%%estimating the quantities
        State_shadow(i,exp) = norm(X_shadow - Xo,'fro');
        State_rls(i,exp) = norm(X_rls - Xo,'fro');

        for j = 1:lengthObs
            Obs_val_shadow(i,j,exp) = Obs_fun(Obs_mtx{j},X_shadow);
            Obs_val_rls(i,j,exp) = Obs_fun(Obs_mtx{j},X_rls);
        end


    end

end
%%
save shadow_vs_rls.mat mX Obs_val_true Obs_val_shadow Obs_val_rls...
    State_shadow State_rls eig_pos_shadow eig_neg_shadow eig_pos_rls eig_neg_rls

%%
load shadow_vs_rls.mat
fontsize = 30;
plotStyle = {'b','k-.','r:','g:','p:'};
Obs_val_shadow = real(Obs_val_shadow);
Obs_val_rls= real(Obs_val_rls);
lengthObs = 3;
for j = 1:lengthObs
    figure
    temp = mean((Obs_val_shadow - Obs_val_true(j)).^2,3); loglog(mX,temp(:,j),plotStyle{1},'linewidth',4);hold on
    temp = mean((Obs_val_rls - Obs_val_true(j)).^2,3); loglog(mX,temp(:,j),plotStyle{2},'linewidth',4);
    %  temp = mean(((Obs_val_rls + Obs_val_shadow)/2 - Obs_val_true(j)).^2,3); loglog(mX,temp(:,j),plotStyle{3},'linewidth',3);
    xlim([min(mX),max(mX)])
    xticks([10 100 1000])
    ylim([4e-4, 1])
    yticks([1e-3 1e-2 1e-1 1])

    xlabel('$M$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
    if j ==1
        legendInfo{1} = ['CS'];
        legendInfo{2} = ['RLS'];
        %  legendInfo{3} = ['Shadow + RLS'];
        legend(legendInfo,'Interpreter','LaTex','Location','southwest')
        ylabel('MSE','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex')
    end
    set(gca,'LineWidth', 2, 'FontSize', fontsize, 'FontName', 'Times New Roman','Color'      , 'white'                 );
    set(gcf, 'Color', 'white');
    fig_name = strcat('shadow_vs_ls/rls_shadow_obs_MSE_',num2str(j-1));
    export_fig(fig_name, '-pdf', '-nocrop')
end


%%% plot the average MSE for random linear observables
figure
mse_shadow = zeros(size(Obs_val_shadow,1),1);
for i = 1:50
    temp = mean((Obs_val_shadow - Obs_val_true(i+3)).^2,3);
    mse_shadow = mse_shadow + temp(:,i+3);
end
mse_shadow = mse_shadow/50;
loglog(mX,mse_shadow,plotStyle{1},'linewidth',4);hold on

mse_rls = zeros(size(Obs_val_shadow,1),1);
for i = 1:50
    temp = mean((Obs_val_rls - Obs_val_true(i+3)).^2,3);
    mse_rls = mse_rls + temp(:,i+3);
end
mse_rls = mse_rls/50;

loglog(mX,mse_rls,plotStyle{2},'linewidth',4);
xlim([min(mX),max(mX)])
xticks([10 100 1000])
ylim([4e-4, 1])
yticks([1e-3 1e-2 1e-1 1])

xlabel('$M$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');


legendInfo{1} = ['CS'];
        legendInfo{2} = ['RLS'];
        %  legendInfo{3} = ['Shadow + RLS'];
        legend(legendInfo,'Interpreter','LaTex','Location','southwest')
        ylabel('MSE','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex')

set(gca,'LineWidth', 2, 'FontSize', fontsize, 'FontName', 'Times New Roman','Color'      , 'white'                 );
set(gcf, 'Color', 'white');
fig_name = strcat('shadow_vs_ls/rls_shadow_obs_MSE_randn');
export_fig(fig_name, '-pdf', '-nocrop')


%%%
figure
loglog(mX,mean(State_shadow'),plotStyle{1},'linewidth',4);hold on
loglog(mX,mean(State_rls'),plotStyle{2},'linewidth',4);
xlim([min(mX),max(mX)])
ylim([0.5,40])
xticks([10 100 1000])
legendInfo{1} = ['CS'];
legendInfo{2} = ['RLS'];
legend(legendInfo,'Interpreter','LaTex','Location','southwest')
xlabel('$M$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
ylabel('$\|\widehat \rho - \rho\|_F$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex')
set(gca,'LineWidth', 2, 'FontSize', fontsize, 'FontName', 'Times New Roman','Color'      , 'white'                 );
set(gcf, 'Color', 'white');
fig_name = strcat('shadow_vs_ls/rls_shadow_state');
export_fig(fig_name, '-pdf', '-nocrop')



%%% plot estimators
for j = 1:lengthObs
    figure
    for exp = 1:size(Obs_val_shadow,3)
        semilogx(mX,Obs_val_shadow(:,j,exp),'b-','linewidth',1); hold on
    end
    xlim([min(mX),max(mX)])
    ylim([-2,3.1])
    xticks([10 100 1000])
    xlabel('$M$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
    if j == 1
        ylabel('CS','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex')
    end
    set(gca,'LineWidth', 2, 'FontSize', fontsize, 'FontName', 'Times New Roman','Color'      , 'white'                 );
    set(gcf, 'Color', 'white');
    fig_name = strcat('shadow_vs_ls/shadow_obs_',num2str(j-1));
    export_fig(fig_name, '-pdf', '-nocrop')
end

for j = 1:lengthObs
    figure
    for exp = 1:size(Obs_val_shadow,3)
        semilogx(mX,Obs_val_rls(:,j,exp),'b-','linewidth',1); hold on
    end
    xlim([min(mX),max(mX)])
    ylim([-2,3.1])
    xticks([10 100 1000])
    xlabel('$M$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
    if j ==1
        ylabel('RLS','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex')
    end
    set(gca,'LineWidth', 2, 'FontSize', fontsize, 'FontName', 'Times New Roman','Color'      , 'white'                 );
    set(gcf, 'Color', 'white');
    fig_name = strcat('shadow_vs_ls/rls_obs_',num2str(j-1));
    export_fig(fig_name, '-pdf', '-nocrop')
end

%%%
plotStyle = {'b','b:','k','k:'};
figure
semilogx(mX,mean(eig_pos_shadow'),plotStyle{1},'linewidth',4); hold on
semilogx(mX,mean(eig_neg_shadow'),plotStyle{2},'linewidth',4);
loglog(mX,mean(eig_pos_rls'),plotStyle{3},'linewidth',4);
loglog(mX,mean(eig_neg_rls'),plotStyle{4},'linewidth',4);
%loglog(mX,MSE_shadow(:,2),plotStyle{2},'linewidth',3); hold on
legendInfo{1} = ['CS: sum of positive eigs'];
legendInfo{2} = ['CS: sum of negative eigs'];
legendInfo{3} = ['RLS: sum of positive eigs'];
legendInfo{4} = ['RLS: sum of negative eigs'];
xlim([min(mX),max(mX)])
xticks([10 100 1000])
ylim([-75,75])
% legendInfo{3} = ['M = 16 repeated measurements'];
% legendInfo{4} = ['M = 64 repeated measurements'];
% legendInfo{5} = ['M = 256 repeated measurements'];
xlabel('$M$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
set(gca,'LineWidth', 2, 'FontSize', fontsize, 'FontName', 'Times New Roman','Color'      , 'white'                 );
set(gcf, 'Color', 'white');
legend(legendInfo,'Interpreter','LaTex','Location','Best','fontsize',24)
fig_name = strcat('shadow_vs_ls/shadow_vs_rls_eig');
export_fig(fig_name, '-pdf', '-nocrop')