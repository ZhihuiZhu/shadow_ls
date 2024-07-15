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
u_4 = randn(d,d);u_4 = u_4/norm(u_4,'fro');
Obs_mtx{1} = u_1; Obs_mtx{2} = u_2; Obs_mtx{3} = u_3; Obs_mtx{4} = u_4;
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

m = 256;
Eta = 0:0.1:0.5;
L = 1; %number of shots for each POVM setting
N_exp = 50;
lambda = 0.1;lengthObs = length(Obs_mtx);
lengthEta = length(Eta);
parfor exp = 1:N_exp
    for i = 1:lengthEta
        if exp == N_exp
            i
        end
        shadow = zeros(d);
        %%% generate measurements
        AtA = zeros(d^2,d^2);
        A = [];
        y = [];
        Aty = zeros(d^2,1);
        eta = Eta(i);
        for t = 1:m
            temp = rand;
            if temp < eta %local rotation
                U = orth(randn(2) + 1i*randn(2));
                for kk = 1:n-1
                    U = kron(U,orth(randn(2) + 1i*randn(2)));
                end
            else %global rotation
                U = orth(randn(d) + 1i*randn(d));
            end
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
        end

        %%% shadow
        X_shadow = shadow/m;
        [U S] = eig(X_shadow);
        s = real(diag(S));
        eig_pos_shadow(i,exp) = sum(max(s,0));
        eig_neg_shadow(i,exp) = sum(min(s,0));

        %%%regularized least-squares
        X_rls = pinv(AtA + lambda*eye(size(AtA)))*Aty;
        X_rls = reshape(X_rls,d,d);
        temp = (X_rls + X_rls')/2;
        [U S] = eig(temp);
        s = real(diag(S));
        eig_pos_rls(i,exp) = sum(max(s,0));
        eig_neg_rls(i,exp) = sum(min(s,0));
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
save shadow_vs_rls_model_uncertainty.mat Eta Obs_val_true Obs_val_shadow Obs_val_rls...
    State_shadow State_rls eig_pos_shadow eig_neg_shadow eig_pos_rls eig_neg_rls

load shadow_vs_rls_model_uncertainty.mat

fontsize = 30;linewidth = 4;
plotStyle = {'b','k-.','r-','g:','p:'};
Obs_val_shadow = real(Obs_val_shadow); 
Obs_val_rls= real(Obs_val_rls); 
lengthObs = 4;
for j = 1:lengthObs-1
    figure
    temp = mean((Obs_val_shadow - Obs_val_true(j)).^2,3); semilogy(Eta,temp(:,j),plotStyle{1},'linewidth',linewidth);hold on
    temp = mean((Obs_val_rls - Obs_val_true(j)).^2,3); semilogy(Eta,temp(:,j),plotStyle{2},'linewidth',linewidth);
    xlim([min(Eta),max(Eta)])
     ylim([4e-4, 1.3])
    yticks([1e-3 1e-2 1e-1 1])
    xlabel('$\eta$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
    if j ==1
    ylabel('MSE','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex')
        legendInfo{1} = ['Shadow'];
    legendInfo{2} = ['RLS'];
    legend(legendInfo,'Interpreter','LaTex','Location','best')
    end
    set(gca,'LineWidth', 2, 'FontSize', fontsize, 'FontName', 'Times New Roman','Color'      , 'white'                 );
    set(gcf, 'Color', 'white');
    fig_name = strcat('shadow_vs_ls/rls_shadow_model_shift_obs_MSE_',num2str(j-1));
    export_fig(fig_name, '-pdf', '-nocrop')
end


% plot estimators
for j = 1:lengthObs-1
    figure
    for exp = 1:size(Obs_val_shadow,3)
        plot(Eta,Obs_val_shadow(:,j,exp),'b-','linewidth',1); hold on
    end
    xlim([min(Eta),max(Eta)])
    ylim([-2,3.1])
    xlabel('$\eta$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
    if j==1
    ylabel('CS','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex')
    end
    set(gca,'LineWidth', 2, 'FontSize', fontsize, 'FontName', 'Times New Roman','Color'      , 'white'                 );
    set(gcf, 'Color', 'white');
    fig_name = strcat('shadow_vs_ls/shadow_model_shift_obs_',num2str(j-1));
    export_fig(fig_name, '-pdf', '-nocrop')
end

for j = 1:lengthObs-1
    figure
    for exp = 1:size(Obs_val_shadow,3)
        plot(Eta,Obs_val_rls(:,j,exp),'b-','linewidth',1); hold on
    end
    xlim([min(Eta),max(Eta)])
    ylim([-2,3.1])
    xlabel('$\eta$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
    if j ==1
    ylabel('RLS','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex')
    end
    set(gca,'LineWidth', 2, 'FontSize', fontsize, 'FontName', 'Times New Roman','Color'      , 'white'                 );
    set(gcf, 'Color', 'white');
    fig_name = strcat('shadow_vs_ls/rls_model_shift_obs_',num2str(j-1));
    export_fig(fig_name, '-pdf', '-nocrop')
end