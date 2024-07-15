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
Obs_val_ls = [];
State_ls = [];
eig_pos_ls = []; eig_neg_ls = [];
%% generate rank-1 PSD probability measure
mX = 2.^([0:9]);
N_exp = 50; 
L = 1;% number of shots for each POVM setting
lambda = 0;lengthX = length(mX);lengthObs = length(Obs_mtx);
parfor exp = 1:N_exp
    for i = 1:lengthX
        if exp == N_exp
        i
        end
        m = mX(i);
        shadow = zeros(d);
        %%% generate measurements
        AtA = zeros(d^2,d^2);
        A = [];
        y = [];
        Aty = zeros(d^2,1);
        for t = 1:m
            [U] = orth(randn(d) + 1i*randn(d));
            %             for ii = 1:d
            %                 A = [A;(kron(U(:,ii),conj(U(:,ii))))'];
            %             end
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
            Aty = Aty + kron(U(:,indx),conj(U(:,indx)));

            %%% V2 for L>1
            %   temp = U*diag(sqrt(counts/N));
            %   shadow = shadow + ( (d+1)*temp*temp' - eye(d));
            %   y = [y counts/N]; % measurements

        end

        %%%least-squares
        X_ls = pinv(AtA + lambda*eye(size(AtA)))*Aty;
        X_ls = reshape(X_ls,d,d);
        [U S] = eig(X_ls);
        s = real(diag(S));
        eig_pos_ls(i,exp) = sum(max(s,0));
        eig_neg_ls(i,exp) = sum(min(s,0));
        %%%estimating the quantities
        State_ls(i,exp) = norm(X_ls - Xo,'fro');

        for j = 1:lengthObs
            Obs_val_ls(i,j,exp) = Obs_fun(Obs_mtx{j},X_ls);
        end

    end

end
save ls.mat mX Obs_val_true Obs_val_ls State_ls eig_pos_ls eig_neg_ls
%% plot the results

fontsize = 30;
plotStyle = {'b','k-.','r-','g:','p:'};
Obs_val_ls= real(Obs_val_ls);
lengthObs = 3;
for j = 1:lengthObs
    figure
    temp = mean((Obs_val_ls - Obs_val_true(j)).^2,3); loglog(mX,temp(:,j),plotStyle{1},'linewidth',4);
    xlim([min(mX),max(mX)])
    xticks([10 100 1000])
    ylim([4e-4, 1])
    yticks([1e-3 1e-2 1e-1 1])
    xlabel('$M$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
    if j ==1
    ylabel('MSE','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex')
    end
    set(gca,'LineWidth', 2, 'FontSize', fontsize, 'FontName', 'Times New Roman','Color'      , 'white'                 );
    set(gcf, 'Color', 'white');
    fig_name = strcat('shadow_vs_ls/ls_obs_MSE_',num2str(j-1));
    export_fig(fig_name, '-pdf', '-nocrop')
end

figure
loglog(mX,mean(State_ls'),plotStyle{1},'linewidth',4);
xlim([min(mX),max(mX)])
xticks([10 100 1000])
ylim([0.5,40])
xlabel('$M$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
ylabel('$\|\widehat \rho - \rho\|_F$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex')
set(gca,'LineWidth', 2, 'FontSize', fontsize, 'FontName', 'Times New Roman','Color'      , 'white'                 );
set(gcf, 'Color', 'white');
fig_name = strcat('shadow_vs_ls/ls_state');
export_fig(fig_name, '-pdf', '-nocrop')



%% plot estimators
j = 1
figure
for exp = 1:size(Obs_val_ls,3)
    semilogx(mX,Obs_val_ls(:,j,exp),'b-','linewidth',1); hold on
end
xlim([min(mX),max(mX)])
ylim([min(min(Obs_val_ls(:,j,:))),max(max(Obs_val_ls(:,j,:)))])
xticks([10 100 1000])
ylim([-2,3.1])
xlabel('$M$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
ylabel('LS','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex')
set(gca,'LineWidth', 2, 'FontSize', fontsize, 'FontName', 'Times New Roman','Color'      , 'white'                 );
set(gcf, 'Color', 'white');
fig_name = strcat('shadow_vs_ls/ls_obs_',num2str(j-1));
export_fig(fig_name, '-pdf', '-nocrop')


j = 2
figure
for exp = 1:size(Obs_val_ls,3)
    semilogx(mX,Obs_val_ls(:,j,exp),'b-','linewidth',1); hold on
end
xlim([min(mX),max(mX)])
ylim([min(min(Obs_val_ls(:,j,:))),max(max(Obs_val_ls(:,j,:)))])
xticks([10 100 1000])
ylim([-2,3.1])
xlabel('$M$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
%ylabel('LS','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex')
set(gca,'LineWidth', 2, 'FontSize', fontsize, 'FontName', 'Times New Roman','Color'      , 'white'                 );
set(gcf, 'Color', 'white');
fig_name = strcat('shadow_vs_ls/ls_obs_',num2str(j-1));
export_fig(fig_name, '-pdf', '-nocrop')


j = 3
figure
for exp = 1:size(Obs_val_ls,3)
    semilogx(mX,Obs_val_ls(:,j,exp),'b-','linewidth',1); hold on
end
xlim([min(mX),max(mX)])
ylim([min(min(Obs_val_ls(:,j,:))),max(max(Obs_val_ls(:,j,:)))])
xticks([10 100 1000])
ylim([-2,3.1])
xlabel('$M$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
%ylabel('LS','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex')
set(gca,'LineWidth', 2, 'FontSize', fontsize, 'FontName', 'Times New Roman','Color'      , 'white'                 );
set(gcf, 'Color', 'white');
fig_name = strcat('shadow_vs_ls/ls_obs_',num2str(j-1));
export_fig(fig_name, '-pdf', '-nocrop')
%%
plotStyle = {'b','b:','r-','r:'};
figure
semilogx(mX,mean(eig_pos_ls'),plotStyle{1},'linewidth',4);hold on
semilogx(mX,mean(eig_neg_ls'),plotStyle{2},'linewidth',4);
%loglog(mX,MSE_shadow(:,2),plotStyle{2},'linewidth',3); hold on
legendInfo{1} = ['sum of positive eigs'];
legendInfo{2} = ['sum of negative eigs'];
xlim([min(mX),max(mX)])
xticks([10 100 1000])
ylim([-75,75])
% legendInfo{3} = ['M = 16 repeated measurements'];
% legendInfo{4} = ['M = 64 repeated measurements'];
% legendInfo{5} = ['M = 256 repeated measurements'];
legend(legendInfo,'Interpreter','LaTex','Location','Best')
xlabel('$M$','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex');
%ylabel('sum of eigs','FontSize',fontsize,'FontName','Times New Roman','Interpreter','LaTex')
set(gca,'LineWidth', 2, 'FontSize', fontsize, 'FontName', 'Times New Roman','Color'      , 'white'                 );
set(gcf, 'Color', 'white');
fig_name = strcat('shadow_vs_ls/ls_eig');
export_fig(fig_name, '-pdf', '-nocrop')
