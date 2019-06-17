%% Processing results from the IKSVD and IKMOD experiments
clear all
clc
%% Open all IKSVD things
fileListIKSVD = dir(['results' filesep 'iksvd*']);
for i=1:size(fileListIKSVD,1)
    load([fileListIKSVD(i).folder filesep fileListIKSVD(i).name])
    
    acc_train_IKSVD(i,:) = accs.acc_train;
    acc_test_IKSVD(i,:) = accs.acc_test;
    acc_valid_IKSVD(i,:) = accs.acc_valid;
    
    recon_train_IKSVD(i,:) = recons.recon_err_train;
    recon_test_IKSVD(i,:) = recons.recon_err_test;
    recon_valid_IKSVD(i,:) = recons.recon_err_valid;
end
%% Open all IKMOD things
fileListIKMOD = dir(['results' filesep 'ikmod*']);
for i=1:size(fileListIKMOD,1)
    load([fileListIKMOD(i).folder filesep fileListIKMOD(i).name])
    
    acc_train_IKMOD(i,:) = accs.acc_train;
    acc_test_IKMOD(i,:) = accs.acc_test;
    acc_valid_IKMOD(i,:) = accs.acc_valid;
    
    recon_train_IKMOD(i,:) = recons.recon_err_train;
    recon_test_IKMOD(i,:) = recons.recon_err_test;
    recon_valid_IKMOD(i,:) = recons.recon_err_valid;
end
%% Plot things
figure(1); clf; hold on;
newAtoms = 1;
h1 = plot(newAtoms*[0:size(acc_train_IKMOD,2)-1], 100*mean(acc_train_IKSVD,1),'-o','LineWidth',2);
h2 = plot(newAtoms*[0:size(acc_train_IKMOD,2)-1], 100*mean(acc_train_IKMOD,1),'-*','LineWidth',2);

plot(newAtoms*[0:size(acc_train_IKMOD,2)-1], 100*(mean(acc_train_IKSVD,1)-std(acc_train_IKSVD,1)), '--o', 'Color', h1.Color, 'HandleVisibility', 'off','LineWidth',.01)
plot(newAtoms*[0:size(acc_train_IKMOD,2)-1], (min([ 100*(mean(acc_train_IKSVD,1)+std(acc_train_IKSVD,1));100*ones(1,size(acc_train_IKSVD,2))],[],1)), '--o', 'Color', h1.Color, 'HandleVisibility', 'off','LineWidth',.01)

plot(newAtoms*[0:size(acc_train_IKMOD,2)-1], 100*(mean(acc_train_IKMOD,1)-std(acc_train_IKMOD,1)), '--*', 'Color', h2.Color, 'HandleVisibility', 'off','LineWidth',.01)
plot(newAtoms*[0:size(acc_train_IKMOD,2)-1], (min([ 100*(mean(acc_train_IKMOD,1)+std(acc_train_IKMOD,1));100*ones(1,size(acc_train_IKMOD,2))],[],1)), '--*', 'Color', h2.Color, 'HandleVisibility', 'off','LineWidth',.01)

legend('IK-SVD', 'IDUO','location','southeast')
xlabel('Number of new atoms added to dictionary')
ylabel('CCR (%)')
title('Correct classification on baseline set')
ylim([99 100])
grid on

figure(2); clf; hold on;
h1 = plot(newAtoms*[0:size(acc_test_IKMOD,2)-1], 100*mean(acc_test_IKSVD,1),'-o','LineWidth',2);
h2 = plot(newAtoms*[0:size(acc_test_IKMOD,2)-1], 100*mean(acc_test_IKMOD,1),'-*','LineWidth',2);

plot(newAtoms*[0:size(acc_test_IKMOD,2)-1], 100*(mean(acc_test_IKSVD,1)-std(acc_test_IKSVD,1)), '--o', 'Color', h1.Color, 'HandleVisibility', 'off','LineWidth',.01)
plot(newAtoms*[0:size(acc_test_IKMOD,2)-1], 100*(mean(acc_test_IKSVD,1)+std(acc_test_IKSVD,1)), '--o', 'Color', h1.Color, 'HandleVisibility', 'off','LineWidth',.01)

plot(newAtoms*[0:size(acc_test_IKMOD,2)-1], 100*(mean(acc_test_IKMOD,1)-std(acc_test_IKMOD,1)), '--*', 'Color', h2.Color, 'HandleVisibility', 'off','LineWidth',.01)
plot(newAtoms*[0:size(acc_test_IKMOD,2)-1], 100*(mean(acc_test_IKMOD,1)+std(acc_test_IKMOD,1)), '--*', 'Color', h2.Color, 'HandleVisibility', 'off','LineWidth',.01)

legend('IK-SVD', 'IDUO','location','southeast')
xlabel('Number of new atoms added to dictionary')
ylabel('CCR (%)')
title('Correct classification on in-situ set')
grid on

figure(3); clf; hold on;
h1 = plot(newAtoms*[0:size(acc_valid_IKMOD,2)-1], 100*mean(acc_valid_IKSVD,1),'-o','LineWidth',2);
h2 = plot(newAtoms*[0:size(acc_valid_IKMOD,2)-1], 100*mean(acc_valid_IKMOD,1),'-*','LineWidth',2);

plot(newAtoms*[0:size(acc_valid_IKMOD,2)-1], 100*(mean(acc_valid_IKSVD,1)-std(acc_valid_IKSVD,1)), '--o', 'Color', h1.Color, 'HandleVisibility', 'off','LineWidth',.01)
plot(newAtoms*[0:size(acc_valid_IKMOD,2)-1], 100*(mean(acc_valid_IKSVD,1)+std(acc_valid_IKSVD,1)), '--o', 'Color', h1.Color, 'HandleVisibility', 'off','LineWidth',.01)

plot(newAtoms*[0:size(acc_valid_IKMOD,2)-1], 100*(mean(acc_valid_IKMOD,1)-std(acc_valid_IKMOD,1)), '--*', 'Color', h2.Color, 'HandleVisibility', 'off','LineWidth',.01)
plot(newAtoms*[0:size(acc_valid_IKMOD,2)-1], 100*(mean(acc_valid_IKMOD,1)+std(acc_valid_IKMOD,1)), '--*', 'Color', h2.Color, 'HandleVisibility', 'off','LineWidth',.01)

legend('IK-SVD', 'IDUO','location','southeast')
xlabel('Number of new atoms added to dictionary')
ylabel('CCR (%)')
title('Correct classification on generalization set')
grid on


figure(4); clf; hold on;
h1 = plot(newAtoms*[0:size(recon_train_IKMOD,2)-1], mean(recon_train_IKSVD,1),'-o','LineWidth',2);
h2 = plot(newAtoms*[0:size(recon_train_IKMOD,2)-1], mean(recon_train_IKMOD,1),'-*','LineWidth',2);

plot(newAtoms*[0:size(recon_train_IKMOD,2)-1], mean(recon_train_IKSVD,1)-std(recon_train_IKSVD,1), '--o', 'Color', h1.Color, 'HandleVisibility', 'off','LineWidth',.01)
plot(newAtoms*[0:size(recon_train_IKMOD,2)-1], mean(recon_train_IKSVD,1)+std(recon_train_IKSVD,1), '--o', 'Color', h1.Color, 'HandleVisibility', 'off','LineWidth',.01)

plot(newAtoms*[0:size(recon_train_IKMOD,2)-1], mean(recon_train_IKMOD,1)-std(recon_train_IKMOD,1), '--*', 'Color', h2.Color, 'HandleVisibility', 'off','LineWidth',.01)
plot(newAtoms*[0:size(recon_train_IKMOD,2)-1], mean(recon_train_IKMOD,1)+std(recon_train_IKMOD,1), '--*', 'Color', h2.Color, 'HandleVisibility', 'off','LineWidth',.01)

legend('IK-SVD', 'IDUO','location','southwest')
xlabel('Number of new atoms added to dictionary')
ylabel('Reconstruction error')
title('Reconstruction error on baseline set')
grid on

figure(5); clf; hold on;
h1 = plot(newAtoms*[0:size(recon_test_IKMOD,2)-1], mean(recon_test_IKSVD,1),'-o','LineWidth',2);
h2 = plot(newAtoms*[0:size(recon_test_IKMOD,2)-1], mean(recon_test_IKMOD,1),'-*','LineWidth',2);

plot(newAtoms*[0:size(recon_test_IKMOD,2)-1], mean(recon_test_IKSVD,1)-std(recon_test_IKSVD,1), '--o', 'Color', h1.Color, 'HandleVisibility', 'off','LineWidth',.01)
plot(newAtoms*[0:size(recon_test_IKMOD,2)-1], mean(recon_test_IKSVD,1)+std(recon_test_IKSVD,1), '--o', 'Color', h1.Color, 'HandleVisibility', 'off','LineWidth',.01)

plot(newAtoms*[0:size(recon_test_IKMOD,2)-1], mean(recon_test_IKMOD,1)-std(recon_test_IKMOD,1), '--*', 'Color', h2.Color, 'HandleVisibility', 'off','LineWidth',.01)
plot(newAtoms*[0:size(recon_test_IKMOD,2)-1], mean(recon_test_IKMOD,1)+std(recon_test_IKMOD,1), '--*', 'Color', h2.Color, 'HandleVisibility', 'off','LineWidth',.01)

legend('IK-SVD', 'IDUO','location','southwest')
xlabel('Number of new atoms added to dictionary')
ylabel('Reconstruction error')
title('Reconstruction error on in-situ set')
grid on

figure(6); clf; hold on;
h1 = plot(newAtoms*[0:size(recon_valid_IKMOD,2)-1], mean(recon_valid_IKSVD,1),'-o','LineWidth',2);
h2 = plot(newAtoms*[0:size(recon_valid_IKMOD,2)-1], mean(recon_valid_IKMOD,1),'-*','LineWidth',2);

plot(newAtoms*[0:size(recon_valid_IKMOD,2)-1], mean(recon_valid_IKSVD,1)-std(recon_valid_IKSVD,1), '--o', 'Color', h1.Color, 'HandleVisibility', 'off','LineWidth',.01)
plot(newAtoms*[0:size(recon_valid_IKMOD,2)-1], mean(recon_valid_IKSVD,1)+std(recon_valid_IKSVD,1), '--o', 'Color', h1.Color, 'HandleVisibility', 'off','LineWidth',.01)

plot(newAtoms*[0:size(recon_valid_IKMOD,2)-1], mean(recon_valid_IKMOD,1)-std(recon_valid_IKMOD,1), '--*', 'Color', h2.Color, 'HandleVisibility', 'off','LineWidth',.01)
plot(newAtoms*[0:size(recon_valid_IKMOD,2)-1], mean(recon_valid_IKMOD,1)+std(recon_valid_IKMOD,1), '--*', 'Color', h2.Color, 'HandleVisibility', 'off','LineWidth',.01)

legend('IK-SVD', 'IDUO','location','southwest')
xlabel('Number of new atoms added to dictionary')
ylabel('Reconstruction error')
title('Reconstruction error on generalization set')
grid on

