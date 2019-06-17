%% running yale stuff with KSVD
clear all
load('./yale_darkIS_darkMediumGen.mat')

params.K = floor(length(dictClass)/4);
params.L = 8;%5;%15;
params.numIteration = 50;
params.errorFlag = 0;
params.preserveDCAtom = 0;
params.InitializationMethod = 'DataElements';
% params.TrueDictionary = 1;
params.displayProgress = 1;

[dict_composite,output] = KSVD(dictSet, params);

%% Save the stuff
dict_composite_class = dictClass;
save('ksvd_dict4.mat','dict_composite','dict_composite_class');

%% Compare coefficients with ROMP
% xTrain_romp = RecursiveOMP(dict_composite, [], dictSet, .5);
% xTest_romp = RecursiveOMP(dict_composite, [], testSet, .5);