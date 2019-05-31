%% running yale stuff with KSVD
clear all
load('./yale_darkIS_darkMediumGen.mat')

params.L = 15;
params.K = length(dictClass);
params.numIteration = 14;
params.errorFlag = 0;
params.preserveDCAtom = 0;
params.InitializationMethod = 'DataElements';
% params.TrueDictionary = 1;
params.displayProgress = 1;

[dict_composite,output] = KSVD(dictSet, params);

%% Save the stuff
dict_composite_class = dictClass;
save('ksvd_dict2.mat','dict_composite','dict_composite_class');

%% Compare coefficients with ROMP
% xTrain_romp = RecursiveOMP(dict_composite, [], dictSet, .5);
% xTest_romp = RecursiveOMP(dict_composite, [], testSet, .5);