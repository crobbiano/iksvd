%% Generate the baseline IKSVD dictionary.
% Create a subdictionary for each class and then create the composite
% dictionary that contains 'ordered' atoms
clear all
load('./yale_darkIS_darkMediumGen.mat')

numClasses = length(unique(dictClass));
availIndices = 1:length(dictClass);
dict_composite = []; 
for class=1:numClasses
    disp(['Class: ' num2str(class)])
    [~, idxs] = find(dictClass == class);
    numInClass = length(idxs);
    tempSet = dictSet(:,availIndices(1:numInClass));
    tempClass = dictClass(availIndices(1:numInClass));
    
    availIndices(1:numInClass) = [];
    params.L = ceil(numInClass/4);
%     params.errorGoal = .01;
    params.errorFlag = 0;
    params.K = ceil(numInClass/2);
    params.numIteration = 20;
    params.preserveDCAtom = 0;
    params.InitializationMethod = 'DataElements';
    % params.TrueDictionary = 1;
    params.displayProgress = 1;
    
%     [Dictionary] = tempSet; output = 0;
    [Dictionary,output] = KSVD(tempSet, params);
    dicts{class} = normc(Dictionary);
    info{class} = output;
    classes(class)= class; 
     
    dict_composite = [dict_composite Dictionary];
end 
%% Save the stuff
dict_composite_class = dictClass;
save('ksvd_dict_multiclass.mat','dict_composite','dicts');

%% Compare coefficients with ROMP
% xTrain_romp = RecursiveOMP(Dictionary, [], dictSet, .5);
% xTest_romp = RecursiveOMP(Dictionary, [], testSet, .5);