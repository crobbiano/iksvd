%% Test IK-SVD against Azimi's IK-SVD.
% Use the learned KSVD dictionaries from other script as the baseline
% dictionary.  Choose parameters for the IK-SVD (number of atoms to learn,
% etc), and partition incremental training data into some number of
% disjoint sets.  Then do the learning, and at each step, evaluate the CCR
% using normal SR classification.
for runNum=1:10
    clearvars -except runNum
    load('./yale_darkIS_darkMediumGen.mat')
    load('./ksvd_dict_multiclass.mat')
    %     load('./ksvd_dict3.mat')
    doAzimi = 0;  
    %% Train an SVM for classification
    % When doing prediction on training and testing sets, first train the SVM
    % on the reconstructed samples using the sparse codes and KSVD dictionary
    useSVM = 0;
    if useSVM
        % Compare coefficients with ROMP
        xTrain_romp = RecursiveOMP(dict_composite, [], trainSet, .1);
        xTest_romp = RecursiveOMP(dict_composite, [], testSet, .1);
        xValid_romp = RecursiveOMP(dict_composite, [], validSet, .1);
        % xTrain_romp = RecursiveOMP(dict_ksvd, [], trainSet, .2);
        % xTest_romp = RecursiveOMP(dict_ksvd, [], testSet, 1);
        
        model = svmtrain(dictClass', [xTrain_romp]', '-s 1 -n .5');
        [predicted_label, accuracy, decisions] = svmpredict(trainClass', [xTrain_romp]', model );
        accs.acc_train(1) = accuracy(1);
        [predicted_label, accuracy, decisions] = svmpredict(testClass', [xTest_romp]', model );
        accs.acc_test(1) = accuracy(1);
        [predicted_label, accuracy, decisions] = svmpredict(validClass', [xValid_romp]', model );
        accs.acc_valid(1) = accuracy(1);
        
        % Check the reconstruction errors
        recon_train = trainSet - dict_composite*xTrain_romp;
        recon_test = testSet - dict_composite*xTest_romp;
        recon_valid = validSet - dict_composite*xValid_romp;
    end
    
    %% Do MSC
    useMSC = 1;
    if useMSC
        [accs.acc_train(1), recon_train] = classifyMSC(dicts, dictSet, dictClass);
        [accs.acc_test(1), recon_test] = classifyMSC(dicts, testSet, testClass);
        [accs.acc_valid(1), recon_valid] = classifyMSC(dicts, validSet, validClass);
        
        % Check the reconstruction errors
        recon_train = trainSet - recon_train;
        recon_test = testSet - recon_test;
        recon_valid = validSet - recon_valid;
    end
    %% Check the reconstruction errors   
    recons.recon_err_train(1) = mean(vecnorm(recon_train));
    recons.recon_err_test(1) = mean(vecnorm(recon_test));
    recons.recon_err_valid(1) = mean(vecnorm(recon_valid));
    %% Do IKSVD
    % split test set into disjoint
    batchSize = 50;
    numBatches = floor(length(testClassSmall)/batchSize);
    available = 1:length(testClassSmall);
    seenSamples = [];
    seenClass = [];
    for smallidx = 1:numBatches
        display(['Batch: ' num2str(smallidx)]);
        random_sample = randperm(length(available), batchSize);
        selected_samples = available(random_sample);
        available(random_sample) = []; % remove added sample from those available to add
        
        testTemp = testSetSmall(:, selected_samples);
        testTempClass = testClassSmall(selected_samples);
        
        % Need to sort the samples
        [testTempClass, sidxs] = sort(testTempClass);
        testTemp = testTemp(:,sidxs);
        
        %         params.K1 = max([floor(batchSize/10), 1]);
        params.K1 = 1;
        params.numIteration = 5;
        params.preserveDCAtom = 0;
        %     params.InitializationMethod = 'DataElements';
        params.InitializationMethod = 'GivenMatrix';
        params.displayProgress = 1;
        params.DataNew_RefineFlag = 0;
        params.DictionaryIncrementalRefineFlag = 0;
        params.coeffCutoff = 16;
        params.MOD_Err = .5;
        params.MOD_diffErr = .5;
        %     params.errorFlag = 0;
        coding.method = 'MP';
        coding.L = 5;%16;
        coding.errorFlag = 0;
        coding.errorGoal = .5;
        
        % need to go class by class and add the recent samples to the dicts
        numClasses = length(dicts);
        for c = 1:numClasses
            params.initialDictionary = dicts{c};
            tempSamples = testTemp(:,testTempClass==c);
            if ~isempty(tempSamples)
                if doAzimi
                    type = 'ikmod';
                    [dicts{c}, output] = IKMOD_rms_new5(tempSamples, [trainSet(:, trainClass == c) seenSamples(:, seenClass == c)], dicts{c}, params, coding);
                else
                    type='iksvd';
                    [newDict, output] = IKSVD(tempSamples, dicts{c}, params, coding);
                    % Add the new column to the dictionary
                    dicts{c} = [dicts{c}, newDict];
                end
            end
        end
        
        seenSamples = [seenSamples testTemp];
        seenClass = [seenClass testTempClass];
        
        if useSVM
            % Check the classification abilities
            xTrain_romp = RecursiveOMP(dict_composite, [], trainSet, .1);
            xTest_romp = RecursiveOMP(dict_composite, [], testSet, .1);
            xTestTemp_romp = RecursiveOMP(dict_composite, [], seenSamples, .1);
            xValid_romp = RecursiveOMP(dict_composite, [], validSet, .1);
            
            model = svmtrain([dictClass seenClass]', [xTrain_romp xTestTemp_romp]', '-s 1 -n .5');
            [predicted_label, accuracy, decisions] = svmpredict(trainClass', [xTrain_romp]', model );
            accs.acc_train(end+1) = accuracy(1);
            [predicted_label, accuracy, decisions] = svmpredict(testClass', [xTest_romp]', model );
            accs.acc_test(end+1) = accuracy(1);
            [predicted_label, accuracy, decisions] = svmpredict(validClass', [xValid_romp]', model );
            accs.acc_valid(end+1) = accuracy(1);
            
            % Check the reconstruction errors
            recon_train = trainSet - dict_composite*xTrain_romp;
            recon_test = testSet - dict_composite*xTest_romp;
            recon_valid = validSet - dict_composite*xValid_romp;
        end
        
        if useMSC
            [accs.acc_train(end+1), recon_train] = classifyMSC(dicts, dictSet, dictClass);
            [accs.acc_test(end+1),  recon_test]  = classifyMSC(dicts, testSet, testClass);
            [accs.acc_valid(end+1), recon_valid] = classifyMSC(dicts, validSet, validClass);
            
            % Check the reconstruction errors
            recon_train = trainSet - recon_train;
            recon_test = testSet - recon_test;
            recon_valid = validSet - recon_valid;
        end
        
        
        recons.recon_err_train(end+1) = mean(vecnorm(recon_train));
        recons.recon_err_test(end+1)  = mean(vecnorm(recon_test));
        recons.recon_err_valid(end+1) = mean(vecnorm(recon_valid));
    end
    save(['results' filesep type '_res' num2str(runNum) '.mat'], 'recons', 'accs', 'dict_composite', 'batchSize', 'params', 'coding')
end
%% Plot things
figure(889);
clf
hold on
plot(accs.acc_train)
plot(accs.acc_test)
plot(accs.acc_valid)
legend('Train','Test','Validation')
xxticks = (xticks)*batchSize;
xticklabels(xxticks)
%% SRC - FIXME not working
trySRC = 0;
if trySRC
    % currSets = testSet;
    % currClasses = testClass;
    % currCodes = xTest_romp;
    currSets = trainSet;
    currClasses = trainClass;
    currCodes = xTrain_romp;
    numClasses = length(unique(currClasses));
    for idx = 1:length(currClasses)
        currSamp = currSets(:,idx);
        currClass = currClasses(idx);
        currCode = currCodes(:,idx);
        
        classErrs = zeros(numClasses, 1);
        availIdxs = 1:length(currCode);
        for class = 1:numClasses
            currDict = dicts{class}.dict;
            dictSize = size(currDict,2);
            classCode = currCode(availIdxs(1:dictSize));
            
            classErrs(class) = norm(currSamp - currDict*classCode);
            availIdxs(1:dictSize) = [];
        end
        [~, minIdx] = min(classErrs);
        predictedClass(idx) = minIdx;
    end
    
    correct = nnz(predictedClass == currClasses);
    disp(['Correct %: ' num2str(correct/length(currClasses))])
end