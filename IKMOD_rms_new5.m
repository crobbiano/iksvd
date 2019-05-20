%% IK-MOD.m
function [DictionaryNew, output] = IKMOD_rms_new5(DataNew, DataOld, DictionaryOld, param, coding)
%% subfunction for Method.A-FastOMP

%% 1) Method introduction:
% The new data samples are introduced into the learning process atom by atom. In the computation, the model mainly focuses on the incremental parts that are
% hard to sparsely decompose using the existed dictionary of the last
% iteration. New atoms will be added one by one for new data samples based
% on "ksvd-learning-v4" by Dr.Azimi. ( FINAL VERSION using r_i^{old} )
%% 3) The dictionary learning algorithm:
% -3.1) The existed data samples are expressed as Y1, and they have been
% trained by K-SVD, and the initial dictionary D1={d1,...,dn} has been
% obtained.
% -3.2) Dictionary update can be reference by "ksvd-learning-v4" by Dr.Azimi.

%% 4) INPUT ARGUMENTS:
% 4.1) DataNew: the new samples, which are used to add atoms one by one to
% the DictionaryOld.
% 4.2) DataOld: the old samples, whose sparse representation error can be
% used as the threshold of convergence of the new atoms adding.
% 4.2) DictionaryOld: the old dictionary that has been trained by the
% former data samples.
% 4.3) param: structure that includes all required parameters for the IK-SVD execution. Required fields:
%  - K1: The maximum number of the new dictionary atoms to train
%  - K1new: number of the new dictionary atoms that have been obtained currently.
%  - numIteration: number of iterations to perform for the added dictionary training.
%  - preserveDCAtom: if =1 then the first atom in the dictionary is set to be constant, and does not
%    ever change. This might be useful for working with natural images (in this case, only param.K-1
%    atoms are trained). *** to be decieded ***
%  - InitializationMethod: method to initialize the dictionary, can be one of the following
%    arguments: 1) 'DataElements' (initialization by the signals themselves),2) 'GivenMatrix'
%    (initialization by a given matrix param.initialDictionary), 3) 'MI',
%    mutual information-based creterion, and select K1 samples with the
%    largest entropy as the initial value of the new atoms.
%  - initialDictionary (optional, see InitializationMethod): if the initialization method is
%    'GivenMatrix', this is the matrix that will be used.
%  - TrueDictionary (optional): if specified, in each iteration the difference between this
%    dictionary and the trained one is measured and displayed.
%  - displayProgress: if =1 progress information is displyed. If coding.errorFlag==0, the average
%    repersentation error (RMSE) is displayed, while if coding.errorFlag==1, the average number of
%    required coefficients for representation of each signal is displayed.
%  - DataNew_RefineFlag: if =1 the part of the new samples that can be
%    sparsely coded by the old dictionary will be removed when the new
%    dictionary is trained.

% *** parameters not defined in original script ***
%    - minFracObs: min % of observations an atom must contribute to, else it is replaced
%    - maxIP: maximum inner product allowed betweem atoms, else it is replaced

% coding: structure containing parameters related to sparse coding stage of K-SVD algorithm
%  - method: method used for sparse coding. Can either be 'MP' or 'BP' for matching pursuit and
%    basis pursuit, respectively.
%  - errorFlag: For MP: if =0, a fix number of coefficients is used for representation of each
%    signal. If so, coding.L must be specified as the number of representing atom. If =1, arbitrary
%    number of atoms represent each signal, until a specific representation error is reached. If so,
%    coding.errorGoal must be specified as the allowed error. For BP: if =0, then the solution must
%    be exact, otherwise BP denoising (with error tolerance) is used.
%  - L(optional, see errorFlag) maximum coefficients to use in OMP coefficient calculations.
%  - errorGoal(optional, see errorFlag): allowed representation error in representing each signal.
%  - denoise_gamma = parameter used for BP denoising that controls the tradeoff between
%    reconstruction accuracy and sparsity in BP denoising.

% =========================================================================
%% 5) OUTPUT ARGUMENTS:
%  DictionaryIncremental         The incremently Added dictionary of size nK1(param.K1).
%  output                      Struct that contains information about the current run. It may include the following fields:
%    CoefMatrix                  The final coefficients matrix (it should
%                                hold that Data equals approximately Dictionary*output.CoefMatrix,
%                                where Dictionary=[DictionaryOld DictionaryIncremental]
%    ratio                       If the true dictionary was defined (in
%                                synthetic experiments), this parameter holds a vector of length
%                                param.numIteration that includes the detection ratios in each
%                                iteration).
%    totalerr                    The total representation error after each
%                                iteration (defined only if
%                                param.displayProgress=1 and
%                                coding.errorFlag = 0).
%    numCoef                     A vector of length param.numIteration that
%                                include the average number of coefficients required for representation
%                                of each signal (in each iteration) (defined only if
%                                param.displayProgress=1 and
%                                coding.errorFlag = 1)
%   idx_ObsvRemoved              The index of the new samples that can be
%                                sparsely coded well by the old dictionarys
%                                and this part of new samples will not be
%                                trained to generate the new dictionary.
%
%% 6) DictionaryNew:
% The generated new dictionary including the old dictionary and the new
% atoms added.
% =========================================================================

%****************************
%% Populate Necessary Fields*
%****************************

if (~isfield(param,'displayProgress'))
    param.displayProgress = 0;
end

if (isfield(coding,'errorFlag')==0)
    coding.errorFlag = 0;
end

if (isfield(param,'TrueDictionary'))
    displayErrorWithTrueDictionary = 1;
    ErrorBetweenDictionaries = zeros(param.numIteration+1,1);
    ratio = zeros(param.numIteration+1,1);
else
    displayErrorWithTrueDictionary = 0;
    ratio = 0;
end

if (param.preserveDCAtom>0)
    FixedDictionaryElement(1:size(DataNew,1),1) = 1/sqrt(size(DataNew,1));
else
    FixedDictionaryElement = [];
end

if param.displayProgress
    if ~coding.errorFlag
        output.totalerr = zeros(param.numIteration, 1);
    else
        output.numCoef = zeros(param.numIteration, 1);
    end
end


%*******************
%% (1) Remove the new samples those can be represented well by the old
%  dictionary.
DataNew0 = DataNew;
DictionaryOld0 = DictionaryOld;
if param.DataNew_RefineFlag
    idx_ObsvRemoved = [];
    DataRefined = [];
    [DataRefined idx_ObsvRemoved CoefMatrixDataNew] = DataRemove(DataNew, DictionaryOld, param, coding);
    DataNew = DataRefined;
    output.idx_ObsvRemoved = idx_ObsvRemoved;
end;

if isempty(DataNew)
    DictionaryNew = DictionaryOld; % All of the new samples can be sparsely coded by DictionaryOld well.
    return;
end;

%% (2) Add atoms one by one using OMP-Combined method.

%% (2.1) initialize errors
% the error of the old data
X_old = OMP(DictionaryOld,DataOld, coding.L, coding.errorGoal);
R_old = DataOld-DictionaryOld*X_old;
R_old_rms = sqrt(mean(sum(R_old.*R_old)));

Para_Xold_ = struct('X','Q','Sel_ind','r');
Para_Xold = OMP_DicOld(DictionaryOld,DataNew, coding.L, coding.errorGoal);
R_old = zeros(size(DataNew));
for k = 1:size(DataNew,2)
    R_old(:,k) = Para_Xold.r{k};
end;

R_new_all_rms = [];
R_new_all_rms = [sqrt(mean(sum(R_old.*R_old)))];
ResError_New_test00 = [];
ResError_New_test00 = R_new_all_rms;

R_diff = 100;
loop_num = 0; % records the number of the new atomes to be added.
fprintf('ResError_Old= %f\n',R_old_rms);

output.dk_update = [];
%% (2.2) Adding atoms
while R_diff>R_old_rms*0.0 && loop_num<param.K1 % && ResError_diff_all>0.000 %(isempty(DataNew)==0) % && abs(ResError(end)-ResError(end-1))>0.05*coding.errorGoal)
    loop_num = loop_num+1;
    
    
    %% (2.2.)iteratively get the new atom to be added.
    if (strcmp(param.InitializationMethod,'GivenMatrix'))
        d_new = ones(size(DataNew,1),1); %randn(size(DataNew,1),1); %  mean(r_i_old,2); %initialize d_new randomly.
    elseif (strcmp(param.InitializationMethod,'IM'))
        paramIM = param;
        paramIM.K1 = 1;
        d_new = IM(DataNew, DictionaryOld, paramIM,coding);
    end
    d_new = d_new/norm(d_new);
    
    %% (2.3) obtain X by fast OMP
    Para_Xnew = struct('X','Q','Sel_ind','r');
    Para_Xnew = OMP_NewAtom(DictionaryOld, DataNew,coding.L, coding.errorGoal,Para_Xold,d_new);
    % get R_new
    R_new = zeros(size(DataNew));
    X_new = zeros(1,size(DataNew,2));
    for k = 1:size(DataNew,2)
        X_k = Para_Xnew.X{k};
        X_new(k) = X_k(end);
        R_new(:,k) = Para_Xnew.r{k}+X_k(end)*d_new;
    end;
    
    
    %% (2.4) X_{k+1} and d_new
    iterNum = 1;
    R_new_temp_rms = 100;
    AtomError_diff12 = 1;
    while iterNum<10 && R_new_temp_rms>param.MOD_Err && AtomError_diff12>param.MOD_diffErr        
 
        NewAtom0 = d_new;
        d_new = (R_new*X_new')/sqrt(X_new*(R_new')*R_new*X_new');
        X_new = d_new'*R_new; % the coefficients of the new samples cooresponding to the new atom added.
        
        R_new_all = R_new-d_new*X_new;
        
        R_new_temp_rms = sqrt(mean(sum(R_new_all.*R_new_all)));  % mean of norm of representation error of the new samples.
        AtomError_diff12 = 1-abs(d_new'*NewAtom0); % the distance between the new atom to be added of this iteration and that of last iteration.
        R_new_all_rms = [R_new_all_rms R_new_temp_rms]; % records the variation of the mean of norm of representation error of the new samples.
        iterNum = iterNum+1;
    end; % iterNum
    
    DictionaryTemp = [DictionaryOld d_new];
    if strcmp(coding.method, 'MP')
        if ~coding.errorFlag
            CoefMatrix = OMP(DictionaryTemp,[DataOld DataNew0], coding.L, coding.errorGoal);
        else
            CoefMatrix = OMP(DictionaryTemp,[DataOld DataNew0], coding.L, coding.errorGoal);
        end
        
    elseif strcmp(coding.method, 'BP')
        if ~coding.errorFlag
            CoefMatrix = basisPursuit(DictionaryTemp, [DataOld DataNew0], 'exact',...
                [], [], false);
        else
            CoefMatrix = basisPursuit(DictionaryTemp, [DataOld DataNew0], 'denoise',...
                coding.errorGoal, coding.denoise_gamma, false);
        end
    end
    %% Stop iteration if convergences
    output.totalerr(loop_num) = sqrt(sum(sum(([DataOld DataNew0]-DictionaryTemp*CoefMatrix).^2)) / size([DataOld DataNew0], 2));
    output.totalerrNew(loop_num) = sqrt(sum(sum((DataNew0-DictionaryTemp*CoefMatrix(:,size(DataOld,2)+1:end)).^2)) / size(DataNew0, 2));
    output.totalerrOld(loop_num) = sqrt(sum(sum((DataOld-DictionaryTemp*CoefMatrix(:,1:size(DataOld,2))).^2)) / size(DataOld, 2));
    output.dk_update = [output.dk_update d_new];

    if isempty(find(X_new))
        break;
    end;
        
    Para_Xnew = OMP_NewAtom(DictionaryOld, DataNew,coding.L, coding.errorGoal,Para_Xold,d_new);
    DictionaryOld = [DictionaryOld d_new];
    Para_Xold = Para_Xnew;
    
    %% evulate the representation error of the new samples when a new atom has been finished adding
    CoefMatrixData = OMP(DictionaryOld, DataNew0, coding.L, coding.errorGoal);
    VecError_New = DataNew0-DictionaryOld*CoefMatrixData;
    
    ResError_New_test = sqrt(mean(sum(VecError_New.*VecError_New)));
    
    ResError_New_test00 = [ResError_New_test00 ResError_New_test];
    ResError_diff = mean(ResError_New_test00(end-min([2,length(ResError_New_test00)])+1:end)); %% the mean of means of the norms of the new samples corresponding to
    % the last 2 atomes adding to the dictionary, which is used as the convergence of new atomes adding.
    fprintf('ResError_diff= %f\n',ResError_diff);
    fprintf('iterNum_K= %d, iterNum= %d, loop_num=%d \n',0,iterNum,loop_num);
    
   
end;

if param.DictionaryIncrementalRefineFlag
    CoefMatrixDataAll = OMP(DictionaryOld,[DataOld DataNew0], coding.L, coding.errorGoal);
    for jj = size(DictionaryOld,2):-1:1 % run through all atoms (backwards since we may remove some)
        G = DictionaryOld'*DictionaryOld; G = G-diag(diag(G));
        if (max(abs(G(jj,:))) > param.maxIP) ||...
                ( (length(find(abs(CoefMatrixDataAll(jj,:)) > param.coeffCutoff)) / size(CoefMatrixDataAll,2)) <= param.minFracObs )
            DictionaryOld(:,jj) = []; % remove the atoms that are not useful
            CoefMatrixDataAll(jj,:) = []; % remove the coefficient row that corresponds to the unuseful atom
        end
    end
end;% if


%% evulate the representation error of all of the samples when the new dictionary has been updated and refined.

CoefMatrixDataAll = OMP(DictionaryOld,[DataOld DataNew0], coding.L, coding.errorGoal);
VecError_New = [DataOld DataNew0]-DictionaryOld*CoefMatrixDataAll;
ResError_New = sqrt(mean(sum(VecError_New.*VecError_New)));
% figure: plot(ResError_Data_newatom);grid; title('Error trend for per atom adding');
% figure:plot(sqrt(sum(VecError_New.*VecError_New)));grid;
% figure: plot(ResError);grid; title('error trend for all atoms adding');
DictionaryNew = DictionaryOld;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% RemoveData: Remove the part of data that can be sparsely coded by old dictionary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [DataRefined idx_ObsvRemoved CoefMatrix] = DataRemove(Data, Dictionary, param, coding)
%% 1) Method
% Calculate the error of OMP sparse coding by dictionary for each sample,
% and remove the samples that the errors are less than the threshold.

%% 2) input:
% Data: Samples which is a nXN matrix, where n is the dimension of
%   sample and N is the number of samples.
% Dictionary: the dictionary with nXK0, where n is the dimension of
%   atom and K0 is the number of atoms. Its columns MUST be normalized.
% param
%  - K1: number of the incremental dictionary atoms to be trained
%  - displayProgress: if =1 progress information is displyed. If coding.errorFlag==0, the average
%       repersentation error (RMSE) is displayed, while if coding.errorFlag==1, the average number of
%       required coefficients for representation of each signal is displayed.
% coding: structure containing parameters related to sparse coding stage of K-SVD algorithm
%  - method: method used for sparse coding. Can either be 'MP' or 'BP' for matching pursuit and
%    basis pursuit, respectively.
%  - errorFlag: For MP: if =0, a fix number of coefficients is used for representation of each
%    signal. If so, coding.L must be specified as the number of representing atom. If =1, arbitrary
%    number of atoms represent each signal, until a specific representation error is reached. If so,
%    coding.errorGoal must be specified as the allowed error. For BP: if =0, then the solution must
%    be exact, otherwise BP denoising (with error tolerance) is used.
%  - L(optional, see errorFlag) maximum coefficients to use in OMP coefficient calculations.
%  - errorGoal(optional, see errorFlag): allowed representation error in representing each signal.
%  - denoise_gamma = parameter used for BP denoising that controls the tradeoff between
%    reconstruction accuracy and sparsity in BP denoising.

%% 3) output:
% DataRefined: the data that has been removed partly
% idx_ObsvRemoved: the index of the samples that have been reomved.

if param.DataNew_RefineFlag
    
    %% 4) Sparse coding for Data using Dictionary via OMP or BP
    if strcmp(coding.method, 'MP')
        if ~coding.errorFlag
            CoefMatrix = OMP(Dictionary,Data, coding.L, coding.errorGoal);
        else
            CoefMatrix = OMP(Dictionary,Data, coding.L, coding.errorGoal);
        end
        
    elseif strcmp(coding.method, 'BP')
        if ~coding.errorFlag
            CoefMatrix = basisPursuit(Dictionary, Data, 'exact',...
                [], [], false);
        else
            CoefMatrix = basisPursuit(Dictionary, Data, 'denoise',...
                coding.errorGoal, coding.denoise_gamma, false);
        end
    end;
    
    %% 5) Determine the part of the new samples that can be sparsely coded well and be removed
    Residual_Error = zeros(size(Data,2),1);
    Residual_ErrorMatrix = Data-Dictionary*CoefMatrix;
    for i_sample = 1:size(Data,2)
        Residual_Error(i_sample) = norm(Residual_ErrorMatrix(:,i_sample));
    end;
    idx_ObsvRemoved = find(Residual_Error<coding.errorGoal);
    Data(:,idx_ObsvRemoved) = [];
    DataRefined = Data;
end;

%% IM function
function [SubDictionarySelected]= IM(Data, Dictionary, param, coding)
%% 1) Method
% Select param.K1 samples from Data that can not be sparsely reprensented
% by Dictionary via calculating the entropy for the sparse representation
% of each atom and selecting the K1 largest entropy.

%% 2) input:
% Data: Samples which is a nXN matrix, where n is the dimension of
%   sample and N is the number of samples.
% Dictionary: the dictionary with nXK0, where n is the dimension of
%   atom and K0 is the number of atoms. Its columns MUST be normalized.
% param
%  - K1: number of the incremental dictionary atoms to be trained
%  - displayProgress: if =1 progress information is displyed. If coding.errorFlag==0, the average
%       repersentation error (RMSE) is displayed, while if coding.errorFlag==1, the average number of
%       required coefficients for representation of each signal is displayed.
% coding: structure containing parameters related to sparse coding stage of K-SVD algorithm
%  - method: method used for sparse coding. Can either be 'MP' or 'BP' for matching pursuit and
%    basis pursuit, respectively.
%  - errorFlag: For MP: if =0, a fix number of coefficients is used for representation of each
%    signal. If so, coding.L must be specified as the number of representing atom. If =1, arbitrary
%    number of atoms represent each signal, until a specific representation error is reached. If so,
%    coding.errorGoal must be specified as the allowed error. For BP: if =0, then the solution must
%    be exact, otherwise BP denoising (with error tolerance) is used.
%  - L(optional, see errorFlag) maximum coefficients to use in OMP coefficient calculations.
%  - errorGoal(optional, see errorFlag): allowed representation error in representing each signal.
%  - denoise_gamma = parameter used for BP denoising that controls the tradeoff between
%    reconstruction accuracy and sparsity in BP denoising.

%% 3) output:
% SubDictionarySelected: the initialized incremental dictionary, which is a
% nXK1 matrix.

%% 4) Sparse coding for Data using Dictionary via OMP or BP
if strcmp(coding.method, 'MP')
    if ~coding.errorFlag
        CoefMatrix = OMP(Dictionary,Data, coding.L, coding.errorGoal);
    else
        CoefMatrix = OMP(Dictionary,Data, coding.L, coding.errorGoal);
    end
    
elseif strcmp(coding.method, 'BP')
    if ~coding.errorFlag
        CoefMatrix = basisPursuit(Dictionary, Data, 'exact',...
            [], [], false);
    else
        CoefMatrix = basisPursuit(Dictionary, Data, 'denoise',...
            coding.errorGoal, coding.denoise_gamma, false);
    end
end;

%% 5) Determine the part of the new samples that can be sparsely coded well and be removed

if size(Data,2)<=param.K1
    SubDictionarySelected = Data;
    SubDictionarySelected = SubDictionarySelected * diag(1./sqrt(sum(SubDictionarySelected.*SubDictionarySelected)));
else
    %% 6) Select K1 Data samples using MI-based method
    % 6.1) Calculate the information entropy for each Sample's Coefficient
    abs_CoefMatrix = abs(CoefMatrix);
    Sum_SampleCoef = sum(abs_CoefMatrix); % the sum of each column of CoefMatrix
    H_SampleCoef = zeros(size(Data,2),1); % the information entropy for each Sample's coefficient
    for i_sample = 1:size(Data,2)
        p_Coef = (abs_CoefMatrix(:,i_sample)/Sum_SampleCoef(i_sample));
        idx_nonzero = find(p_Coef);
        if isempty(idx_nonzero)
            H_SampleCoef(i_sample) = 0;
        else
            H_SampleCoef(i_sample) = (-1)*sum(p_Coef(idx_nonzero).*log(p_Coef(idx_nonzero))); % Shannon
        end;
    end;
    % 6.2) Select the K1 maximum Samples as the initial dictionary atoms
    [Sample_idx H_SampleCoefDescend] = sort(H_SampleCoef,'descend');
    DataSelected = Data(:,H_SampleCoefDescend(1:param.K1));
    
    % 6.3) Normalize the Dictionary
    SubDictionarySelected = zeros(size(Data,1),param.K1);
    SubDictionarySelected = DataSelected * diag(1./sqrt(sum(DataSelected.*DataSelected)));
end;
clear abs_CoefMatrix;
SubDictionarySelected = SubDictionarySelected .* repmat(sign(SubDictionarySelected(1,:)),size(SubDictionarySelected,1),1); % multiply in the sign of the first element.

