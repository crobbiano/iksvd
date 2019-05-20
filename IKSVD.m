%% IK-SVD.m
function [DictionaryIncremental, output] = IKSVD(Data, DictionaryOld, param, coding)
%% 1) Ref. 
% IK-SVD: Dictionary Learning for Spatial Big Data via Incremental Atom Update. by L.Wang, K.Lu, P.Liu, R.R.I and L.Chen
% IEEE Computing in Science & Engineering. 2004
%% 2) Method introduction:
% The new data samples are introduced into the learning process group by group. In the computation, the model mainly focuses on the incremental parts that are
% hard to sparsely decompose using the last dictionary of the last iteration. New atoms will be added for current data samples, and an active learning scheme based 
% on maximum mutual information is employed to determine the initial value
% of the new atoms.
%% 3) The dictionary learning algorithm:
% -3.1) The existed data samples are expressed as Y1, and they have been
% trained by K-SVD, and the initial dictionary D1={d1,...,dn} has been
% obtained. Set J=1, where J represents the Jth iteration for the new sample subset.
% -3.2) Solve the object function Eq.(14), select m samples based on
% Eq.(16), and D(J) = D(J-1) U {dn+1,...,dn+m}.
% -3.3) For the sparse coding stage, use the OMP algorithm to compute the
% representation Ys by the solution of Eq.(20).
% -3.4) In the atoms-update stage, for each new atom dk in dictionary
% D(J),where k=n+1,...,n+m, update it as follows: Define the group of
% examples that use this atom dk, wk={i|1<=i<=r, alpha_t^k(i)!=0}. Compute
% the overall representation error matrix by Eq.(11), and get E_s^k. Next
% construct hat{E}_s^k by choosing only the columns corresponding to wk
% within E_s^k. Apply SVD decomposition hat{E}_s^k=U \Lambda V^T and choose
% the first column of U to be the updated atom dk. Update the coefficient
% vector alpha_T^k to be the first column of V multiplied \Lambda(1,1).
% -3.5) Update the dictionary.
% -3.6) Set J=J+1. Repeat steps 3 through 6 until convergence.

%% 4) INPUT ARGUMENTS:
% 4.1) Data: the new samples added, which is nXN matrix that contains N signals (Y), each of dimension n. 
% 4.2) DictionaryOld: the old dictionary that has been trained by the
% former data samples, which is nXK0 matrix that contains K0 atoms (d),
% each of dimension n.
% 4.3) param: structure that includes all required parameters for the IK-SVD execution. Required fields:
%  - K1: initial number of the new dictionary atoms to train 
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
    FixedDictionaryElement(1:size(Data,1),1) = 1/sqrt(size(Data,1));
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
%% (1) Initialize Dictionary *
%  (1.1) Remove the new samples those can be represented well by the old
%  dictionary.
if param.DataNew_RefineFlag
    idx_ObsvRemoved = [];
    DataRefined = [];
    [DataRefined idx_ObsvRemoved] = DataRemove(Data, DictionaryOld, param, coding);
    Data = DataRefined;
    output.idx_ObsvRemoved = idx_ObsvRemoved;
end;
%  (1.2) Initialize Dictionary
if (size(Data,2) < param.K1-param.preserveDCAtom)
    disp('Size of data is smaller than the dictionary size. Trivial solution...');
    DictionaryIncremental = Data;
%     DictionaryIncremental = DictionaryIncremental * diag(1./sqrt(sum(DictionaryIncremental.*DictionaryIncremental)));
%     DictionaryIncremental = DictionaryIncremental .* repmat(sign(DictionaryIncremental(1,:)),size(DictionaryIncremental,1),1); % multiply in the sign of the first element.    
%     return;
elseif (strcmp(param.InitializationMethod,'DataElements'))
    DictionaryIncremental = Data(:, randsample(size(Data, 2), param.K1-param.preserveDCAtom)); %% select param.K1-param.preserveDCAtom) columns from Data 
elseif (strcmp(param.InitializationMethod,'GivenMatrix'))
    DictionaryIncremental = param.initialDictionary(:, 1:param.K1-param.preserveDCAtom); %% select param.K1-param.preserveDCAtom) columns from initial dictionary inputed
elseif (strcmp(param.InitializationMethod,'IM'))
    [DictionaryIncremental] = IM(Data, DictionaryOld, param,coding);
end

% *** reduce the components in Dictionary that are spanned by the fixed elements ***
if param.preserveDCAtom %% confused
    tmpMat = FixedDictionaryElement \ DictionaryIncremental;
    DictionaryIncremental = DictionaryIncremental - FixedDictionaryElement*tmpMat;
end

% *** normalize the dictionary ***
DictionaryIncremental = DictionaryIncremental * diag(1./sqrt(sum(DictionaryIncremental.*DictionaryIncremental)));
DictionaryIncremental = DictionaryIncremental .* repmat(sign(DictionaryIncremental(1,:)),size(DictionaryIncremental,1),1); % multiply in the sign of the first element.

if strcmp(coding.method, 'MP')
    if ~coding.errorFlag
        CoefMatrix = OMP([DictionaryOld,FixedDictionaryElement,DictionaryIncremental],Data, coding.L, coding.errorGoal);
    else
        CoefMatrix = OMP([DictionaryOld,FixedDictionaryElement,DictionaryIncremental],Data, coding.L, coding.errorGoal);
    end
    
elseif strcmp(coding.method, 'BP')
    if ~coding.errorFlag
        CoefMatrix = basisPursuit([DictionaryOld,FixedDictionaryElement,DictionaryIncremental], Data, 'exact',...
            [], [], false);
    else
        CoefMatrix = basisPursuit([DictionaryOld,FixedDictionaryElement,DictionaryIncremental], Data, 'denoise',...
            coding.errorGoal, coding.denoise_gamma, false);
    end
end

%% (2) Sparse coding and Updating dictionary
for iterNum = 1:param.numIteration
   
   
    % *** update dictionary one atom at a time (using function I_findBetterDictionaryElement) ***
    replacedVectorCounter = 0;
% 	rPerm = size(DictionaryOld,2)+(1:param.K1); %randperm(size(Dictionary,2));
    rPerm = size(DictionaryOld,2)+(1:size(DictionaryIncremental,2));
    for j = rPerm % iteration
        j_incremental = j-size(DictionaryOld,2);
        [betterDictionaryElement,CoefMatrix,addedNewVector] = I_findBetterDictionaryElement(Data,...
            [DictionaryOld,FixedDictionaryElement,DictionaryIncremental],j, CoefMatrix);
        DictionaryIncremental(:,j_incremental) = betterDictionaryElement;
        if (param.preserveDCAtom)
            tmpCoef = FixedDictionaryElement\betterDictionaryElement;
            DictionaryIncremental(:,j_incremental) = betterDictionaryElement - FixedDictionaryElement*tmpCoef;
            DictionaryIncremental(:,j_incremental) = DictionaryIncremental(:,j_incremental)./sqrt(DictionaryIncremental(:,j_incremental)'*DictionaryIncremental(:,j_incremental));
        end
        replacedVectorCounter = replacedVectorCounter+addedNewVector;
        DictionaryIncremental(:,j_incremental) = sign(DictionaryIncremental(1,j_incremental))*DictionaryIncremental(:,j_incremental);
    end    
    
    % *** condition dictionary (remove redundencies, etc.) ***
    if param.DictionaryIncrementalRefineFlag
        param.K1new = size([FixedDictionaryElement,DictionaryIncremental],2);
        DictionaryIncremental = I_clearDictionaryIncremental([DictionaryOld,FixedDictionaryElement,DictionaryIncremental], CoefMatrix, Data, param);
    end;

    % *** find the coefficients using sparse coding ***
    if strcmp(coding.method, 'MP')
        if ~coding.errorFlag
            CoefMatrix = OMP([DictionaryOld,FixedDictionaryElement,DictionaryIncremental],Data, coding.L, coding.errorGoal);
        else
            CoefMatrix = OMP([DictionaryOld,FixedDictionaryElement,DictionaryIncremental],Data, coding.L, coding.errorGoal);
        end
        
    elseif strcmp(coding.method, 'BP')
        if ~coding.errorFlag
            CoefMatrix = basisPursuit([DictionaryOld,FixedDictionaryElement,DictionaryIncremental], Data, 'exact',...
                [], [], false);
        else
            CoefMatrix = basisPursuit([DictionaryOld,FixedDictionaryElement,DictionaryIncremental], Data, 'denoise',...
                coding.errorGoal, coding.denoise_gamma, false);
        end
    end   
    
    
    % *** display progress in terms of alternate constraint (# coefficients or error) ***
        output.totalerr(iterNum) = sqrt(sum(sum((Data-[DictionaryOld,FixedDictionaryElement,DictionaryIncremental]*CoefMatrix).^2)) / size(Data, 2)); %% total error after each sparse coding for each sample
        output.numCoef(iterNum) = length(find(abs(CoefMatrix) >= param.coeffCutoff)) / size(Data,2); %% sparsity of Coeffecient matix for each sample
        max_IP = max(max([DictionaryOld,FixedDictionaryElement,DictionaryIncremental]'*[DictionaryOld,FixedDictionaryElement,DictionaryIncremental] - eye(size([DictionaryOld,FixedDictionaryElement,DictionaryIncremental], 2)))); %% maximum inner product between each pair of the dictionary atoms
        percent_obs = 100*min(sum(abs(CoefMatrix) >= param.coeffCutoff, 2) / size(CoefMatrix, 2)); %% the dictionary atom that has been used by sparse coding with minimum probability.
     if param.displayProgress

        disp(['Iter: ' num2str(iterNum) '  Avg. err: ' num2str(output.totalerr(iterNum)) ...
            '  Avg. # coeff: ' num2str(output.numCoef(iterNum)) '  Max IP: ' num2str(max_IP)...
            '  Min % obs: ' num2str(percent_obs) '%']);
    end
    
    if displayErrorWithTrueDictionary 
        [ratio(iterNum+1),ErrorBetweenDictionaries(iterNum+1)] = I_findDistanseBetweenDictionaries(param.TrueDictionary,[DictionaryOld,FixedDictionaryElement,DictionaryIncremental]);
%         disp(strcat(['Iteration: ', num2str(iterNum),' ratio of restored elements: ',num2str(ratio(iterNum+1))]));
        output.ratio = ratio;
    end    
    
    %% Stop iteration if convergences
    if strcmp(coding.method, 'MP')
        if ~coding.errorFlag
            if output.numCoef(iterNum)<=coding.L 
                output.CoefMatrix = CoefMatrix;
                break ;
            end;
        else
            if output.totalerr(iterNum)<=coding.errorGoal
                output.CoefMatrix = CoefMatrix;
                break;
            end;
        end
        
    elseif strcmp(coding.method, 'BP')
        continue; %% no coding for BP
    end    
    
    
end;

% *** remove atoms that are not as useful ***
if param.DictionaryIncrementalRefineFlag
    for jj = size(DictionaryIncremental,2):-1:1 % run through all atoms (backwards since we may remove some)
        G = DictionaryIncremental'*DictionaryIncremental; G = G-diag(diag(G));
        if (max(abs(G(jj,:))) > param.maxIP) ||...
                ( (length(find(abs(CoefMatrix(jj+size([DictionaryOld,FixedDictionaryElement],2),:)) > param.coeffCutoff)) / size(CoefMatrix,2)) <= param.minFracObs )
            DictionaryIncremental(:,jj) = []; % remove the atoms that are not useful
            CoefMatrix(size(DictionaryOld,2)+param.preserveDCAtom+jj,:) = []; % remove the coefficient row that corresponds to the unuseful atom
        end
    end
end;
output.CoefMatrix = CoefMatrix;

DictionaryIncremental = [FixedDictionaryElement,DictionaryIncremental];




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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% RemoveData: Remove the part of data that can be sparsely coded by old dictionary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [DataRefined idx_ObsvRemoved] = DataRemove(Data, Dictionary, param, coding)
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
    idx_ObsvRemoved = find(Residual_Error<=coding.errorGoal);
    Data(:,idx_ObsvRemoved) = [];
    DataRefined = Data;
end;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  findBetterDictionaryElement
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [betterDictionaryElement,CoefMatrix,NewVectorAdded] = I_findBetterDictionaryElement(Data,Dictionary,j,CoefMatrix)

relevantDataIndices = find(CoefMatrix(j,:)); % the data indices that uses the j'th dictionary element.
if (length(relevantDataIndices)<1) %(length(relevantDataIndices)==0), the jth element is not used.
    ErrorMat = Data-Dictionary*CoefMatrix;
    ErrorNormVec = sum(ErrorMat.^2);
    [~,i] = max(ErrorNormVec);
    betterDictionaryElement = Data(:,i);%ErrorMat(:,i); %
    betterDictionaryElement = betterDictionaryElement./sqrt(betterDictionaryElement'*betterDictionaryElement);
    betterDictionaryElement = betterDictionaryElement.*sign(betterDictionaryElement(1));
    CoefMatrix(j,:) = 0;
    NewVectorAdded = 1;
    return;
end

NewVectorAdded = 0;
tmpCoefMatrix = CoefMatrix(:,relevantDataIndices); 
tmpCoefMatrix(j,:) = 0;% the coeffitients of the element we now improve are not relevant.
errors =(Data(:,relevantDataIndices) - Dictionary*tmpCoefMatrix); % vector of errors that we want to minimize with the new element

% better dictionary element and values of beta found using svd to approximate the matrix 'errors' 
% with a one-rank matrix.

[betterDictionaryElement,singularValue,betaVector] = svds(errors,1);
CoefMatrix(j,relevantDataIndices) = singularValue*betaVector';% *signOfFirstElem

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  findDistanseBetweenDictionaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ratio,totalDistances] = I_findDistanseBetweenDictionaries(original,new)
% first, all the column in oiginal starts with positive values.
catchCounter = 0;
totalDistances = 0;
for i = 1:size(new,2)
    new(:,i) = sign(new(1,i))*new(:,i);
end
for i = 1:size(original,2)
    d = sign(original(1,i))*original(:,i);
    distances =sum ( (new-repmat(d,1,size(new,2))).^2);
    [~,index] = min(distances);
    errorOfElement = 1-abs(new(:,index)'*d);
    totalDistances = totalDistances+errorOfElement;
    catchCounter = catchCounter+(errorOfElement<0.01);
end
ratio = 100*catchCounter/size(original,2);


%%  I_clearDictionary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function DictionaryIncremental = I_clearDictionaryIncremental(Dictionary, CoefMatrix, Data, param)
% *** replace atoms that:
% 1) exceed maximum allowed inner product with another atom
% 2) is used an insufficient number of times for reconstructing observations

Er = sum((Data-Dictionary*CoefMatrix).^2,1); % error in representation
dim_DicAll = size(Dictionary,2); 
for jj = (size(Dictionary,2)-param.K1new+1):size(Dictionary,2) % run through all atoms
%     Er = sum((Data-Dictionary*CoefMatrix).^2,1); % error in representation
    G = Dictionary'*Dictionary; G = G-diag(diag(G)); % matrix of inner products (diagonal removed)    
    if (max(abs(G(jj,:))) > param.maxIP) ||...
            ( (length(find(abs(CoefMatrix(jj,:)) > param.coeffCutoff)) / size(CoefMatrix,2)) <= param.minFracObs )
        [~, pos] = max(Er); % sorted indices of obseravtions with highest reconstruction errors
        
        % replace jj'th atom with normalized data vector with highest reconstruction error
        Er(pos(1)) = 0;
        Dictionary(:,jj) = Data(:,pos(1)) / norm(Data(:,pos(1)));
        Dictionary(:,jj) = sign(Dictionary(1,jj))*Dictionary(:,jj);
    end
end
DictionaryIncremental = Dictionary(:,(size(Dictionary,2)-param.K1new+1):size(Dictionary,2));
