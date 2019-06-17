function [accuracy reconstructions] = classifyMSC(dictionaries, samples, classes)
%classifyMSC Calculates the class based on matched subspace classification
%   Calculates the class based on matched subspace classification as
%   presented in a paper by Jack Hall.  Infers the number of classes from
%   the size dictionaries, a cell aray.  Calculates sparse codes of each 
%   sample to each dictionarythe error between the sample and 
%   reconstruction for all classes, and chooses the minimum error as the 
%   estimated class.
%
%   dicionaries - cell array with each cell being a dictionary matrix with
%   column atoms
%   samples - samples matrix with columns as samples
%   classes - true classes of the samples
%   accuracy - the correct classification rate
%   reconstructions - sparse coded reconstructions of samples

% get number of classes from dictionaries
errorGoal = .1;

dict = horzcat(dictionaries{:});
numClasses = length(dictionaries);
numSamples = size(samples, 2);
errors = zeros(numClasses, numSamples);
reconstructions = zeros(size(samples,1),numSamples);
codes = cell(numClasses,1);
for class=1:numClasses
    %         display(['Calculating error for class: ' num2str(class)])
    codes{class} = RecursiveOMP(dictionaries{class}, [], samples, errorGoal);
    errors(class,:) = vecnorm(samples - dictionaries{class}*codes{class});
end
[~, minErrors] = min(errors,[],1);
accuracy = nnz(minErrors == classes)/numSamples;

for sample=1:numSamples
    bestCodes = codes{minErrors(sample)};
    reconstructions(:,sample) = dictionaries{minErrors(sample)}*bestCodes(:,sample);
end

end

