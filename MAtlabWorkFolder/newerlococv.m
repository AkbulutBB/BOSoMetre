%% Deep Learning with Leave-One-Patient-Out Cross-Validation
% This script performs LOPO CV on a preloaded dataset 
% 'BOSoMetreDataCSFDataForTraining'. It uses the predictor variables:
% RPerc, GPerc, BPerc, and CPerc to predict infection status.
%
% The classification type can be set to:
%   - 'binary' using infClassIDSA (0/1), or 
%   - 'triphasic' using triClassIDSA (0/1/2, where 2 indicates indeterminate).
%
% Missing (NaN) values in the predictors and response are removed.
%
% References:
%   - Varma, S., & Simon, R. (2006). Bias in error estimation when using 
%     cross-validation for model selection. BMC Bioinformatics, 7, 91.
%     https://doi.org/10.1186/1471-2105-7-91
%   - MathWorks. (n.d.). Train a Deep Learning Network for Tabular Data.
%     https://www.mathworks.com/help/deeplearning/ug/train-a-deep-learning-network-for-tabular-data.html

%% Preliminary Setup
clc;      % Clear the command window
clearvars -except BOSoMetreDataCSFDataForTraining

% Verify that the preloaded dataset exists in the workspace.
if ~exist('BOSoMetreDataCSFDataForTraining', 'var')
    error('BOSoMetreDataCSFDataForTraining is not found in the workspace.');
end

% Use the preloaded dataset.
data = BOSoMetreDataCSFDataForTraining;

%% Exclude Specific Patients
% Define the patient IDs to be excluded.
excludedPatients = [9, 17, 18, 19];

% Remove rows corresponding to the excluded patient IDs.
data = data(~ismember(data.InStudyID, excludedPatients), :);

%% Specify Classification Type and Preprocess Data
% Set classificationType to 'binary' (using infClassIDSA) or 'triphasic' 
% (using triClassIDSA).
classificationType = 'triphasic';  % Change to 'triphasic' if needed

% Define the predictor variables.
predictorNames = {'RPerc', 'GPerc', 'BPerc', 'CPerc'};

% Remove rows with missing values in the predictors and the chosen response.
if strcmpi(classificationType, 'binary')
    colsToCheck = [predictorNames, {'infClassIDSA'}];
elseif strcmpi(classificationType, 'triphasic')
    colsToCheck = [predictorNames, {'triClassIDSA'}];
end
data = rmmissing(data, 'DataVariables', colsToCheck);

% Select the response variable based on classification type.
switch lower(classificationType)
    case 'binary'
        Y = data.infClassIDSA;
    case 'triphasic'
        Y = data.triClassIDSA;
    otherwise
        error('Invalid classification type. Choose either "binary" or "triphasic".');
end

% Convert the response to categorical for deep learning.
Y = categorical(Y);
data.Response = Y;  % Add the response to the table for convenience

% (Optional) Extract predictors as a numeric matrix.
X = data{:, predictorNames};

%% Define the Deep Learning Network Architecture
% For tabular data, a feedforward (fully connected) network is typically used.
numFeatures = numel(predictorNames);
numClasses  = numel(categories(Y));

layers = [ ...
    featureInputLayer(numFeatures, 'Normalization','zscore','Name','input')
    fullyConnectedLayer(64, 'Name','fc1')
    reluLayer('Name','relu1')
    dropoutLayer(0.5, 'Name','dropout1')
    fullyConnectedLayer(32, 'Name','fc2')
    reluLayer('Name','relu2')
    dropoutLayer(0.5, 'Name','dropout2')
    fullyConnectedLayer(numClasses, 'Name','fc3')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classOutput')];

% Define training options using the Adam optimizer.
options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 32, ...
    'Verbose', false, ...
    'Plots', 'none');

%% Leave-One-Patient-Out Cross-Validation (LOPO CV)
% Identify unique patients by the InStudyID variable.
uniquePatients = unique(data.InStudyID);
numPatients = numel(uniquePatients);

% Preallocate a cell array to store predictions for each observation.
predictedLabels_dl = cell(height(data), 1);

fprintf('Starting Leave-One-Patient-Out CV with Deep Learning...\n');
for i = 1:numPatients
    fprintf('Processing patient %d of %d...\n', i, numPatients);
    
    % Identify test indices: all rows corresponding to the current patient.
    testIdx = ismember(data.InStudyID, uniquePatients(i));
    trainIdx = ~testIdx;
    
    % Create training and test tables.
    trainTbl = data(trainIdx, [predictorNames, {'Response'}]);
    testTbl  = data(testIdx, predictorNames);
    
    % Check if the training set contains at least two classes.
    uniqueTrainClasses = unique(trainTbl.Response);
    if numel(uniqueTrainClasses) < 2
        warning('Fold %d: Training set contains only one class (%s). Assigning that class to test observations.', ...
            i, string(uniqueTrainClasses));
        predictedLabels_dl(testIdx) = repmat(cellstr(uniqueTrainClasses), sum(testIdx), 1);
        continue;  % Skip training for this fold
    end
    
    % Train the deep learning model on the current training data.
    net = trainNetwork(trainTbl, layers, options);
    
    % Classify the test data using the trained network.
    predicted = classify(net, testTbl);
    
    % Store predictions.
    predictedLabels_dl(testIdx) = cellstr(predicted);
end

% Convert the cell array of predictions to a categorical array.
predictedLabels_dl = categorical(predictedLabels_dl);

%% Evaluate the Model Performance
% Compute the misclassification rate.
misclassRate = sum(~strcmp(string(predictedLabels_dl), string(data.Response))) / height(data);
fprintf('Deep Learning LOPO CV Misclassification Rate: %.4f\n', misclassRate);

% Compute and display the confusion matrix.
confMat = confusionmat(data.Response, predictedLabels_dl);
disp('Deep Learning LOPO CV Confusion Matrix:');
disp(confMat);
