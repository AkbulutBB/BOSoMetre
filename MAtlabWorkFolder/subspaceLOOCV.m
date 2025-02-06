%% Subspace KNN with LOPO CV Excluding Specific Patients
% This script uses a subspace KNN ensemble to predict infection status (binary or triphasic)
% using predictor variables: RPerc, GPerc, BPerc, and CPerc.
% It implements Leave-One-Patient-Out (LOPO) cross-validation and excludes patients with
% InStudyID equal to 9, 17, 18, or 19 (e.g., patients with bleeding in their samples).

%% Preliminary Setup
clc;        % Clear the Command Window
clearvars -except BOSoMetreDataCSFDataForTraining;      % Clear the workspace

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
% Set classificationType to either 'binary' (using infClassIDSA) or 'triphasic' (using triClassIDSA)
classificationType = 'binary';  % Change to 'triphasic' if desired

% Define predictor variables.
predictorNames = {'RPerc', 'GPerc', 'BPerc', 'CPerc'};

% Remove rows with missing values in the predictors and chosen response.
if strcmpi(classificationType, 'triphasic')
    colsToCheck = [predictorNames, {'infClassIDSA'}];
elseif strcmpi(classificationType, 'triphasic')
    colsToCheck = [predictorNames, {'triClassIDSA'}];
end
data = rmmissing(data, 'DataVariables', colsToCheck);

% Select and convert the response variable to categorical.
switch lower(classificationType)
    case 'binary'
        Y = data.infClassIDSA;
    case 'triphasic'
        Y = data.triClassIDSA;
    otherwise
        error('Invalid classification type. Choose either "binary" or "triphasic".');
end
Y = categorical(Y);
data.Response = Y;  % Append the response as a new column for convenience

%% Leave-One-Patient-Out CV with Subspace KNN
% Identify unique patients based on InStudyID.
uniquePatients = unique(data.InStudyID);
numPatients = numel(uniquePatients);

% Preallocate cell array for predictions.
predictedLabels = cell(height(data), 1);

fprintf('Starting Leave-One-Patient-Out CV with Subspace KNN...\n');
for i = 1:numPatients
    fprintf('Processing patient %d of %d...\n', i, numPatients);
    
    % Determine the test indices for the current patient.
    testIdx = ismember(data.InStudyID, uniquePatients(i));
    trainIdx = ~testIdx;
    
    % Create training and test tables.
    trainTbl = data(trainIdx, [predictorNames, {'Response'}]);
    testTbl  = data(testIdx, predictorNames);
    
    % Check that the training set contains at least two classes.
    if numel(unique(trainTbl.Response)) < 2
        warning('Fold %d: Training set contains only one class (%s). Assigning that class to test observations.', ...
            i, string(unique(trainTbl.Response)));
        predictedLabels(testIdx) = repmat(cellstr(unique(trainTbl.Response)), sum(testIdx), 1);
        continue;
    end
    
    % Define a KNN learner template.
    knnTemplate = templateKNN('NumNeighbors', 10);
    
    % Train a subspace ensemble using the 'Subspace' method.
    model = fitcensemble(trainTbl{:, predictorNames}, trainTbl.Response, ...
        'Method', 'Subspace', 'Learners', knnTemplate, 'NumLearningCycles', 30);
    
    % Predict the responses for the test set.
    pred = predict(model, testTbl{:, predictorNames});
    
    % Store the predictions.
    predictedLabels(testIdx) = cellstr(pred);
end

% Convert the cell array of predictions to a categorical array.
predictedLabels = categorical(predictedLabels);

%% Evaluate Model Performance
misclassRate = sum(predictedLabels ~= data.Response) / height(data);
fprintf('LOPO CV Misclassification Rate: %.4f\n', misclassRate);

confMat = confusionmat(data.Response, predictedLabels);
disp('LOPO CV Confusion Matrix:');
disp(confMat);
