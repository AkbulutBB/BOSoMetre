%% Model Evaluation with LOPO CV (Binary & Triphasic) with Infection-Sensitive Weighting
% This script evaluates several classifiers using Leave-One-Patient-Out CV.
% It supports both 'binary' (using infClassIDSA) and 'triphasic' (using triClassIDSA)
% classification schemes. A weighting scheme is applied so that the models are more
% sensitive to infections (i.e., they are biased to predict the Infection class when in doubt).
%
% References:
%   Varma, S., & Simon, R. (2006). Bias in error estimation when using 
%     cross-validation for model selection. BMC Bioinformatics, 7, 91.
%     https://doi.org/10.1186/1471-2105-7-91
%
%   Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning.
%
%   MathWorks. (n.d.). Train a Deep Learning Network for Tabular Data.
%   MathWorks. (n.d.). Fit ECOC Models for Multi-Class Classification.

%% Preliminary Setup
clc;
clearvars -except BOSoMetreDataCSFDataForTraining

if ~exist('BOSoMetreDataCSFDataForTraining', 'var')
    error('BOSoMetreDataCSFDataForTraining is not found in the workspace.');
end
data = BOSoMetreDataCSFDataForTraining;

%% Exclude Specific Patients
excludedPatients = [9, 17, 18, 19];
data = data(~ismember(data.InStudyID, excludedPatients), :);

%% Specify Classification Type and Preprocess Data
% Choose either 'binary' (using infClassIDSA) or 'triphasic' (using triClassIDSA)
classificationType = 'triphasic';  % Change to 'binary' if needed

predictorNames = {'RPerc', 'GPerc', 'BPerc', 'CPerc'};

if strcmpi(classificationType, 'binary')
    colsToCheck = [predictorNames, {'infClassIDSA'}];
elseif strcmpi(classificationType, 'triphasic')
    colsToCheck = [predictorNames, {'triClassIDSA'}];
end
data = rmmissing(data, 'DataVariables', colsToCheck);

% Convert response to categorical with explicit ordering.
if strcmpi(classificationType, 'binary')
    % For binary: 0 -> 'NonInfection', 1 -> 'Infection'
    Y = categorical(data.infClassIDSA, [0 1], {'NonInfection','Infection'});
elseif strcmpi(classificationType, 'triphasic')
    % For triphasic: 0 -> 'NonInfection', 1 -> 'Infection', 2 -> 'Indeterminate'
    Y = categorical(data.triClassIDSA, [0 1 2], {'NonInfection','Infection','Indeterminate'});
end
data.Response = Y;

%% Define Weighting Scheme
% Cost matrix penalizes misclassifying an infection more heavily.
if strcmpi(classificationType, 'binary')
    % For binary classification:
    % Rows: true class, Columns: predicted class.
    % Here, a false negative (predicting 'NonInfection' for a true 'Infection') is penalized.
    costMat = [0, 1; 5, 0];  
    classWeights = [1, 5];    % Higher weight for Infection (assumed second class)
elseif strcmpi(classificationType, 'triphasic')
    % For triphasic, we similarly penalize misclassifying Infection.
    costMat = [0, 1, 1; 5, 0, 5; 1, 1, 0];
    classWeights = [1, 5, 1]; % Increase weight for Infection (second class)
end

%% Define List of Models to Evaluate
modelList = {'DecisionTree', 'kNN', 'SVM', 'Ensemble', 'DeepLearning'};
results = struct;

% Identify unique patients for LOPO CV.
uniquePatients = unique(data.InStudyID);
numPatients = numel(uniquePatients);

%% LOPO Cross-Validation Loop for Each Model
for m = 1:length(modelList)
    modelName = modelList{m};
    predictedLabels = cell(height(data), 1);
    fprintf('Evaluating model: %s\n', modelName);
    
    % Loop over each patient.
    for i = 1:numPatients
        fprintf('  Processing patient %d of %d...\n', i, numPatients);
        
        % Define test and training indices.
        testIdx = ismember(data.InStudyID, uniquePatients(i));
        trainIdx = ~testIdx;
        trainTbl = data(trainIdx, [predictorNames, {'Response'}]);
        testTbl  = data(testIdx, predictorNames);
        
        % Ensure the training set has at least two classes.
        uniqueTrainClasses = unique(trainTbl.Response);
        if numel(uniqueTrainClasses) < 2
            warning('Fold %d: Training set contains only one class (%s). Assigning that class to test observations.', ...
                i, string(uniqueTrainClasses));
            predictedLabels(testIdx) = repmat(cellstr(uniqueTrainClasses), sum(testIdx), 1);
            continue;
        end
        
        % Train and predict with the selected model.
        switch modelName
            case 'DecisionTree'
                mdl = fitctree(trainTbl(:, predictorNames), trainTbl.Response, 'Cost', costMat);
                predicted = predict(mdl, testTbl);
                
            case 'kNN'
                mdl = fitcknn(trainTbl(:, predictorNames), trainTbl.Response);
                predicted = predict(mdl, testTbl);
                
            case 'SVM'
                % For multi-class SVM use ECOC.
                mdl = fitcecoc(trainTbl(:, predictorNames), trainTbl.Response, 'Cost', costMat);
                predicted = predict(mdl, testTbl);
                
            case 'Ensemble'
                mdl = fitcensemble(trainTbl(:, predictorNames), trainTbl.Response, 'Method', 'Bag', 'Cost', costMat);
                predicted = predict(mdl, testTbl);
                
            case 'DeepLearning'
                % Define the deep learning network architecture.
                numFeatures = numel(predictorNames);
                numClasses  = numel(categories(trainTbl.Response));
                layers = [ ...
                    featureInputLayer(numFeatures, 'Normalization', 'zscore', 'Name', 'input')
                    fullyConnectedLayer(64, 'Name', 'fc1')
                    reluLayer('Name', 'relu1')
                    dropoutLayer(0.5, 'Name', 'dropout1')
                    fullyConnectedLayer(32, 'Name', 'fc2')
                    reluLayer('Name', 'relu2')
                    dropoutLayer(0.5, 'Name', 'dropout2')
                    fullyConnectedLayer(numClasses, 'Name', 'fc3')
                    softmaxLayer('Name', 'softmax')];
                
                % Append a weighted classification layer.
                % Now we explicitly specify 'Classes' along with 'ClassWeights'.
                weightedClassLayer = classificationLayer('Name', 'classOutput', ...
                    'Classes', categories(trainTbl.Response), ...
                    'ClassWeights', classWeights);
                layers = [layers; weightedClassLayer];
                
                options = trainingOptions('adam', ...
                    'MaxEpochs', 30, ...
                    'MiniBatchSize', 32, ...
                    'Verbose', false, ...
                    'Plots', 'none');
                
                net = trainNetwork(trainTbl, layers, options);
                predicted = classify(net, testTbl);
                
            otherwise
                error('Unknown model: %s', modelName);
        end
        
        predictedLabels(testIdx) = cellstr(predicted);
    end
    
    % Convert predictions to a categorical array with the same order as the true responses.
    predictedLabelsCat = categorical(predictedLabels, categories(data.Response));
    
    % Compute misclassification rate.
    misclassRate = sum(~strcmp(string(predictedLabelsCat), string(data.Response))) / height(data);
    confMat = confusionmat(data.Response, predictedLabelsCat);
    
    % Compute sensitivity and specificity.
    if strcmpi(classificationType, 'binary')
        % Assuming ordering: 1 = NonInfection, 2 = Infection.
        TN = confMat(1,1);
        FP = confMat(1,2);
        FN = confMat(2,1);
        TP = confMat(2,2);
        
        sensitivity = TP / (TP + FN);
        specificity = TN / (TN + FP);
        
        fprintf('%s: Sensitivity = %.4f, Specificity = %.4f\n', modelName, sensitivity, specificity);
        
    elseif strcmpi(classificationType, 'triphasic')
        % For triphasic, evaluate the Infection class (assumed to be the second category)
        % using a one-versus-all approach.
        actualBinary = (data.Response == 'Infection');
        predictedBinary = (predictedLabelsCat == 'Infection');
        
        TP = sum(actualBinary & predictedBinary);
        FN = sum(actualBinary & ~predictedBinary);
        TN = sum(~actualBinary & ~predictedBinary);
        FP = sum(~actualBinary & predictedBinary);
        
        sensitivity = TP / (TP + FN);
        specificity = TN / (TN + FP);
        
        fprintf('%s (Infection vs. Not): Sensitivity = %.4f, Specificity = %.4f\n', modelName, sensitivity, specificity);
    end
    
    % Store results.
    results.(modelName).misclassRate = misclassRate;
    results.(modelName).confMat = confMat;
    results.(modelName).sensitivity = sensitivity;
    results.(modelName).specificity = specificity;
    
    fprintf('%s: Misclassification Rate = %.4f\n', modelName, misclassRate);
    disp('Confusion Matrix:');
    disp(confMat);
end

%% Summary of Results
disp('Summary of LOPO CV Results:');
disp(results);
