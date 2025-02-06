%% Model Evaluation with LOPO CV (Binary & Triphasic) with Reject Option
% This script evaluates several classifiers using Leave-One-Patient-Out CV.
% In addition to the standard prediction, the script now implements a reject
% option: if the classifier’s confidence (maximum predicted probability) is
% below a specified threshold, the system will output 'Reject' rather than a class.
%
% Additional models added:
%   - BoostedTree: An ensemble using AdaBoostM1.
%   - MixedEffects: A generalized linear mixed-effects model (only for binary classification).
%
% References:
%   Chow, C. K. (1970). A Note on Optimal Classification with Rejection.
%   Geifman, Y., & El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks.
%   Varma, S., & Simon, R. (2006). Bias in error estimation when using cross-validation for model selection.

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
classificationType = 'binary';  % Change to 'binary' if needed
predictorNames = {'RPerc', 'GPerc', 'BPerc', 'CPerc'};

if strcmpi(classificationType, 'binary')
    colsToCheck = [predictorNames, {'infClassIDSA'}];
elseif strcmpi(classificationType, 'triphasic')
    colsToCheck = [predictorNames, {'triClassIDSA'}];
end
data = rmmissing(data, 'DataVariables', colsToCheck);

% Convert response to categorical with explicit ordering.
if strcmpi(classificationType, 'binary')
    Y = categorical(data.infClassIDSA, [0 1], {'NonInfection','Infection'});
elseif strcmpi(classificationType, 'triphasic')
    Y = categorical(data.triClassIDSA, [0 1 2], {'NonInfection','Infection','Unsure'});
end
data.Response = Y;

%% Define Weighting Scheme
if strcmpi(classificationType, 'binary')
    costMat = [0, 1; 2, 0];  
    classWeights = [1, 2];    % Higher weight for Infection (assumed second class)
elseif strcmpi(classificationType, 'triphasic')
    costMat = [0, 1, 1; 2, 0, 2; 1, 1, 0];
    classWeights = [1, 2, 1]; % Increase weight for Infection (second class)
end

%% Define Confidence Threshold for Reject Option
confidenceThreshold = 0.95; % If max predicted probability < threshold, output 'Reject'

%% Define List of Models to Evaluate
modelList = {'DecisionTree', 'kNN', 'SVM', 'Ensemble', 'DeepLearning', 'BoostedTree', 'MixedEffects'};
results = struct;

% Identify unique patients for LOPO CV.
uniquePatients = unique(data.InStudyID);
numPatients = numel(uniquePatients);

%% LOPO Cross-Validation Loop for Each Model
for m = 1:length(modelList)
    modelName = modelList{m};
    predictedLabels = cell(height(data), 1);
    fprintf('Evaluating model: %s\n', modelName);
    
    % Preallocate vector for per-patient accuracies.
    patientAcc = zeros(numPatients, 1);
    
    % Loop over each patient.
    for i = 1:numPatients
        fprintf('  Processing patient %d of %d...\n', i, numPatients);
        
        % Define test and training indices.
        testIdx = ismember(data.InStudyID, uniquePatients(i));
        trainIdx = ~testIdx;
        trainTbl = data(trainIdx, [predictorNames, {'Response', 'InStudyID'}]);
        testTbl  = data(testIdx, predictorNames);
        
        % Ensure the training set has at least two classes.
        uniqueTrainClasses = unique(trainTbl.Response);
        if numel(uniqueTrainClasses) < 2
            warning('Fold %d: Training set contains only one class (%s). Assigning that class to test observations.', ...
                i, string(uniqueTrainClasses));
            predictedLabels(testIdx) = repmat(cellstr(uniqueTrainClasses), sum(testIdx), 1);
            patientAcc(i) = 1;
            continue;
        end
        
        % Train and predict with the selected model.
        switch modelName
            case 'DecisionTree'
                % Obtain both predicted labels and scores.
                [predictedTemp, score] = predict(fitctree(trainTbl(:, predictorNames), trainTbl.Response, 'Cost', costMat), testTbl);
                [~, idx] = max(score, [], 2);
                classNames = categories(trainTbl.Response);
                predictedCell = cell(size(predictedTemp));
                for j = 1:length(predictedTemp)
                    if score(j, idx(j)) < confidenceThreshold
                        predictedCell{j} = 'Reject';
                    else
                        predictedCell{j} = char(predictedTemp(j));
                    end
                end
                predicted = categorical(predictedCell, [classNames; {'Reject'}]);
                
            case 'kNN'
                [predictedTemp, score] = predict(fitcknn(trainTbl(:, predictorNames), trainTbl.Response), testTbl);
                [~, idx] = max(score, [], 2);
                classNames = categories(trainTbl.Response);
                predictedCell = cell(size(predictedTemp));
                for j = 1:length(predictedTemp)
                    if score(j, idx(j)) < confidenceThreshold
                        predictedCell{j} = 'Reject';
                    else
                        predictedCell{j} = char(predictedTemp(j));
                    end
                end
                predicted = categorical(predictedCell, [classNames; {'Reject'}]);
                
            case 'SVM'
                mdl = fitcecoc(trainTbl(:, predictorNames), trainTbl.Response, 'Cost', costMat, 'FitPosterior', true);
                [predictedTemp, score] = predict(mdl, testTbl);
                [~, idx] = max(score, [], 2);
                classNames = categories(trainTbl.Response);
                predictedCell = cell(size(predictedTemp));
                for j = 1:length(predictedTemp)
                    if score(j, idx(j)) < confidenceThreshold
                        predictedCell{j} = 'Reject';
                    else
                        predictedCell{j} = char(predictedTemp(j));
                    end
                end
                predicted = categorical(predictedCell, [classNames; {'Reject'}]);
                
            case 'Ensemble'
                [predictedTemp, score] = predict(fitcensemble(trainTbl(:, predictorNames), trainTbl.Response, 'Method', 'Bag', 'Cost', costMat), testTbl);
                [~, idx] = max(score, [], 2);
                classNames = categories(trainTbl.Response);
                predictedCell = cell(size(predictedTemp));
                for j = 1:length(predictedTemp)
                    if score(j, idx(j)) < confidenceThreshold
                        predictedCell{j} = 'Reject';
                    else
                        predictedCell{j} = char(predictedTemp(j));
                    end
                end
                predicted = categorical(predictedCell, [classNames; {'Reject'}]);
                
            case 'DeepLearning'
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
                weightedClassLayer = classificationLayer('Name', 'classOutput', ...
                    'Classes', categories(trainTbl.Response), ...
                    'ClassWeights', classWeights);
                layers = [layers; weightedClassLayer];
                options = trainingOptions('adam', 'MaxEpochs', 30, 'MiniBatchSize', 32, 'Verbose', false, 'Plots', 'none');
                XTrain = table2array(trainTbl(:, predictorNames));
                YTrain = trainTbl.Response;
                net = trainNetwork(XTrain, YTrain, layers, options);
                XTest = table2array(testTbl);
                YProb = predict(net, XTest);
                [maxScore, idx] = max(YProb, [], 2);
                classNames = categories(trainTbl.Response);
                predictedCell = cell(size(maxScore));
                for j = 1:length(maxScore)
                    if maxScore(j) < confidenceThreshold
                        predictedCell{j} = 'Reject';
                    else
                        predictedCell{j} = char(classNames{idx(j)});
                    end
                end
                predicted = categorical(predictedCell, [classNames; {'Reject'}]);
                
            case 'BoostedTree'
                [predictedTemp, score] = predict(fitcensemble(trainTbl(:, predictorNames), trainTbl.Response, 'Method', 'AdaBoostM1', 'Cost', costMat), testTbl);
                [~, idx] = max(score, [], 2);
                classNames = categories(trainTbl.Response);
                predictedCell = cell(size(predictedTemp));
                for j = 1:length(predictedTemp)
                    if score(j, idx(j)) < confidenceThreshold
                        predictedCell{j} = 'Reject';
                    else
                        predictedCell{j} = char(predictedTemp(j));
                    end
                end
                predicted = categorical(predictedCell, [classNames; {'Reject'}]);
                
            case 'MixedEffects'
                if strcmpi(classificationType, 'binary')
                    trainTblGLME = trainTbl;
                    trainTblGLME.ResponseNum = double(trainTblGLME.Response == 'Infection');
                    trainTblGLME.InStudyID = categorical(trainTblGLME.InStudyID);
                    formula = 'ResponseNum ~ RPerc + GPerc + BPerc + CPerc + (1|InStudyID)';
                    glme = fitglme(trainTblGLME, formula, 'Distribution', 'Binomial');
                    testTblGLME = testTbl;
                    testTblGLME.InStudyID = categorical(data.InStudyID(testIdx));
                    predictedProb = predict(glme, testTblGLME);
                    predictedCell = cell(size(predictedProb));
                    for j = 1:length(predictedProb)
                        if predictedProb(j) < confidenceThreshold
                            predictedCell{j} = 'Reject';
                        else
                            predictedCell{j} = 'Infection';
                        end
                    end
                    predicted = categorical(predictedCell, [{'NonInfection','Infection','Reject'}]);
                else
                    warning('MixedEffects model is only implemented for binary classification. Skipping fold %d.', i);
                    predicted = repmat(cellstr(mode(trainTbl.Response)), sum(testIdx), 1);
                end
                
            otherwise
                error('Unknown model: %s', modelName);
        end
        
        predictedLabels(testIdx) = cellstr(predicted);
        
        % Compute per-patient accuracy.
        trueLabels = data.Response(testIdx);
        patientAcc(i) = sum(strcmp(cellstr(predicted), cellstr(trueLabels))) / sum(testIdx);
    end
    
    fprintf('%s: Per-patient accuracies:\n', modelName);
    disp(patientAcc);
    results.(modelName).patientAcc = patientAcc;
    
    predictedLabelsCat = categorical(predictedLabels, [categories(data.Response); {'Reject'}]);
    misclassRate = sum(~strcmp(string(predictedLabelsCat), string(data.Response))) / height(data);
    confMat = confusionmat(data.Response, predictedLabelsCat);
    
    if strcmpi(classificationType, 'binary')
        TN = confMat(1,1);
        FP = confMat(1,2);
        FN = confMat(2,1);
        TP = confMat(2,2);
        sensitivity = TP / (TP + FN);
        specificity = TN / (TN + FP);
        fprintf('%s: Sensitivity = %.4f, Specificity = %.4f\n', modelName, sensitivity, specificity);
    elseif strcmpi(classificationType, 'triphasic')
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
