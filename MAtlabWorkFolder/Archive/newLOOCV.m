%% Final LOOCV Script with Class Weighting Fix
clc;

% 1. Load Data
data = BOSoMetreDataCSFDataForTraining;  % Replace with your dataset

% 2. Prepare Target Variable
classificationType = 'binary';  % 'binary' or 'triphasic'
switch lower(classificationType)
    case 'binary'
        Y = data.infClassIDSA;
    case 'triphasic'
        Y = data.triClassIDSA;
    otherwise
        error('Invalid classification type.');
end
Y = grp2idx(Y) - 1;  % Convert to 0-based numeric labels (0,1 or 0,1,2)

% 3. Remove rows with NaN in Y
nanMask = isnan(Y);
if any(nanMask)
    warning('Removing rows with NaN in target variable.');
    data(nanMask, :) = [];
    Y = data.(['infClassIDSA' (classificationType == 'triphasic')*'triClassIDSA']);  % Re-extract Y
    Y = grp2idx(Y) - 1;
end

% 4. Prepare Predictors
predictorNames = {'RPerc', 'GPerc', 'BPerc', 'CPerc'};
X = data{:, predictorNames};
patientIDs = data.InStudyID;
uniquePatients = unique(patientIDs);
numPatients = numel(uniquePatients);

% 5. Configure Class Weighting
useClassWeights = true;  % Toggle to enable/disable

% 6. Preallocate Results
predictedLabels = nan(size(Y));

%% Leave-One-Patient-Out CV
for i = 1:numPatients
    % Identify current patient
    testIdx = (patientIDs == uniquePatients(i));
    trainIdx = ~testIdx;
    
    % Get training data
    X_train = X(trainIdx, :);
    Y_train = Y(trainIdx);
    
    % Calculate observation weights (class balancing)
    if useClassWeights
        uniqueClasses = unique(Y_train);
        classCounts = histcounts(Y_train, [uniqueClasses; max(uniqueClasses)+1]);  % Ensure counts align
        classWeights = 1 ./ classCounts;
        classWeights = classWeights / sum(classWeights);
        
        % Map Y_train to class indices (1-based)
        [~, loc] = ismember(Y_train, uniqueClasses);
        obsWeights = classWeights(loc)';
    else
        obsWeights = ones(size(Y_train));
    end
    
    % Train logistic regression with weights
    model = fitclinear(X_train, Y_train, ...
        'Learner', 'logistic', ...
        'Regularization', 'ridge', ...
        'Lambda', 0.1, ...
        'Weights', obsWeights);
    
    % Predict on left-out patient
    predictedLabels(testIdx) = predict(model, X(testIdx, :));
end

%% Performance Evaluation
% Confusion Matrix
[confMat, order] = confusionmat(Y, predictedLabels);
disp('Confusion Matrix:');
disp(array2table(confMat, ...
    'RowNames', strcat('True_', cellstr(num2str(order))), ...
    'VariableNames', strcat('Predicted_', cellstr(num2str(order)))));

% Accuracy
accuracy = sum(predictedLabels == Y) / numel(Y);
fprintf('\nLOOCV Accuracy: %.2f%%\n', accuracy*100);

% Precision/Recall/F1 (for binary only)
if numel(order) == 2
    precision = confMat(2,2) / sum(confMat(:,2));
    recall = confMat(2,2) / sum(confMat(2,:));
    f1 = 2 * (precision * recall) / (precision + recall);
    fprintf('Precision: %.2f%%, Recall: %.2f%%, F1: %.2f%%\n', ...
        precision*100, recall*100, f1*100);
end

% List misclassified patients
misclassified = unique(patientIDs(predictedLabels ~= Y));
fprintf('\nMisclassified Patient IDs:\n');
disp(misclassified);