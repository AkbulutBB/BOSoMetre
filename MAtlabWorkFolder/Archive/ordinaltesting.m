%% Binary Logistic Regression Model with Reject Option
% This script demonstrates a binary probabilistic model using multinomial
% logistic regression (via mnrfit/mnrval) with a reject option.
%
% Assumptions:
%   - Data is stored in a table called BOSoMetreDataCSFDataForTraining.
%   - Predictors: 'RPerc', 'GPerc', 'BPerc', 'CPerc'
%   - Outcome: 'infClassIDSA' where 0 = NonInfection, 1 = Infection
%
% The script uses a hold-out (80/20) random split for demonstration.
% You can later embed this into your LOPO or cross-validation framework.
%
% References:
%   Chow, C. K. (1970). A Note on Optimal Classification with Rejection.
%   Geifman, Y., & El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks.
%   Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning.

%% Preliminary Setup
clc;
clearvars -except BOSoMetreDataCSFDataForTraining

% Load your data. (Ensure BOSoMetreDataCSFDataForTraining is in your workspace.)
if exist('BOSoMetreDataCSFDataForTraining','var')
    data = BOSoMetreDataCSFDataForTraining;
else
    error('BOSoMetreDataCSFDataForTraining not found in workspace.');
end

% (Optional) Exclude specific patients if needed.
excludedPatients = [9, 17, 18, 19];
data = data(~ismember(data.InStudyID, excludedPatients), :);

% Remove rows with missing values in the predictors or outcome.
predictorNames = {'RPerc','GPerc','BPerc','CPerc'};
colsToCheck = [predictorNames, {'infClassIDSA'}];
data = rmmissing(data, 'DataVariables', colsToCheck);

%% Define Outcome and Predictors
% Create a binary categorical variable for the outcome.
% Here, we map 0 -> 'NonInfection' and 1 -> 'Infection'.
Y = categorical(data.infClassIDSA, [0, 1], {'NonInfection','Infection'});
X = table2array(data(:, predictorNames));

%% Split Data into Training and Testing Sets (80/20 Split)
cv = cvpartition(height(data),'HoldOut',0.2);
trainIdx = training(cv);
testIdx = test(cv);

XTrain = X(trainIdx, :);
YTrain = Y(trainIdx);
XTest  = X(testIdx, :);
YTest  = Y(testIdx);

%% Fit the Binary Logistic Regression Model
% mnrfit for binary data expects the response as numeric indices: 1 for the first
% category and 2 for the second category.
YTrainNum = double(YTrain);  % This yields 1 for 'NonInfection' and 2 for 'Infection'

% Fit the model.
B = mnrfit(XTrain, YTrainNum);

%% Predict Probabilities on the Test Set
% mnrval returns an N-by-2 matrix of probabilities.
probabilities = mnrval(B, XTest);

%% Examine the Distribution of Maximum Predicted Probabilities
[maxProb, predIdx] = max(probabilities, [], 2);
figure;
histogram(maxProb, 'Normalization','probability');
xlabel('Maximum Predicted Probability');
ylabel('Frequency');
title('Distribution of Maximum Predicted Probabilities (Test Set)');

%% Apply Reject Option Based on Confidence Threshold
% Here we use a lower threshold (e.g., 0.80) compared to the ordinal model.
confidenceThreshold = 0.80;
% Initialize predicted labels as "Reject"
predictedLabels = repmat("Reject", size(predIdx));
% Get the category names (order as in YTrain)
catNames = categories(YTrain);  % {'NonInfection', 'Infection'}

for i = 1:length(maxProb)
    if maxProb(i) >= confidenceThreshold
        predictedLabels(i) = catNames{predIdx(i)};
    else
        predictedLabels(i) = "Reject";
    end
end

% Convert predictedLabels to a categorical variable with an extended set of labels.
predictedLabels = categorical(predictedLabels, [catNames; {'Reject'}]);

%% Evaluate Model Performance
% Compute confusion matrix and error/rejection rates.
confMat = confusionmat(YTest, predictedLabels);
misclassificationRate = sum(~strcmp(string(predictedLabels), string(YTest))) / length(YTest);
rejectionRate = sum(predictedLabels == "Reject") / length(predictedLabels);

fprintf('Binary Logistic Regression with Reject Option\n');
fprintf('Misclassification Rate (including rejects as errors): %.4f\n', misclassificationRate);
fprintf('Rejection Rate: %.4f\n', rejectionRate);
disp('Confusion Matrix:');
disp(confMat);
