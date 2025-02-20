%% LOOCV_ModelTraining_WithExclusions.m
% This script demonstrates LOOCV-based training and evaluation of several 
% classification models for predicting CSF status based on sensor readings,
% while excluding specified patients (IDs: 9, 17, 18, 19) from the analysis.
%
% Two classification schemes are performed:
%   1. Binary classification (infClassIDSA: 0 = clean, 1 = infective)
%   2. Triphasic classification (triClassIDSA: 0 = clean, 1 = infected, 2 = in between)
%
% The sensor features are: RPerc, GPerc, BPerc, CPerc (from the TCS3200 sensor).
%
% References:
%   MathWorks. (n.d.). *Waitbar*. Retrieved from https://www.mathworks.com/help/matlab/ref/waitbar.html
%   MathWorks. (n.d.). *Using fprintf and drawnow*. Retrieved from https://www.mathworks.com/help/matlab/ref/drawnow.html
%   MathWorks. (n.d.). *ismember*. Retrieved from https://www.mathworks.com/help/matlab/ref/ismember.html
%   IDSA Guidelines (2017). Infectious Diseases Society of America.
%
% Author: [Your Name]
% Date: [Today's Date]

%% Clear Workspace and Load Data
clc;
clearvars -except BOSoMetreDataCSFDataForTraining

if ~exist('BOSoMetreDataCSFDataForTraining', 'var')
    error('BOSoMetreDataCSFDataForTraining is not found in the workspace.');
end
data = BOSoMetreDataCSFDataForTraining;

%% Exclude Specific Patients from Analysis
% Exclude patients with IDs 9, 17, 18, 19 so that they are not used for training or testing.
excludePatients = [9, 17, 18, 19, 21];
data = data(~ismember(data.InStudyID, excludePatients), :);

%% Extract Features and Labels
% Extract sensor features (adjust the variable names if needed)
features = data{:, {'RPerc', 'GPerc', 'BPerc', 'CPerc'}};

% Extract labels for binary and triphasic classification
labels_binary = data.infClassIDSA;   % 0 for clean, 1 for infective
labels_triphasic = data.triClassIDSA;  % 0 for clean, 1 for infected, 2 for in between

% Identify unique patient IDs for LOOCV (the variable InStudyID is assumed numeric or categorical)
patient_ids = unique(data.InStudyID);

%% Preallocate Arrays for Storing Predictions and True Labels
% For binary classification
preds_tree_binary = [];
preds_svm_binary  = [];
preds_knn_binary  = [];
true_labels_binary = [];

% For triphasic classification
preds_tree_triphasic = [];
preds_svm_triphasic  = [];
preds_knn_triphasic  = [];
true_labels_triphasic = [];

%% Initialize Waitbar for LOOCV Progress
hWait = waitbar(0, 'Performing LOOCV...');

%% LOOCV: Iterate Over Each Patient as the Test Set
for i = 1:length(patient_ids)
    % Update the console with the current patient number and flush output
    fprintf('Processing patient %d of %d...\n', i, length(patient_ids));
    drawnow;  % Ensures immediate display of the message
    
    % Update the waitbar
    waitbar(i/length(patient_ids), hWait, sprintf('Processing patient %d of %d', i, length(patient_ids)));
    
    % Create logical indices for the current test patient and the training set
    test_idx  = (data.InStudyID == patient_ids(i));
    train_idx = ~test_idx;
    
    % Partition the data for binary classification
    X_train = features(train_idx, :);
    y_train_bin = labels_binary(train_idx);
    X_test  = features(test_idx, :);
    y_test_bin  = labels_binary(test_idx);
    
    % Partition the data for triphasic classification
    y_train_tri = labels_triphasic(train_idx);
    y_test_tri  = labels_triphasic(test_idx);
    
    %% Binary Classification Models
    % 1. Decision Tree
    model_tree_bin = fitctree(X_train, y_train_bin);
    pred_tree_bin = predict(model_tree_bin, X_test);
    
    % 2. Support Vector Machine (SVM) with a linear kernel
    model_svm_bin = fitcsvm(X_train, y_train_bin, 'KernelFunction', 'linear');
    pred_svm_bin = predict(model_svm_bin, X_test);
    
    % 3. k-Nearest Neighbors (kNN) with k=5 (adjustable)
    model_knn_bin = fitcknn(X_train, y_train_bin, 'NumNeighbors', 5);
    pred_knn_bin = predict(model_knn_bin, X_test);
    
    % Append the predictions and ground truth for binary classification
    preds_tree_binary = [preds_tree_binary; pred_tree_bin];
    preds_svm_binary  = [preds_svm_binary;  pred_svm_bin];
    preds_knn_binary  = [preds_knn_binary;  pred_knn_bin];
    true_labels_binary = [true_labels_binary; y_test_bin];
    
    %% Triphasic Classification Models
    % 1. Decision Tree
    model_tree_tri = fitctree(X_train, y_train_tri);
    pred_tree_tri = predict(model_tree_tri, X_test);
    
    % 2. Support Vector Machine (SVM) for multiclass classification 
    %    using the Error-Correcting Output Codes (ECOC) framework
    model_svm_tri = fitcecoc(X_train, y_train_tri);
    pred_svm_tri = predict(model_svm_tri, X_test);
    
    % 3. k-Nearest Neighbors (kNN) with k=5
    model_knn_tri = fitcknn(X_train, y_train_tri, 'NumNeighbors', 5);
    pred_knn_tri = predict(model_knn_tri, X_test);
    
    % Append the predictions and ground truth for triphasic classification
    preds_tree_triphasic = [preds_tree_triphasic; pred_tree_tri];
    preds_svm_triphasic  = [preds_svm_triphasic;  pred_svm_tri];
    preds_knn_triphasic  = [preds_knn_triphasic;  pred_knn_tri];
    true_labels_triphasic = [true_labels_triphasic; y_test_tri];
end

% Close the waitbar after completion
close(hWait);

%% Evaluation of Model Performance
% Binary Classification Accuracy
accuracy_tree_bin = sum(preds_tree_binary == true_labels_binary) / numel(true_labels_binary);
accuracy_svm_bin  = sum(preds_svm_binary  == true_labels_binary) / numel(true_labels_binary);
accuracy_knn_bin  = sum(preds_knn_binary  == true_labels_binary) / numel(true_labels_binary);

fprintf('\nBinary Classification Accuracy:\n');
fprintf('  Decision Tree: %.2f%%\n', accuracy_tree_bin * 100);
fprintf('  SVM:           %.2f%%\n', accuracy_svm_bin * 100);
fprintf('  k-NN:          %.2f%%\n', accuracy_knn_bin * 100);

% Triphasic Classification Accuracy
accuracy_tree_tri = sum(preds_tree_triphasic == true_labels_triphasic) / numel(true_labels_triphasic);
accuracy_svm_tri  = sum(preds_svm_triphasic  == true_labels_triphasic) / numel(true_labels_triphasic);
accuracy_knn_tri  = sum(preds_knn_triphasic  == true_labels_triphasic) / numel(true_labels_triphasic);

fprintf('\nTriphasic Classification Accuracy:\n');
fprintf('  Decision Tree: %.2f%%\n', accuracy_tree_tri * 100);
fprintf('  SVM:           %.2f%%\n', accuracy_svm_tri * 100);
fprintf('  k-NN:          %.2f%%\n', accuracy_knn_tri * 100);

%% Optional: Confusion Matrices Visualization
% Display confusion matrix for the decision tree classifier (binary classification)
figure;
confusionchart(true_labels_binary, preds_tree_binary);
title('Confusion Matrix: Decision Tree (Binary Classification)');

% Similarly, confusion charts can be created for the other classifiers or the triphasic classification task.
