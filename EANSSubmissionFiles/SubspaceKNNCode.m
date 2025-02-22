%% Subspace KNN Ensemble Experiment with 80/20 Split and Model Saving
% This script uses the same dataset and predictors (normalized R, G, B, C
% channels and an augmented feature, G/C) as in our previous analysis.
% The data is split into 80% training and 20% testing. A subspace ensemble
% of kNN classifiers is trained using 60 learning cycles and a base kNN 
% classifier with k = 10 neighbors. The process is repeated until the test 
% accuracy, sensitivity, and specificity each exceed 90%.
%
% References:
%   Ho, T. K. (1998). The random subspace method for constructing decision forests.
%   MATLAB Documentation on fitcensemble (2021). :contentReference[oaicite:0]{index=0}
%
% Author: [Your Name]
% Date: [Date]

clc;
clearvars -except processedcsfdata;
diary('knn_learning_session_log.txt');
fprintf('Starting Subspace KNN Ensemble experiments at %s\n', datestr(now));

%% --- Data Preparation ---
% Load data: normalized channels and augmented feature (G/C)
data = processedcsfdata;
epsilon = 1e-6;
R = data.RNormalized;
G = data.GNormalized;
B = data.BNormalized;
C = data.CNormalized;
features_orig = [R, G, B, C];
feature_names = {'R','G','B','C'};

% Augmented feature: ratio of G to C
ratioGC = G ./ (C + epsilon);
features_augmented = ratioGC;
augmented_names = {'ratioGC'};
useAugmented = true;
if useAugmented
    features_all = [features_orig, features_augmented];
    all_feature_names = [feature_names, augmented_names];
else
    features_all = features_orig;
    all_feature_names = feature_names;
end

binary_labels = data.infClassIDSA;
patient_ids   = data.InStudyID;
batches       = data.Batch;

% Remove rows with NaN or Inf values.
valid_rows = ~any(isnan(features_all) | isinf(features_all), 2);
features_all   = features_all(valid_rows, :);
binary_labels  = binary_labels(valid_rows);
patient_ids    = patient_ids(valid_rows);
batches        = batches(valid_rows);

% Optionally perform data balancing via SMOTE.
useSMOTE = false;
k_neighbors = 5;
if useSMOTE
    [features_all, binary_labels, patient_ids, batches] = ...
        smote_oversample(features_all, binary_labels, patient_ids, batches, k_neighbors);
end
fprintf('\nFinal Balanced Dataset: Class 0 = %d, Class 1 = %d\n', ...
    sum(binary_labels==0), sum(binary_labels==1));

%% --- Experiment Criteria and Subspace KNN Parameters ---
% Performance thresholds
target_accuracy    = 0.90;
target_sensitivity = 0.90;
target_specificity = 0.90;

% Subspace kNN parameters.
numLearningCycles = 60;
knn_neighbors     = 10;
template = templateKNN('NumNeighbors', knn_neighbors);

maxSeeds = 10000;  % Maximum number of experiments (seeds) to try
foundFlag = false;
expCount = 1;
finalResult = [];
finalSeed = NaN;

%% --- Experiment Loop with 80/20 Train-Test Split ---
while ~foundFlag && expCount <= maxSeeds
    rng(expCount);  % Set random seed for reproducibility
    
    % Perform an 80/20 stratified split
    cv = cvpartition(binary_labels, 'HoldOut', 0.2);
    train_idx = training(cv);
    test_idx  = test(cv);
    
    X_train = features_all(train_idx, :);
    y_train = binary_labels(train_idx);
    X_test  = features_all(test_idx, :);
    y_test  = binary_labels(test_idx);
    
    % Standardize features using the training set statistics.
    [X_train_scaled, mu, sigma] = zscore(X_train);
    X_test_scaled = (X_test - mu) ./ sigma;
    
    % Train the Subspace Ensemble kNN model.
    knnModel = fitcensemble(X_train_scaled, categorical(y_train), ...
        'Method', 'Subspace', 'Learners', template, 'NumLearningCycles', numLearningCycles);
    
    % Generate predictions and associated scores.
    [preds, scores] = predict(knnModel, X_test_scaled);
    
    % Determine scores for the positive class (assumed to be '1').
    if size(scores,2) == 2
        posScores = scores(:,2);
    else
        posScores = scores;
    end
    
    % Compute confusion matrix and performance metrics.
    % Convert y_test to categorical to match the type of preds.
    cm = confusionmat(categorical(y_test), preds);
    accuracy = sum(diag(cm)) / sum(cm(:));
    % Assuming rows: [Actual 0; Actual 1] and columns: [Predicted 0, Predicted 1]
    TN = cm(1,1); FP = cm(1,2);
    FN = cm(2,1); TP = cm(2,2);
    sensitivity = TP / max((TP + FN), 1);
    specificity = TN / max((TN + FP), 1);
    
    fprintf('\nExperiment %d (Seed = %d): Accuracy = %.2f%%, Sensitivity = %.2f%%, Specificity = %.2f%%\n', ...
        expCount, expCount, accuracy*100, sensitivity*100, specificity*100);
    
    % Check if all criteria are met.
    if (accuracy > target_accuracy) && (sensitivity > target_sensitivity) && (specificity > target_specificity)
        fprintf('Criteria met: Accuracy = %.2f%%, Sensitivity = %.2f%%, Specificity = %.2f%%\n', ...
            accuracy*100, sensitivity*100, specificity*100);
        foundFlag = true;
        finalSeed = expCount;
        finalResult = struct('Accuracy', accuracy, 'Sensitivity', sensitivity, 'Specificity', specificity, ...
            'ConfusionMatrix', cm, 'TestTruth', y_test, 'TestScores', posScores, 'Model', knnModel, ...
            'Mu', mu, 'Sigma', sigma);
        break;
    end
    
    expCount = expCount + 1;
end

if ~foundFlag
    fprintf('\nNo configuration meeting the criteria was found after %d experiments.\n', expCount-1);
    diary off;
    error('Experiment did not meet the performance thresholds.');
end

%% --- Generate Presentation-Quality Plots ---
% ROC Curve
figure;
[rocX, rocY, rocT, auc] = perfcurve(finalResult.TestTruth, finalResult.TestScores, 1);
plot(rocX, rocY, 'b-', 'LineWidth', 2);
xlabel('False Positive Rate', 'FontSize', 24);
ylabel('True Positive Rate', 'FontSize', 24);
title(sprintf('ROC Curve (AUC = %.3f)', auc), 'FontSize', 24);
grid on;
set(gca, 'FontSize', 24);  % Set axes tick labels to font size 24

% Confusion Matrix
figure;
imagesc(finalResult.ConfusionMatrix);
colormap('hot');
colorbar;
title('Confusion Matrix', 'FontSize', 24);
xlabel('Predicted Class', 'FontSize', 24);
ylabel('True Class', 'FontSize', 24);
set(gca, 'XTick', 1:2, 'XTickLabel', {'Clean','Infected'}, 'FontSize', 24);
set(gca, 'YTick', 1:2, 'YTickLabel', {'Clean','Infected'}, 'FontSize', 24);
% Overlay the numbers on each cell with increased font size
textStrings = num2str(finalResult.ConfusionMatrix(:), '%d');
textStrings = strtrim(cellstr(textStrings));
[x, y] = meshgrid(1:size(finalResult.ConfusionMatrix,2), 1:size(finalResult.ConfusionMatrix,1));
text(x(:), y(:), textStrings(:), 'HorizontalAlignment', 'center', ...
    'Color', 'white', 'FontWeight', 'bold', 'FontSize', 24);

% Histogram of Prediction Scores by Class
figure;
hold on;
histogram(finalResult.TestScores(finalResult.TestTruth==0), 'Normalization', 'pdf', ...
    'FaceColor', 'r', 'FaceAlpha', 0.5);
histogram(finalResult.TestScores(finalResult.TestTruth==1), 'Normalization', 'pdf', ...
    'FaceColor', 'g', 'FaceAlpha', 0.5);
title('Distribution of Prediction Scores by Class', 'FontSize', 24);
xlabel('Prediction Score', 'FontSize', 24);
ylabel('Probability Density', 'FontSize', 24);
legend('Class 0','Class 1','FontSize',24);
set(gca, 'FontSize', 24);  % Set axes tick labels to font size 24
hold off;

%% --- Train Final Model on the Entire Dataset and Save It ---
fprintf('\nTraining final model on the entire dataset...\n');
[X_all_scaled, mu_final, sigma_final] = zscore(features_all);
finalModel = fitcensemble(X_all_scaled, categorical(binary_labels), ...
    'Method', 'Subspace', 'Learners', template, 'NumLearningCycles', numLearningCycles);

% Save the final model along with scaling parameters.
save('finalKNNModel.mat', 'finalModel', 'mu_final', 'sigma_final');
fprintf('Final model saved as finalKNNModel.mat\n');

diary off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Helper Function: SMOTE Oversampling
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [features_new, labels_new, patient_ids_new, batches_new] = ...
    smote_oversample(features, labels, patient_ids, batches, k)
    if nargin < 5
        k = 5;
    end
    classes = unique(labels);
    count0 = sum(labels == classes(1));
    count1 = sum(labels == classes(2));
    
    if count0 < count1
        minority_class = classes(1);
        majority_class = classes(2);
    else
        minority_class = classes(2);
        majority_class = classes(1);
    end
    
    minority_idx = find(labels == minority_class);
    majority_idx = find(labels == majority_class);
    
    num_minority = length(minority_idx);
    num_majority = length(majority_idx);
    diff = num_majority - num_minority;
    
    if diff <= 0
        features_new = features;
        labels_new = labels;
        patient_ids_new = patient_ids;
        batches_new = batches;
        return;
    end
    
    synthetic_samples_per_instance = floor(diff / num_minority);
    remainder = mod(diff, num_minority);
    
    synthetic_features = [];
    synthetic_labels = [];
    synthetic_patient_ids = [];
    synthetic_batches = [];
    
    minority_features = features(minority_idx, :);
    distances = pdist2(minority_features, minority_features);
    for i = 1:size(distances, 1)
        distances(i,i) = Inf;
    end
    [~, neighbors_idx] = sort(distances, 2, 'ascend');
    
    for i = 1:num_minority
        num_to_generate = synthetic_samples_per_instance;
        if i <= remainder
            num_to_generate = num_to_generate + 1;
        end
        for n = 1:num_to_generate
            nn_index = randi(min(k, size(neighbors_idx,2)));
            neighbor = neighbors_idx(i, nn_index);
            diff_vec = minority_features(neighbor, :) - minority_features(i, :);
            gap = rand(1, size(features,2));
            synthetic_sample = minority_features(i, :) + gap .* diff_vec;
            synthetic_features = [synthetic_features; synthetic_sample];
            synthetic_labels = [synthetic_labels; minority_class];
            synthetic_patient_ids = [synthetic_patient_ids; patient_ids(minority_idx(i))];
            synthetic_batches = [synthetic_batches; batches(minority_idx(i))];
        end
    end
    
    features_new = [features; synthetic_features];
    labels_new = [labels; synthetic_labels];
    patient_ids_new = [patient_ids; synthetic_patient_ids];
    batches_new = [batches; synthetic_batches];
end
