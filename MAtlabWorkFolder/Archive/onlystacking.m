%% Only Stacking Ensemble with Selected Features (Revised)
% This script implements a stacking ensemble classifier using only R, G, and C 
% channels (with an augmented feature ratioGC). The B channel and features 
% involving B (B/C and R/C) have been removed.
%
% Author: [Your Name]
% Date: [Date]

clc;
clearvars -except processedcsfdata;
diary('training_log.txt');
fprintf('Starting stacking ensemble experiments at %s\n', datestr(now));

% Optionally suppress AdaBoostM1 warnings if desired:
warning('off', 'classreg:learning:modifier:AdaBoostM1:incrementNumIterations');

%% Check for Processed Data
if ~exist('processedcsfdata', 'var')
    error('processedcsfdata is not found in the workspace.');
end

%% User-Defined Parameters
useDynamicWindow = true;          % Enable dynamic window sizing
min_window_size    = 1;
max_window_size    = 24;
variance_threshold = 0.05;         % Threshold for window variance
windowLabelingMode = 'majority';   % Label windows using majority vote
useAugmented       = true;          % Use augmented features
performHyperparameterTuning = true;
hyperOptMethod     = 'bayesopt';
screeningMode      = true;         % Do not override threshold
screeningThreshold = 0.4;           % (Not used if screeningMode is false)
metaWeightFactor   = 3;             % Weight multiplier for infected class in meta-learning

%% Data Preparation and Feature Selection
data = processedcsfdata;
epsilon = 1e-6;

% Extract only R, G, and C channels (removing B)
R = data.RNormalized;
G = data.GNormalized;
C = data.CNormalized;
features_orig = [R, G, C];
feature_names = {'R', 'G', 'C'};

% Augmented feature: only ratioGC is retained (removing ratioRC and ratioBC)
ratioGC = G ./ (C + epsilon);
features_augmented = ratioGC;  % column vector
augmented_names = {'ratioGC'};

if useAugmented
    features_all = [features_orig, features_augmented];
    all_feature_names = [feature_names, augmented_names];
else
    features_all = features_orig;
    all_feature_names = feature_names;
end

% Extract labels and patient identifiers
binary_labels = data.infClassIDSA;
patient_ids   = data.InStudyID;
batches       = data.Batch;

% Remove rows with NaN or Inf values
valid_rows = ~any(isnan(features_all) | isinf(features_all), 2);
features_all   = features_all(valid_rows, :);
binary_labels  = binary_labels(valid_rows);
patient_ids    = patient_ids(valid_rows);
batches        = batches(valid_rows);

%% Data Balancing with SMOTE Oversampling
fprintf('\nBalancing the Dataset...\n');
useSMOTE = true;
if useSMOTE
    k_neighbors = 5;
    [features_all, binary_labels, patient_ids, batches] = ...
        smote_oversample(features_all, binary_labels, patient_ids, batches, k_neighbors);
end
fprintf('Final Balanced Dataset: Class 0 = %d, Class 1 = %d\n', ...
    sum(binary_labels==0), sum(binary_labels==1));

%% Stacking Ensemble Experiment (LOOCV over Patients)
fprintf('\nRunning stacking ensemble experiment...\n');

all_test_preds  = [];
all_test_truth  = [];
all_test_scores = [];  % store probability scores for threshold optimization

unique_patients = unique(patient_ids);
n_patients = length(unique_patients);

% Preallocate cell arrays for parallel computation
test_preds_cell  = cell(n_patients, 1);
test_truth_cell  = cell(n_patients, 1);
test_scores_cell = cell(n_patients, 1);

parfor i = 1:n_patients
    curr_patient = unique_patients(i);
    % Create dynamic rolling windows for each patient
    [windowed_features, windowed_labels, windowed_patient_ids, windowed_batches] = ...
        create_dynamic_rolling_window_per_patient(features_all, binary_labels, patient_ids, batches, ...
        min_window_size, max_window_size, variance_threshold, windowLabelingMode);
    
    % Select windows for the test patient
    patient_test_idx = (windowed_patient_ids == curr_patient);
    if sum(patient_test_idx) == 0
        test_preds_cell{i}  = [];
        test_truth_cell{i}  = [];
        test_scores_cell{i} = [];
        continue;
    end
    
    % Leave-one-patient-out: training on windows from other patients
    train_idx = ~patient_test_idx;
    X_train = windowed_features(train_idx, :);
    y_train = windowed_labels(train_idx);
    X_test  = windowed_features(patient_test_idx, :);
    y_test  = windowed_labels(patient_test_idx);
    
    % Standardize features based on training set statistics
    [X_train_scaled, mu, sigma] = zscore(X_train);
    X_test_scaled = (X_test - mu) ./ sigma;
    
    % Train the stacking ensemble
    model = trainStackingClassifier(X_train_scaled, y_train, performHyperparameterTuning, hyperOptMethod, metaWeightFactor);
    [preds, scores] = predictStackingClassifier(model, X_test_scaled);
    
    test_preds_cell{i}  = preds;
    test_truth_cell{i}  = y_test;
    test_scores_cell{i} = scores(:,2);  % probability for infected class
end

% Aggregate predictions, ground truths, and scores across patients
for i = 1:n_patients
    all_test_preds  = [all_test_preds; test_preds_cell{i}];
    all_test_truth  = [all_test_truth; test_truth_cell{i}];
    all_test_scores = [all_test_scores; test_scores_cell{i}];
end

% Optimize threshold using Youden's J statistic
optimized_threshold = optimizeThreshold(all_test_truth, all_test_scores);
if screeningMode
    used_threshold = screeningThreshold;
else
    used_threshold = optimized_threshold;
end
% Apply threshold to convert probability scores to binary predictions
all_test_preds = double(all_test_scores >= used_threshold);

% Evaluate performance metrics
binary_cm = confusionmat(all_test_truth, all_test_preds);
[precision, recall, f1] = calculate_metrics(binary_cm);
accuracy = sum(diag(binary_cm)) / sum(binary_cm(:));
TN = binary_cm(1,1); FP = binary_cm(1,2);
FN = binary_cm(2,1); TP = binary_cm(2,2);
sensitivity = TP / max((TP+FN), 1);
specificity = TN / max((TN+FP), 1);

fprintf(['Stacking Ensemble Results:\n' ...
         'Optimized Threshold: %.2f | Accuracy: %.2f | F1 (Clean): %.4f | F1 (Infected): %.4f\n' ...
         'Sensitivity: %.4f | Specificity: %.4f\n'], ...
         used_threshold, accuracy, f1(1), f1(2), sensitivity, specificity);

% Store and export the performance metrics
results = struct('Method', 'Stacking', ...
    'OptimizedThreshold', optimized_threshold, ...
    'UsedThreshold', used_threshold, ...
    'Accuracy', accuracy, ...
    'F1_Clean', f1(1), ...
    'F1_Infected', f1(2), ...
    'ConfusionMatrix', binary_cm, ...
    'Sensitivity', sensitivity, ...
    'Specificity', specificity);
% Wrap the confusion matrix in a cell to ensure consistent dimensions
results.ConfusionMatrix = {results.ConfusionMatrix};
resultsTable = struct2table(results, 'AsArray', true);
writetable(resultsTable, 'stacking_performance_metrics.csv');
fprintf('\nPerformance metrics exported to stacking_performance_metrics.csv\n');

%% Augmented Feature Visualization
if useAugmented
    figure;
    subplot(1,2,1);
    boxplot(features_orig, 'Labels', feature_names);
    title('Original Features (R, G, C)');
    grid on;
    subplot(1,2,2);
    boxplot(features_augmented, 'Labels', augmented_names);
    title('Augmented Feature (ratioGC)');
    grid on;
    suptitle('Comparison of Original and Augmented Feature Distributions');
end

%% Patient Data Visualization
visualize_patient_data(features_all, binary_labels, patient_ids, all_feature_names);

diary off;

%% Helper Functions

function out = ternary(condition, trueVal, falseVal)
    if condition
        out = trueVal;
    else
        out = falseVal;
    end
end

function [precision, recall, f1] = calculate_metrics(cm)
    n_classes = size(cm, 1);
    precision = zeros(n_classes, 1);
    recall = zeros(n_classes, 1);
    f1 = zeros(n_classes, 1);
    for i = 1:n_classes
        tp = cm(i, i);
        fp = sum(cm(:, i)) - tp;
        fn = sum(cm(i, :)) - tp;
        precision(i) = tp / max(tp + fp, 1);
        recall(i) = tp / max(tp + fn, 1);
        f1(i) = 2 * (precision(i) * recall(i)) / max(precision(i) + recall(i), 1);
    end
end

function bestThreshold = optimizeThreshold(truth, scores)
    thresholds = 0:0.01:1;
    bestMetric = -Inf;
    bestThreshold = 0.5; % default value
    for t = thresholds
        preds = double(scores >= t);
        cm = confusionmat(truth, preds);
        if size(cm, 1) < 2
            continue;
        end
        TP = cm(2, 2);
        TN = cm(1, 1);
        FP = cm(1, 2);
        FN = cm(2, 1);
        sensitivity = TP / max((TP + FN), 1);
        specificity = TN / max((TN + FP), 1);
        J = sensitivity + specificity - 1;
        if J > bestMetric
            bestMetric = J;
            bestThreshold = t;
        end
    end
    fprintf('Optimized threshold: %.2f with Youden''s J: %.4f\n', bestThreshold, bestMetric);
end

function [windowed_features, windowed_labels, windowed_patient_ids, windowed_batches] = ...
    create_dynamic_rolling_window_per_patient(features, labels, patient_ids, batches, minSize, maxSize, varThreshold, labelingMode)
    windowed_features = [];
    windowed_labels = [];
    windowed_patient_ids = [];
    windowed_batches = [];
    
    unique_patients = unique(patient_ids);
    for p = 1:length(unique_patients)
        curr_patient = unique_patients(p);
        idx = find(patient_ids == curr_patient);
        [~, sortOrder] = sort(batches(idx));
        idx_sorted = idx(sortOrder);
        
        curr_features = features(idx_sorted, :);
        curr_labels = labels(idx_sorted);
        curr_batches = batches(idx_sorted);
        n = size(curr_features, 1);
        
        i = 1;
        while i <= n - minSize + 1
            window_end = i + minSize - 1;
            dynamic_window = curr_features(i:window_end, :);
            while (window_end < n) && ((window_end - i + 1) < maxSize)
                current_std = std(dynamic_window(:));
                if current_std > varThreshold
                    break;
                end
                window_end = window_end + 1;
                dynamic_window = curr_features(i:window_end, :);
            end
            window_length = window_end - i + 1;
            if window_length < minSize
                i = i + 1;
                continue;
            end
            window = reshape(curr_features(i:window_end, :)', 1, []);
            switch labelingMode
                case 'union'
                    label = double(any(curr_labels(i:window_end)==1));
                case 'majority'
                    label = double(mean(curr_labels(i:window_end)) >= 0.5);
                otherwise
                    error('Unsupported labeling mode: %s', labelingMode);
            end
            batch_val = curr_batches(window_end);
            windowed_features = [windowed_features; window];
            windowed_labels = [windowed_labels; label];
            windowed_patient_ids = [windowed_patient_ids; curr_patient];
            windowed_batches = [windowed_batches; batch_val];
            i = window_end + 1;
        end
    end
end

function model = trainStackingClassifier(X_train, y_train, performHyperparameterTuning, hyperOptMethod, metaWeightFactor)
    % Train base learners using 5-fold cross-validation to generate meta-features
    baseMethods = {'GentleBoost', 'AdaBoostM1', 'Bag'};
    baseModels = cell(length(baseMethods), 1);
    metaFeatures = zeros(size(X_train, 1), length(baseMethods));
    
    cvp = cvpartition(y_train, 'KFold', 5);
    for m = 1:length(baseMethods)
        temp_meta = zeros(size(X_train, 1), 1);
        for fold = 1:cvp.NumTestSets
            trIdx = cvp.training(fold);
            valIdx = cvp.test(fold);
            baseModel = trainClassifier(X_train(trIdx, :), y_train(trIdx), baseMethods{m}, struct(), performHyperparameterTuning, hyperOptMethod);
            [~, scores] = predict(baseModel, X_train(valIdx, :));
            temp_meta(valIdx) = scores(:, 2);
        end
        metaFeatures(:, m) = temp_meta;
        baseModels{m} = trainClassifier(X_train, y_train, baseMethods{m}, struct(), performHyperparameterTuning, hyperOptMethod);
    end
    
    % Assign a higher weight to infected cases in the meta-learner
    weights = ones(size(y_train));
    weights(y_train == 1) = metaWeightFactor;
    metaModel = fitglm(metaFeatures, y_train, 'Distribution', 'binomial', 'Weights', weights);
    
    model.baseModels = baseModels;
    model.metaModel = metaModel;
end

function [preds, scores] = predictStackingClassifier(model, X_test)
    nBases = length(model.baseModels);
    metaFeatures = zeros(size(X_test, 1), nBases);
    for m = 1:nBases
        [~, baseScores] = predict(model.baseModels{m}, X_test);
        metaFeatures(:, m) = baseScores(:, 2);
    end
    probs = predict(model.metaModel, metaFeatures);
    scores = [1 - probs, probs];  % probability scores for class 0 and class 1
    preds = double(probs >= 0.5);
end

function model = trainClassifier(X_train, y_train, method, params, performHyperparameterTuning, hyperOptMethod)
    switch method
        case 'GentleBoost'
            if ~isfield(params, 'NumLearningCycles')
                params.NumLearningCycles = 50;
            end
            if ~isfield(params, 'LearnRate')
                params.LearnRate = 0.1;
            end
            if performHyperparameterTuning
                opts = struct('ShowPlots', false, 'Verbose', 0, 'Optimizer', hyperOptMethod);
                model = fitcensemble(X_train, y_train, 'Method', 'GentleBoost', ...
                    'NumLearningCycles', params.NumLearningCycles, ...
                    'LearnRate', params.LearnRate, ...
                    'OptimizeHyperparameters', 'all', ...
                    'HyperparameterOptimizationOptions', opts);
            else
                model = fitcensemble(X_train, y_train, 'Method', 'GentleBoost', ...
                    'NumLearningCycles', params.NumLearningCycles, ...
                    'LearnRate', params.LearnRate);
            end
        case 'AdaBoostM1'
            if performHyperparameterTuning
                opts = struct('ShowPlots', false, 'Verbose', 0, 'Optimizer', hyperOptMethod);
                model = fitcensemble(X_train, y_train, 'Method', 'AdaBoostM1', ...
                    'OptimizeHyperparameters', 'all', ...
                    'HyperparameterOptimizationOptions', opts);
            else
                model = fitcensemble(X_train, y_train, 'Method', 'AdaBoostM1');
            end
        case 'Bag'
            if performHyperparameterTuning
                opts = struct('ShowPlots', false, 'Verbose', 0, 'Optimizer', hyperOptMethod);
                model = fitcensemble(X_train, y_train, 'Method', 'Bag', ...
                    'OptimizeHyperparameters', 'all', ...
                    'HyperparameterOptimizationOptions', opts);
            else
                model = fitcensemble(X_train, y_train, 'Method', 'Bag');
            end
        otherwise
            error('Unsupported model method: %s', method);
    end
end

function visualize_patient_data(features, binary_labels, patient_ids, feature_names)
    n_feats = size(features, 2);
    
    % Boxplots of feature distributions by infection status
    figure('Position', [100 100 1200 800]);
    num_rows = ceil(n_feats / 2);
    for i = 1:n_feats
        subplot(num_rows, 2, i);
        clean_data = features(binary_labels == 0, i);
        infected_data = features(binary_labels == 1, i);
        boxplot([clean_data; infected_data], ...
                [zeros(size(clean_data)); ones(size(infected_data))], ...
                'Colors', 'br', 'Labels', {'Clean','Infected'});
        title(feature_names{i}, 'Interpreter', 'none');
        grid on;
    end
    sgtitle('Feature Distributions by Infection Status');
    
    % PCA Projection and Feature Correlation
    figure;
    subplot(2, 1, 1);
    [~, scores] = pca(zscore(features));
    scatter(scores(:, 1), scores(:, 2), 30, binary_labels, 'filled');
    title('PCA Projection (PC1 vs. PC2)');
    colorbar;
    
    subplot(2, 1, 2);
    imagesc(corr(features));
    colorbar;
    title('Feature Correlation Matrix');
    xticks(1:n_feats);
    yticks(1:n_feats);
    xticklabels(feature_names);
    yticklabels(feature_names);
    xtickangle(45);
    
    % Patient Trajectory Visualization
    figure('Position', [100 100 1200 800]);
    unique_patients = unique(patient_ids);
    nPatients = length(unique_patients);
    num_patient_rows = ceil(nPatients / 2);
    
    for p = 1:nPatients
        subplot(num_patient_rows, 2, p);
        idx = (patient_ids == unique_patients(p));
        patient_data = features(idx, :);
        patient_labels = binary_labels(idx);
        num_timepoints = size(patient_data, 1);
        hold on;
        h = plot(patient_data, 'LineWidth', 1.5);
        yl = ylim;
        for t = 1:num_timepoints
            if patient_labels(t) == 1
                patch([t-0.5, t+0.5, t+0.5, t-0.5], ...
                      [yl(1) yl(1) yl(2) yl(2)], ...
                      'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
            end
        end
        hold off;
        title(sprintf('Patient %d', unique_patients(p)));
        legend(h, feature_names, 'Location', 'eastoutside', 'Interpreter', 'none');
        grid on;
    end
    sgtitle('Patient Trajectories with Infected Regions Highlighted');
end

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
        distances(i, i) = Inf;
    end
    [~, neighbors_idx] = sort(distances, 2, 'ascend');
    
    for i = 1:num_minority
        num_to_generate = synthetic_samples_per_instance;
        if i <= remainder
            num_to_generate = num_to_generate + 1;
        end
        for n = 1:num_to_generate
            nn_index = randi(min(k, size(neighbors_idx, 2)));
            neighbor = neighbors_idx(i, nn_index);
            diff_vec = minority_features(neighbor, :) - minority_features(i, :);
            gap = rand(1, size(features, 2));
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
