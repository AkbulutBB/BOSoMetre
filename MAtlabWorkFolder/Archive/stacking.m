%% Ensemble Model Comparison with Multiple Experiments (Revised)
% This script implements LOOCV with both fixed and dynamic rolling windows,
% oversampling (SMOTE), and evaluation of different models and hyperparameter 
% configurations. In addition, it supports:
%   - Dynamic window sizing based on local variability.
%   - Augmented feature visualization to reduce device-related scale differences.
%   - Alternative hyperparameter optimization (e.g., bayesopt, randomsearch).
%   - Decision threshold optimization via ROC analysis (Youden's J statistic).
%   - A stacking ensemble that combines predictions of base learners.
%   - Two labeling strategies for rolling windows: union (pure) and majority vote.
%
% Additionally, a "screening mode" is implemented: when enabled, the optimized
% threshold is overridden by a lower value (screeningThreshold) to favor higher sensitivity.
%
% For the stacking ensemble, the meta-learner is trained using weighted logistic 
% regression. The infected class (class 1) is given a higher weight (metaWeightFactor)
% to further favor sensitivity.
%
% Author: [Your Name]
% Date: [Date]

clc;
clearvars -except processedcsfdata

diary('training_log.txt');
fprintf('Starting experiments at %s\n', datestr(now));

%% Check for processed data
if ~exist('processedcsfdata', 'var')
    error('processedcsfdata is not found in the workspace.');
end

%% User-Defined Parameters
performHyperparameterTuning = true;  % Toggle hyperparameter tuning
hyperOptMethod = 'bayesopt';         % Options: 'bayesopt' or 'randomsearch'
useSMOTE = true;                     % Enable SMOTE oversampling
% Initial infection threshold (will be optimized later)
infection_threshold = 0.4;
% Fixed window sizes for comparison (if not using dynamic windows)
window_sizes = [1, 2, 4, 6, 8, 12, 24];

% Dynamic window parameters
useDynamicWindow = true;    % Toggle dynamic window sizing
min_window_size = 1;        % Changed to 1, as requested
max_window_size = 24;
variance_threshold = 0.05;  % Threshold on standard deviation of features

% Window labeling mode: 'union' (if any block infected => label 1)
% or 'majority' (label according to majority vote within the window)
windowLabelingMode = 'majority';

% Option to use augmented features (e.g., ratios)
useAugmented = true;

% Toggle stacking ensemble (this is our focus)
useStacking = true;

% --- New parameters for screening mode ---
screeningMode = true;         % If true, override the optimized threshold with a lower value
screeningThreshold = 0.5;     % Lower threshold to favor sensitivity

% --- New parameter for cost-sensitive meta-learner in stacking ---
metaWeightFactor = 2;         % Weight multiplier for infected class in meta-learner training

%% Define Experiments (Models & Parameter Configurations)
% Define base classifier configurations.
gentleBoostParams = { ...
    struct('Name','GentleBoost_50cycles_0.1', 'NumLearningCycles',50, 'LearnRate',0.1), ...
    struct('Name','GentleBoost_100cycles_0.1','NumLearningCycles',100,'LearnRate',0.1), ...
    struct('Name','GentleBoost_50cycles_0.05','NumLearningCycles',50, 'LearnRate',0.05) ...
};
adaBoostParams = { ...
    struct('Name','AdaBoost_default','Optimize',performHyperparameterTuning) ...
};
randomForestParams = { ...
    struct('Name','RandomForest_default','Optimize',performHyperparameterTuning) ...
};

% Combine base experiments
experiments = {};
for i = 1:length(gentleBoostParams)
    experiments{end+1} = struct('method', 'GentleBoost', 'params', gentleBoostParams{i});
end
for i = 1:length(adaBoostParams)
    experiments{end+1} = struct('method', 'AdaBoostM1', 'params', adaBoostParams{i});
end
for i = 1:length(randomForestParams)
    experiments{end+1} = struct('method', 'Bag', 'params', randomForestParams{i});
end
% Add stacking ensemble experiment if enabled
if useStacking
    stackingParams = struct('Name', 'Stacking_Ensemble');
    experiments{end+1} = struct('method', 'Stacking', 'params', stackingParams);
end

%% Data Preparation and Feature Selection
data = processedcsfdata;
batches = data.Batch;
epsilon = 1e-6;

% Original features
R = data.RNormalized;
G = data.GNormalized;
B = data.BNormalized;
C = data.CNormalized;
features_orig = [R, G, B, C];
feature_names = {'R','G','B','C'};

% Augmented features (ratios)
ratioRC = R ./ (C + epsilon);
ratioGC = G ./ (C + epsilon);
ratioBC = B ./ (C + epsilon);
features_augmented = [ratioRC, ratioGC, ratioBC];
augmented_names = {'ratioRC','ratioGC', 'ratioBC'};

if useAugmented
    features_all = [features_orig, features_augmented];
    all_feature_names = [feature_names, augmented_names];
else
    features_all = features_orig;
    all_feature_names = feature_names;
end

% Extract labels and patient identifiers
binary_labels = data.infClassIDSA;
patient_ids = data.InStudyID;

% Remove rows with NaN or Inf values
valid_rows = ~any(isnan(features_all) | isinf(features_all), 2);
features_all = features_all(valid_rows, :);
binary_labels = binary_labels(valid_rows);
patient_ids = patient_ids(valid_rows);
batches = batches(valid_rows);

%% Data Balancing with SMOTE Oversampling
fprintf('\nBalancing the Dataset...\n');
if useSMOTE
    k_neighbors = 5;
    [features_all, binary_labels, patient_ids, batches] = ...
        smote_oversample(features_all, binary_labels, patient_ids, batches, k_neighbors);
else
    % Simple random oversampling if SMOTE is disabled
    class0_idx = find(binary_labels == 0);
    class1_idx = find(binary_labels == 1);
    max_samples = max(length(class0_idx), length(class1_idx));
    rng(42);
    class0_oversampled = class0_idx(randi(length(class0_idx), max_samples, 1));
    class1_oversampled = class1_idx(randi(length(class1_idx), max_samples, 1));
    balanced_idx = [class0_oversampled; class1_oversampled];
    features_all = features_all(balanced_idx, :);
    binary_labels = binary_labels(balanced_idx);
    patient_ids = patient_ids(balanced_idx);
    batches = batches(balanced_idx);
end
fprintf('Final Balanced Dataset: Class 0 = %d, Class 1 = %d\n', ...
    sum(binary_labels == 0), sum(binary_labels == 1));

%% Experiment Loop: Evaluate Each Model and Parameter Configuration
allResults = [];
for exp = 1:length(experiments)
    currentExperiment = experiments{exp};
    method = currentExperiment.method;
    params = currentExperiment.params;
    expName = params.Name;
    fprintf('\nRunning experiment: %s (Method: %s)\n', expName, method);
    
    % For each experiment, aggregate predictions and scores across patients.
    all_test_preds = [];
    all_test_truth = [];
    all_test_scores = [];  % store probability scores for threshold optimization
    
    unique_patients = unique(patient_ids);
    n_patients = length(unique_patients);
    
    % Preallocate cell arrays for parallel computation
    test_preds_cell = cell(n_patients,1);
    test_truth_cell = cell(n_patients,1);
    test_scores_cell = cell(n_patients,1);
    
    parfor i = 1:n_patients
        curr_patient = unique_patients(i);
        test_idx = (patient_ids == curr_patient);
        % Depending on windowing strategy, call the appropriate function
        if useDynamicWindow
            [windowed_features, windowed_labels, windowed_patient_ids, windowed_batches] = ...
                create_dynamic_rolling_window_per_patient(features_all, binary_labels, patient_ids, batches, ...
                min_window_size, max_window_size, variance_threshold, windowLabelingMode);
        else
            [windowed_features, windowed_labels, windowed_patient_ids, windowed_batches] = ...
                create_rolling_window_per_patient(features_all, binary_labels, patient_ids, batches, window_sizes(end), windowLabelingMode);
        end
        
        % For test patient, select windows corresponding to this patient
        patient_test_idx = (windowed_patient_ids == curr_patient);
        if sum(patient_test_idx) == 0
            test_preds_cell{i} = [];
            test_truth_cell{i} = [];
            test_scores_cell{i} = [];
            continue;
        end
        % Leave-one-patient-out: training on windows from other patients
        train_idx = ~patient_test_idx;
        X_train = windowed_features(train_idx, :);
        y_train = windowed_labels(train_idx);
        X_test = windowed_features(patient_test_idx, :);
        y_test = windowed_labels(patient_test_idx);
        
        % Standardize features based on training set statistics
        [X_train_scaled, mu, sigma] = zscore(X_train);
        X_test_scaled = (X_test - mu) ./ sigma;
        
        % Train classifier (or stacking ensemble) using a modular function
        if strcmp(method, 'Stacking')
            model = trainStackingClassifier(X_train_scaled, y_train, performHyperparameterTuning, hyperOptMethod, metaWeightFactor);
            [preds, scores] = predictStackingClassifier(model, X_test_scaled);
        else
            model = trainClassifier(X_train_scaled, y_train, method, params, performHyperparameterTuning, hyperOptMethod);
            [~, scores] = predict(model, X_test_scaled);
            preds = double(scores(:,2) >= infection_threshold);
        end
        
        test_preds_cell{i} = preds;
        test_truth_cell{i} = y_test;
        test_scores_cell{i} = scores(:,2);  % probability for infected class
    end
    
    % Aggregate predictions, ground truths, and scores over patients
    for i = 1:n_patients
        all_test_preds = [all_test_preds; test_preds_cell{i}];
        all_test_truth = [all_test_truth; test_truth_cell{i}];
        all_test_scores = [all_test_scores; test_scores_cell{i}];
    end
    
    % Optimize threshold based on Youden's J statistic over the test predictions
    optimized_threshold = optimizeThreshold(all_test_truth, all_test_scores);
    % If screening mode is enabled, override the threshold to favor sensitivity.
    if screeningMode
        used_threshold = screeningThreshold;
    else
        used_threshold = optimized_threshold;
    end
    % Recompute binary predictions using the chosen threshold.
    all_test_preds = double(all_test_scores >= used_threshold);
    
    % Evaluate performance metrics
    binary_cm = confusionmat(all_test_truth, all_test_preds);
    [precision, recall, f1] = calculate_metrics(binary_cm);
    accuracy = sum(diag(binary_cm)) / sum(binary_cm(:));
    TN = binary_cm(1,1); FP = binary_cm(1,2);
    FN = binary_cm(2,1); TP = binary_cm(2,2);
    sensitivity = TP / max((TP + FN), 1);
    specificity = TN / max((TN + FP), 1);
    FP_rate = FP / max((TN + FP), 1);
    FN_rate = FN / max((TP + FN), 1);
    
    % Store the results for the current experiment
    allResults = [allResults; struct(...
        'Experiment', expName, ...
        'Method', method, ...
        'Windowing', ternary(useDynamicWindow, 'Dynamic', 'Fixed'), ...
        'LabelingMode', windowLabelingMode, ...
        'OptimizedThreshold', optimized_threshold, ...
        'UsedThreshold', used_threshold, ...
        'Accuracy', accuracy, ...
        'F1_Clean', f1(1), ...
        'F1_Infected', f1(2), ...
        'ConfusionMatrix', {binary_cm}, ...
        'Sensitivity', sensitivity, ...
        'Specificity', specificity, ...
        'FP_rate', FP_rate, ...
        'FN_rate', FN_rate)];
    
    fprintf('Exp: %s, Method: %s, Windowing: %s, Labeling: %s, Used Threshold: %.2f | Accuracy: %.2f | F1 (Clean): %.4f | F1 (Infected): %.4f\n', ...
        expName, method, ternary(useDynamicWindow, 'Dynamic', 'Fixed'), windowLabelingMode, used_threshold, accuracy, f1(1), f1(2));
end

%% Export Aggregated Performance Metrics
metricsTable = struct2table(allResults);
writetable(metricsTable, 'ensemble_performance_metrics.csv');
fprintf('\nPerformance metrics exported to ensemble_performance_metrics.csv\n');

%% Augmented Feature Visualization
if useAugmented
    figure;
    subplot(1,2,1);
    boxplot(features_orig, 'Labels', feature_names);
    title('Original Features');
    grid on;
    subplot(1,2,2);
    boxplot(features_augmented, 'Labels', augmented_names);
    title('Augmented Ratio Features');
    grid on;
    sgtitle('Comparison of Original and Augmented Feature Distributions');
end

%% Patient Data Visualization (including augmented features)
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

function model = trainClassifier(X_train, y_train, method, params, performHyperparameterTuning, hyperOptMethod)
    switch method
        case 'GentleBoost'
            % Set default values if not provided
            if ~isfield(params, 'NumLearningCycles')
                params.NumLearningCycles = 50;  % Default value
            end
            if ~isfield(params, 'LearnRate')
                params.LearnRate = 0.1;         % Default value
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
            if performHyperparameterTuning && isfield(params, 'Optimize') && params.Optimize
                opts = struct('ShowPlots', false, 'Verbose', 0, 'Optimizer', hyperOptMethod);
                model = fitcensemble(X_train, y_train, 'Method', 'AdaBoostM1', ...
                    'OptimizeHyperparameters', 'all', ...
                    'HyperparameterOptimizationOptions', opts);
            else
                model = fitcensemble(X_train, y_train, 'Method', 'AdaBoostM1');
            end
        case 'Bag'
            if performHyperparameterTuning && isfield(params, 'Optimize') && params.Optimize
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

function [precision, recall, f1] = calculate_metrics(cm)
    % CALCULATE_METRICS computes precision, recall, and F1-score for each class.
    n_classes = size(cm, 1);
    precision = zeros(n_classes, 1);
    recall = zeros(n_classes, 1);
    f1 = zeros(n_classes, 1);
    for i = 1:n_classes
        tp = cm(i,i);
        fp = sum(cm(:,i)) - tp;
        fn = sum(cm(i,:)) - tp;
        precision(i) = tp / max(tp + fp, 1);
        recall(i) = tp / max(tp + fn, 1);
        f1(i) = 2 * (precision(i) * recall(i)) / max(precision(i) + recall(i), 1);
    end
end

function bestThreshold = optimizeThreshold(truth, scores)
    % OPTIMIZETHRESHOLD searches over a range of thresholds to maximize Youden's J statistic.
    thresholds = 0:0.01:1;
    bestMetric = -Inf;
    bestThreshold = 0.5; % default
    for t = thresholds
        preds = double(scores >= t);
        cm = confusionmat(truth, preds);
        if size(cm,1) < 2
            continue;
        end
        TP = cm(2,2);
        TN = cm(1,1);
        FP = cm(1,2);
        FN = cm(2,1);
        sensitivity = TP / max((TP+FN),1);
        specificity = TN / max((TN+FP),1);
        J = sensitivity + specificity - 1;
        if J > bestMetric
            bestMetric = J;
            bestThreshold = t;
        end
    end
    fprintf('Optimized threshold: %.2f with Youden''s J: %.4f\n', bestThreshold, bestMetric);
end

function [windowed_features, windowed_labels, windowed_patient_ids, windowed_batches] = ...
    create_rolling_window_per_patient(features, labels, patient_ids, batches, window_size, labelingMode)
    % CREATE_ROLLING_WINDOW_PER_PATIENT creates fixed rolling windows.
    % labelingMode: 'union' or 'majority'
    
    windowed_features = [];
    windowed_labels = [];
    windowed_patient_ids = [];
    windowed_batches = [];
    unique_patients = unique(patient_ids);
    
    for p = 1:length(unique_patients)
        curr_patient = unique_patients(p);
        idx = find(patient_ids == curr_patient);
        [sorted_batches, sortOrder] = sort(batches(idx));
        idx_sorted = idx(sortOrder);
        
        curr_features = features(idx_sorted, :);
        curr_labels = labels(idx_sorted);
        curr_batches = batches(idx_sorted);
        n_rows = size(curr_features, 1);
        if n_rows < window_size
            continue;
        end
        
        for j = 1:(n_rows - window_size + 1)
            window = reshape(curr_features(j:j+window_size-1, :)', 1, []);
            switch labelingMode
                case 'union'
                    label = double(any(curr_labels(j:j+window_size-1) == 1));
                case 'majority'
                    label = double(mean(curr_labels(j:j+window_size-1)) >= 0.5);
                otherwise
                    error('Unsupported labeling mode: %s', labelingMode);
            end
            batch_val = curr_batches(j+window_size-1);
            windowed_features = [windowed_features; window];
            windowed_labels = [windowed_labels; label];
            windowed_patient_ids = [windowed_patient_ids; curr_patient];
            windowed_batches = [windowed_batches; batch_val];
        end
    end
end

function [windowed_features, windowed_labels, windowed_patient_ids, windowed_batches] = ...
    create_dynamic_rolling_window_per_patient(features, labels, patient_ids, batches, minSize, maxSize, varThreshold, labelingMode)
    % CREATE_DYNAMIC_ROLLING_WINDOW_PER_PATIENT creates windows with lengths that
    % adapt based on the local standard deviation of the features.
    %
    % For each patient, we start at index i and increase the window until the 
    % standard deviation (computed on all features) exceeds varThreshold or
    % the window reaches maxSize. If the window size is below minSize, it is skipped.
    
    windowed_features = [];
    windowed_labels = [];
    windowed_patient_ids = [];
    windowed_batches = [];
    
    unique_patients = unique(patient_ids);
    for p = 1:length(unique_patients)
        curr_patient = unique_patients(p);
        idx = find(patient_ids == curr_patient);
        % Sort indices according to time (assuming batches indicate order)
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
            % Increase window size until variance exceeds threshold or maxSize is reached
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
            % Labeling based on the chosen mode
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
    % TRAINSTACKINGCLASSIFIER trains a stacking ensemble that combines three base
    % classifiers (GentleBoost, AdaBoostM1, and Bag) using logistic regression as meta-learner.
    % A simple 5-fold cross-validation is used to generate meta-features.
    %
    % The meta-learner is trained with weighted logistic regression: the infected class
    % (class 1) is given extra weight (metaWeightFactor) to improve sensitivity.
    
    baseMethods = {'GentleBoost', 'AdaBoostM1', 'Bag'};
    baseModels = cell(length(baseMethods),1);
    % Preallocate meta-features matrix
    metaFeatures = zeros(size(X_train,1), length(baseMethods));
    
    cvp = cvpartition(y_train, 'KFold', 5);
    for m = 1:length(baseMethods)
        temp_meta = zeros(size(X_train,1),1);
        for fold = 1:cvp.NumTestSets
            trIdx = cvp.training(fold);
            valIdx = cvp.test(fold);
            % Use default parameters for base models (could be tuned further)
            baseModel = trainClassifier(X_train(trIdx,:), y_train(trIdx), baseMethods{m}, struct(), performHyperparameterTuning, hyperOptMethod);
            [~, scores] = predict(baseModel, X_train(valIdx,:));
            temp_meta(valIdx) = scores(:,2);
        end
        metaFeatures(:,m) = temp_meta;
        % Train final base model on full training data
        baseModels{m} = trainClassifier(X_train, y_train, baseMethods{m}, struct(), performHyperparameterTuning, hyperOptMethod);
    end
    % Calculate weights for the meta-learner: assign a higher weight to infected cases.
    weights = ones(size(y_train));
    weights(y_train == 1) = metaWeightFactor;
    % Train meta-classifier (logistic regression) using weighted training
    metaModel = fitglm(metaFeatures, y_train, 'Distribution', 'binomial', 'Weights', weights);
    % Store the meta-model and base models in a structure
    model.baseModels = baseModels;
    model.metaModel = metaModel;
end

function [preds, scores] = predictStackingClassifier(model, X_test)
    % PREDICTSTACKINGCLASSIFIER generates predictions using the stacking ensemble.
    nBases = length(model.baseModels);
    metaFeatures = zeros(size(X_test,1), nBases);
    for m = 1:nBases
        [~, baseScores] = predict(model.baseModels{m}, X_test);
        metaFeatures(:,m) = baseScores(:,2);
    end
    % Obtain probability from meta-model
    probs = predict(model.metaModel, metaFeatures);
    scores = [1-probs, probs];  % [score for class 0, score for class 1]
    preds = double(probs >= 0.5);
end

function analyze_model_behavior(features, binary_labels, patient_ids, batches, window_size, feature_names)
    % ANALYZE_MODEL_BEHAVIOR performs variable importance analysis.
    [windowed_features, windowed_labels, ~, ~, ~] = ...
        create_rolling_window_per_patient(features, binary_labels, patient_ids, batches, window_size, 'union');
    
    [windowed_features_scaled, ~, ~] = zscore(windowed_features);
    model = fitcensemble(windowed_features_scaled, windowed_labels, 'Method', 'Bag');
    imp = predictorImportance(model);
    
    num_original_feats = length(feature_names);
    total_feats_in_window = size(windowed_features_scaled, 2);
    if mod(total_feats_in_window, num_original_feats) == 0
        times_per_feature = total_feats_in_window / num_original_feats;
        imp_reshaped = reshape(imp, num_original_feats, times_per_feature);
    else
        imp_reshaped = imp(:);
    end
    
    figure;
    if size(imp_reshaped, 2) > 1
        % Generate time labels using sprintf and arrayfun instead of using the plus operator
        timeLabels = arrayfun(@(x) sprintf('T-%d', x), 0:(size(imp_reshaped,2)-1), 'UniformOutput', false);
        heatmap(feature_names, timeLabels, imp_reshaped');
        title(sprintf('Variable Importance (Window Size: %d)', window_size));
    else
        bar(imp_reshaped);
        xticks(1:num_original_feats);
        xticklabels(feature_names);
        title(sprintf('Variable Importance (Window Size: %d)', window_size));
        ylabel('Importance');
    end
    
    if size(imp_reshaped, 2) > 1
        avg_importance = mean(imp_reshaped, 2);
        importance_table = table(feature_names(:), avg_importance, ...
            'VariableNames', {'Feature', 'AverageImportance'});
        csv_filename = sprintf('variable_importance_window%d.csv', window_size);
        writetable(importance_table, csv_filename);
        fprintf('Variable importance exported to %s\n', csv_filename);
    else
        importance_table = table(feature_names(:), imp_reshaped, ...
            'VariableNames', {'Feature','Importance'});
        csv_filename = sprintf('variable_importance_window%d.csv', window_size);
        writetable(importance_table, csv_filename);
        fprintf('Variable importance exported to %s\n', csv_filename);
    end
end

function visualize_patient_data(features, binary_labels, patient_ids, feature_names)
    % VISUALIZE_PATIENT_DATA produces boxplots, PCA projections, and patient trajectory plots.
    n_feats = size(features, 2);
    
    % Boxplots of Feature Distributions by Infection Status
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
    subplot(2,1,1);
    [~, scores] = pca(zscore(features));
    scatter(scores(:,1), scores(:,2), 30, binary_labels, 'filled');
    title('PCA Projection (PC1 vs. PC2)');
    colorbar;
    
    subplot(2,1,2);
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
    % SMOTE_OVERSAMPLE implements SMOTE to balance an imbalanced dataset.
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
