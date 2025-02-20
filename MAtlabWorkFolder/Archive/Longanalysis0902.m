%% Longanalysis0902.m
% This script performs a parameter sweep over different metaWeightFactor and 
% screeningThreshold values using a stacking ensemble classifier. Only the R, G, 
% and C channels (and the augmented feature ratioGC) are used. The code uses 
% leave-one-patient-out cross-validation (LOOCV) and logs results for later analysis.
%
% Author: [Your Name]
% Date: [Date]

clc;
clearvars -except processedcsfdata;
diary('learning_session_log.txt');
fprintf('Starting multiple experiments at %s\n', datestr(now));

%% --- Set Fixed Parameters ---
% These parameters control data preprocessing and model training.
useDynamicWindow = true;          % Enable dynamic window sizing
min_window_size    = 1;
max_window_size    = 24;
variance_threshold = 0.05;         % Threshold for window variance
windowLabelingMode = 'majority';   % Label windows using majority vote
useAugmented       = true;          % Use augmented features
performHyperparameterTuning = true;
hyperOptMethod     = 'bayesopt';

% Data preparation (using only R, G, C channels and augmented ratioGC)
data = processedcsfdata;
epsilon = 1e-6;
R = data.RNormalized;
G = data.GNormalized;
C = data.CNormalized;
features_orig = [R, G, C];
feature_names = {'R', 'G', 'C'};
ratioGC = G ./ (C + epsilon);
features_augmented = ratioGC;  % single column vector
augmented_names = {'ratioGC'};
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

% Remove rows with NaN or Inf values
valid_rows = ~any(isnan(features_all) | isinf(features_all), 2);
features_all   = features_all(valid_rows, :);
binary_labels  = binary_labels(valid_rows);
patient_ids    = patient_ids(valid_rows);
batches        = batches(valid_rows);

% Data balancing via SMOTE
useSMOTE = true;
if useSMOTE
    k_neighbors = 5;
    [features_all, binary_labels, patient_ids, batches] = ...
        smote_oversample(features_all, binary_labels, patient_ids, batches, k_neighbors);
end
fprintf('\nFinal Balanced Dataset: Class 0 = %d, Class 1 = %d\n', ...
    sum(binary_labels==0), sum(binary_labels==1));

%% --- Define the Parameter Grid for the Experiment ---
metaWeightOptions = [2, 3, 4];            % Candidate values for metaWeightFactor
screeningThresholdOptions = [0.3, 0.4, 0.5];% Candidate screening thresholds

% Preallocate a structure array to log the results
allResults = [];
expCount = 1;

%% --- Loop Over Parameter Configurations ---
for mw = metaWeightOptions
    for st = screeningThresholdOptions
        % Set the parameters for this experiment:
        metaWeightFactor = mw;
        screeningMode = true;       % Use screening mode so that the forced threshold is used
        screeningThreshold = st;    % Lower threshold for higher sensitivity
        
        fprintf('\nExperiment %d: metaWeightFactor = %d, screeningThreshold = %.2f\n', expCount, mw, st);
        
        % Run the stacking ensemble experiment for the current configuration.
        result = runStackingExperiment(metaWeightFactor, screeningMode, screeningThreshold, ...
            features_all, binary_labels, patient_ids, batches, ...
            useDynamicWindow, min_window_size, max_window_size, variance_threshold, windowLabelingMode, ...
            performHyperparameterTuning, hyperOptMethod);
        
        % Append the configuration details to the result structure.
        result.metaWeightFactor = mw;
        result.screeningThreshold = st;
        allResults = [allResults; result];  %#ok<AGROW>
        expCount = expCount + 1;
    end
end

%% --- Export the Experiment Results ---
resultsTable = struct2table(allResults, 'AsArray', true);
writetable(resultsTable, 'experiment_results.csv');
fprintf('\nAll experiment results exported to experiment_results.csv\n');
diary off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Helper Function Definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function result = runStackingExperiment(metaWeightFactor, screeningMode, screeningThreshold, ...
    features_all, binary_labels, patient_ids, batches, ...
    useDynamicWindow, minSize, maxSize, varThreshold, labelingMode, ...
    performHyperparameterTuning, hyperOptMethod)
%RUNSTACKINGEXPERIMENT Runs the stacking ensemble experiment using LOOCV.
%   This function performs leave-one-patient-out cross-validation, aggregates
%   predictions, optimizes the threshold (via Youden's J statistic), and computes
%   performance metrics.
    
    unique_patients = unique(patient_ids);
    n_patients = length(unique_patients);
    all_test_preds  = [];
    all_test_truth  = [];
    all_test_scores = [];
    
    % Preallocate cell arrays for parallel processing.
    test_preds_cell  = cell(n_patients, 1);
    test_truth_cell  = cell(n_patients, 1);
    test_scores_cell = cell(n_patients, 1);
    
    parfor i = 1:n_patients
        curr_patient = unique_patients(i);
        if useDynamicWindow
            [windowed_features, windowed_labels, windowed_patient_ids, ~] = ...
                create_dynamic_rolling_window_per_patient(features_all, binary_labels, patient_ids, batches, ...
                minSize, maxSize, varThreshold, labelingMode);
        else
            % Optionally, add a fallback to fixed windowing if needed.
            error('Fixed windowing is not implemented in this script.');
        end
        
        patient_test_idx = (windowed_patient_ids == curr_patient);
        if sum(patient_test_idx) == 0
            test_preds_cell{i} = [];
            test_truth_cell{i} = [];
            test_scores_cell{i} = [];
            continue;
        end
        
        train_idx = ~patient_test_idx;
        X_train = windowed_features(train_idx, :);
        y_train = windowed_labels(train_idx);
        X_test  = windowed_features(patient_test_idx, :);
        y_test  = windowed_labels(patient_test_idx);
        
        % Standardize features using training statistics.
        [X_train_scaled, mu, sigma] = zscore(X_train);
        X_test_scaled = (X_test - mu) ./ sigma;
        
        % Train the stacking ensemble.
        model = trainStackingClassifier(X_train_scaled, y_train, performHyperparameterTuning, hyperOptMethod, metaWeightFactor);
        [preds, scores] = predictStackingClassifier(model, X_test_scaled);
        
        test_preds_cell{i}  = preds;
        test_truth_cell{i}  = y_test;
        test_scores_cell{i} = scores(:,2);  % probability for infected class.
    end
    
    % Aggregate predictions and ground truth.
    for i = 1:n_patients
        all_test_preds  = [all_test_preds; test_preds_cell{i}];
        all_test_truth  = [all_test_truth; test_truth_cell{i}];
        all_test_scores = [all_test_scores; test_scores_cell{i}];
    end
    
    % Optimize threshold using Youden's J statistic.
    optimized_threshold = optimizeThreshold(all_test_truth, all_test_scores);
    if screeningMode
        used_threshold = screeningThreshold;
    else
        used_threshold = optimized_threshold;
    end
    all_test_preds = double(all_test_scores >= used_threshold);
    
    % Compute performance metrics.
    binary_cm = confusionmat(all_test_truth, all_test_preds);
    [precision, recall, f1] = calculate_metrics(binary_cm);
    accuracy = sum(diag(binary_cm)) / sum(binary_cm(:));
    TN = binary_cm(1,1); FP = binary_cm(1,2);
    FN = binary_cm(2,1); TP = binary_cm(2,2);
    sensitivity = TP / max((TP+FN), 1);
    specificity = TN / max((TN+FP), 1);
    
    % Return the results as a structure.
    result = struct('OptimizedThreshold', optimized_threshold, ...
                    'UsedThreshold', used_threshold, ...
                    'Accuracy', accuracy, ...
                    'F1_Clean', f1(1), ...
                    'F1_Infected', f1(2), ...
                    'Sensitivity', sensitivity, ...
                    'Specificity', specificity, ...
                    'ConfusionMatrix', {binary_cm});
end

function model = trainStackingClassifier(X_train, y_train, performHyperparameterTuning, hyperOptMethod, metaWeightFactor)
%TRAINSTACKINGCLASSIFIER Trains a stacking ensemble with 5-fold CV for meta-features.
%   Base methods include GentleBoost, AdaBoostM1, and Bag.
    
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
    
    % Increase the weight for infected cases in the meta-learner.
    weights = ones(size(y_train));
    weights(y_train == 1) = metaWeightFactor;
    metaModel = fitglm(metaFeatures, y_train, 'Distribution', 'binomial', 'Weights', weights);
    
    model.baseModels = baseModels;
    model.metaModel = metaModel;
end

function [preds, scores] = predictStackingClassifier(model, X_test)
%PREDICTSTACKINGCLASSIFIER Generates predictions using the stacking ensemble.
    nBases = length(model.baseModels);
    metaFeatures = zeros(size(X_test, 1), nBases);
    for m = 1:nBases
        [~, baseScores] = predict(model.baseModels{m}, X_test);
        metaFeatures(:, m) = baseScores(:, 2);
    end
    probs = predict(model.metaModel, metaFeatures);
    scores = [1 - probs, probs];  % [score for class 0, score for class 1]
    preds = double(probs >= 0.5);
end

function model = trainClassifier(X_train, y_train, method, params, performHyperparameterTuning, hyperOptMethod)
%TRAINCLASSIFIER Trains an ensemble classifier based on the specified method.
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

function [precision, recall, f1] = calculate_metrics(cm)
%CALCULATE_METRICS Computes precision, recall, and F1-score for each class.
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
%OPTIMIZETHRESHOLD Finds the threshold that maximizes Youden's J statistic.
    thresholds = 0:0.01:1;
    bestMetric = -Inf;
    bestThreshold = 0.5; % default
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

function visualize_patient_data(features, binary_labels, patient_ids, feature_names)
%VISUALIZE_PATIENT_DATA Plots boxplots, PCA projection, and patient trajectories.
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
                patch([t-0.5, t+0.5, t+0.5, t-0.5], [yl(1) yl(1) yl(2) yl(2)], 'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
            end
        end
        hold off;
        title(sprintf('Patient %d', unique_patients(p)));
        legend(h, feature_names, 'Location', 'eastoutside', 'Interpreter', 'none');
        grid on;
    end
    sgtitle('Patient Trajectories with Infected Regions Highlighted');
end

function [features_new, labels_new, patient_ids_new, batches_new] = smote_oversample(features, labels, patient_ids, batches, k)
%SMOTE_OVERSAMPLE Balances the dataset using SMOTE.
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

function [windowed_features, windowed_labels, windowed_patient_ids, windowed_batches] = ...
    create_dynamic_rolling_window_per_patient(features, labels, patient_ids, batches, minSize, maxSize, varThreshold, labelingMode)
%CREATE_DYNAMIC_ROLLING_WINDOW_PER_PATIENT Creates adaptive-length windows per patient.
%   For each patient, starting at index i, the window is expanded until either
%   the maximum window size is reached or the standard deviation within the window
%   exceeds varThreshold.
%
%   Outputs:
%       windowed_features - matrix where each row is a flattened window of features
%       windowed_labels   - vector of labels (derived by 'union' or 'majority')
%       windowed_patient_ids - vector of patient IDs corresponding to each window
%       windowed_batches  - vector indicating the batch/time of the last entry in each window

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
            
            % Flatten the window into a row vector.
            window = reshape(curr_features(i:window_end, :)', 1, []);
            % Determine the label based on the specified labeling mode.
            switch labelingMode
                case 'union'
                    label = double(any(curr_labels(i:window_end) == 1));
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
