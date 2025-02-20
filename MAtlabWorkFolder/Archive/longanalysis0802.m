%% Ensemble Model Comparison with Multiple Experiments
% This script implements LOOCV with rolling windows, oversampling (SMOTE),
% and evaluation of different models and hyperparameter configurations.
% Models include GentleBoost (with different parameters), AdaBoostM1, and
% RandomForest (via bagging). Aggregated performance metrics are exported
% for further analysis.
%
% Author: [Your Name]
% Date: [Date]

clc;
clearvars -except processedcsfdata

% Start logging (diary)
diary('training_log.txt');
fprintf('Starting experiments at %s\n', datestr(now));

%% Check for processed data
if ~exist('processedcsfdata', 'var')
    error('processedcsfdata is not found in the workspace.');
end

%% User-Defined Parameters
performHyperparameterTuning = true;  % Set to false to disable tuning
useSMOTE = true;                     % Enable SMOTE oversampling
infection_threshold = 0.4;             % Threshold for classifying infection
window_sizes = [1, 2, 4, 6, 8, 12, 24];  % Rolling window sizes

%% Define Experiments (Models & Parameter Configurations)
% For GentleBoost, we test three parameter configurations.
gentleBoostParams = { ...
    struct('Name','GentleBoost_50cycles_0.1', 'NumLearningCycles',50, 'LearnRate',0.1), ...
    struct('Name','GentleBoost_100cycles_0.1','NumLearningCycles',100,'LearnRate',0.1), ...
    struct('Name','GentleBoost_50cycles_0.05','NumLearningCycles',50, 'LearnRate',0.05) ...
};

% For AdaBoostM1, we use a default configuration; hyperparameter tuning
% is applied if enabled.
adaBoostParams = { ...
    struct('Name','AdaBoost_default','Optimize',performHyperparameterTuning) ...
};

% For RandomForest (implemented as bagging) we also use a default configuration.
randomForestParams = { ...
    struct('Name','RandomForest_default','Optimize',performHyperparameterTuning) ...
};

% Combine experiments into a single cell array.
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

%% Data Preparation and Feature Selection
data = processedcsfdata;
batches = data.Batch;

% Original features (with a small epsilon for ratio computations)
epsilon = 1e-6;
R = data.RNormalized;
G = data.GNormalized;
B = data.BNormalized;
C = data.CNormalized;
features_orig = [R, G, B, C];
feature_names = {'R','G','B','C'};

% Augmented features (example: ratios)
ratioRC = R ./ (C + epsilon);
ratioGC = G ./ (C + epsilon);
features_augmented = [ratioRC, ratioGC];
augmented_names = {'ratioRC','ratioGC'};

% Option to use augmented features
useAugmented = false;  % Set to true to include augmented features
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
    k_neighbors = 5;  % Number of neighbors for SMOTE
    [features_all, binary_labels, patient_ids, batches] = ...
        smote_oversample(features_all, binary_labels, patient_ids, batches, k_neighbors);
else
    % Fallback: simple random oversampling
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
    
    % Loop over rolling window sizes
    for w = 1:length(window_sizes)
        window_size = window_sizes(w);
        fprintf('\nLOOCV with Window Size: %d\n', window_size);
        
        % Create rolling windows per patient (only pure windows flagged as pure)
        [windowed_features, windowed_labels, windowed_patient_ids, windowed_batches, windowed_is_pure] = ...
            create_rolling_window_per_patient(features_all, binary_labels, patient_ids, batches, window_size);
        
        all_test_preds = [];
        all_test_truth = [];
        unique_patients = unique(windowed_patient_ids);
        n_patients = length(unique_patients);
        
        % Preallocate cell arrays to collect predictions in parallel
        test_preds_cell = cell(n_patients,1);
        test_truth_cell = cell(n_patients,1);
        
        % Parallel leave-one-patient-out cross-validation
        parfor i = 1:n_patients
            curr_patient = unique_patients(i);
            test_idx = (windowed_patient_ids == curr_patient);
            pure_test_idx = test_idx & windowed_is_pure;
            if sum(pure_test_idx) == 0
                % No pure windows available for this patient
                test_preds_cell{i} = [];
                test_truth_cell{i} = [];
                continue;
            end
            train_idx = ~test_idx;
            X_train = windowed_features(train_idx, :);
            y_train = windowed_labels(train_idx);
            X_test = windowed_features(pure_test_idx, :);
            y_test = windowed_labels(pure_test_idx);
            
            % Standardize features using training set statistics
            [X_train_scaled, mu, sigma] = zscore(X_train);
            X_test_scaled = (X_test - mu) ./ sigma;
            
            % Train classifier using the modular trainClassifier function
            model = trainClassifier(X_train_scaled, y_train, method, params, performHyperparameterTuning);
            [~, scores] = predict(model, X_test_scaled);
            preds_adjusted = double(scores(:,2) >= infection_threshold);
            
            test_preds_cell{i} = preds_adjusted;
            test_truth_cell{i} = y_test;
        end
        
        % Aggregate predictions and ground truths
        for i = 1:n_patients
            all_test_preds = [all_test_preds; test_preds_cell{i}];
            all_test_truth = [all_test_truth; test_truth_cell{i}];
        end
        
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
        
        % Store the results for the current experiment and window size
        allResults = [allResults; struct(...
            'Experiment', expName, ...
            'Method', method, ...
            'WindowSize', window_size, ...
            'Accuracy', accuracy, ...
            'F1_Clean', f1(1), ...
            'F1_Infected', f1(2), ...
            'ConfusionMatrix', {binary_cm}, ...
            'Sensitivity', sensitivity, ...
            'Specificity', specificity, ...
            'FP_rate', FP_rate, ...
            'FN_rate', FN_rate)];
        
        fprintf('Exp: %s, Method: %s, Window Size: %d | Accuracy: %.2f | F1 (Clean): %.4f | F1 (Infected): %.4f | Sensitivity: %.4f | Specificity: %.4f | FP Rate: %.4f | FN Rate: %.4f\n', ...
            expName, method, window_size, accuracy, f1(1), f1(2), sensitivity, specificity, FP_rate, FN_rate);
    end
end

%% Export Aggregated Performance Metrics
metricsTable = struct2table(allResults);
writetable(metricsTable, 'ensemble_performance_metrics.csv');
fprintf('\nPerformance metrics exported to ensemble_performance_metrics.csv\n');

%% Diagnostic Analysis: Variable Importance for a Chosen Window Size (e.g., 24)
chosen_window_size = 24;
analyze_model_behavior(features_all, binary_labels, patient_ids, batches, chosen_window_size, all_feature_names);

%% Patient Data Visualization
visualize_patient_data(features_all, binary_labels, patient_ids, all_feature_names);

diary off;

%% Helper Functions

function model = trainClassifier(X_train, y_train, method, params, performHyperparameterTuning)
    % TRAINCLASSIFIER trains an ensemble classifier given a method and parameter configuration.
    %
    %   Inputs:
    %       X_train - Standardized training features.
    %       y_train - Training labels.
    %       method  - A string indicating the method ('GentleBoost', 'AdaBoostM1', 'Bag').
    %       params  - A struct containing hyperparameter settings.
    %       performHyperparameterTuning - Boolean flag for tuning.
    %
    %   Output:
    %       model - Trained classification ensemble.
    
    switch method
        case 'GentleBoost'
            if performHyperparameterTuning
                model = fitcensemble(X_train, y_train, 'Method', 'GentleBoost', ...
                    'NumLearningCycles', params.NumLearningCycles, ...
                    'LearnRate', params.LearnRate, ...
                    'OptimizeHyperparameters', 'all', ...
                    'HyperparameterOptimizationOptions', struct('ShowPlots', false, 'Verbose', 0));
            else
                model = fitcensemble(X_train, y_train, 'Method', 'GentleBoost', ...
                    'NumLearningCycles', params.NumLearningCycles, ...
                    'LearnRate', params.LearnRate);
            end
        case 'AdaBoostM1'
            if performHyperparameterTuning && isfield(params, 'Optimize') && params.Optimize
                model = fitcensemble(X_train, y_train, 'Method', 'AdaBoostM1', ...
                    'OptimizeHyperparameters', 'all', ...
                    'HyperparameterOptimizationOptions', struct('ShowPlots', false, 'Verbose', 0));
            else
                model = fitcensemble(X_train, y_train, 'Method', 'AdaBoostM1');
            end
        case 'Bag'
            if performHyperparameterTuning && isfield(params, 'Optimize') && params.Optimize
                model = fitcensemble(X_train, y_train, 'Method', 'Bag', ...
                    'OptimizeHyperparameters', 'all', ...
                    'HyperparameterOptimizationOptions', struct('ShowPlots', false, 'Verbose', 0));
            else
                model = fitcensemble(X_train, y_train, 'Method', 'Bag');
            end
        otherwise
            error('Unsupported model method: %s', method);
    end
end

function [precision, recall, f1] = calculate_metrics(cm)
    % CALCULATE_METRICS computes precision, recall, and F1-score from a confusion matrix.
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

function windowed_data = create_rolling_window(data, window_size)
    % CREATE_ROLLING_WINDOW creates a rolling window representation of the data.
    [rows, cols] = size(data);
    windowed_data = zeros(rows - window_size + 1, cols * window_size);
    for i = 1:(rows - window_size + 1)
        windowed_data(i, :) = reshape(data(i:i+window_size-1, :)', 1, []);
    end
end

function [windowed_features, windowed_labels, windowed_patient_ids, windowed_batches, windowed_is_pure] = ...
    create_rolling_window_per_patient(features, binary_labels, patient_ids, batches, window_size)
    % CREATE_ROLLING_WINDOW_PER_PATIENT creates rolling windows for each patient.
    % Each window is labeled infected (1) if any block in the window is infected.
    % A purity flag is also computed.
    windowed_features = [];
    windowed_labels = [];
    windowed_patient_ids = [];
    windowed_batches = [];
    windowed_is_pure = [];
    unique_patients = unique(patient_ids);
    
    for p = 1:length(unique_patients)
        curr_patient = unique_patients(p);
        idx = find(patient_ids == curr_patient);
        [sorted_batches, sortOrder] = sort(batches(idx));
        idx_sorted = idx(sortOrder);
        
        curr_features = features(idx_sorted, :);
        curr_labels = binary_labels(idx_sorted);
        curr_batches = batches(idx_sorted);
        n_rows = size(curr_features, 1);
        
        if n_rows < window_size
            continue;
        end
        
        for j = 1:(n_rows - window_size + 1)
            window = reshape(curr_features(j:j+window_size-1, :)', 1, []);
            % Union rule: label as infected if any block is infected
            if any(curr_labels(j:j+window_size-1) == 1)
                label = 1;
            else
                label = 0;
            end
            % Purity flag: all labels in window are the same
            is_pure = all(curr_labels(j:j+window_size-1) == curr_labels(j));
            
            batch_val = curr_batches(j+window_size-1);
            windowed_features = [windowed_features; window];
            windowed_labels = [windowed_labels; label];
            windowed_patient_ids = [windowed_patient_ids; curr_patient];
            windowed_batches = [windowed_batches; batch_val];
            windowed_is_pure = [windowed_is_pure; is_pure];
        end
    end
end

function analyze_model_behavior(features, binary_labels, patient_ids, batches, window_size, feature_names)
    % ANALYZE_MODEL_BEHAVIOR performs variable importance analysis using a bagged ensemble.
    [windowed_features, windowed_labels, ~, ~, ~] = ...
        create_rolling_window_per_patient(features, binary_labels, patient_ids, batches, window_size);
    
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
        heatmap(feature_names, cellstr("T-" + string(0:(size(imp_reshaped,2)-1))), imp_reshaped');
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
    
    % 1) Boxplots of Feature Distributions by Infection Status
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
    
    % 2) PCA Projection and Feature Correlation
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
    
    % 3) Patient Trajectory Visualization
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
