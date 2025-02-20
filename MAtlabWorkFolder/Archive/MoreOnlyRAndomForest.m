%% Expanded RandomForest Experiment with Additional Hyperparameters
% This script performs a parameter sweep for a random forest classifier 
% using TreeBagger with LOOCV. In addition to NTrees, MinLeafSize, and 
% ScreeningThreshold, it now varies:
%   - NumPredictorsToSample (number of predictors sampled at each split)
%   - MaxNumSplits (maximum number of splits per tree)
%
% The experiment terminates early if any configuration achieves an accuracy 
% exceeding 70%.
%
% References:
%   Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
%   Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning.
%
% Author: [Your Name]
% Date: [Date]

clc;
clearvars -except processedcsfdata;
diary('rf_learning_session_log.txt');
fprintf('Starting Expanded Random Forest experiments at %s\n', datestr(now));

%% --- Fixed Parameters ---
useDynamicWindow = true;          % Use dynamic windowing for feature extraction
min_window_size    = 2;             % From old analysis
max_window_size    = 24;
variance_threshold = 0.05;
windowLabelingMode = 'majority';   % As in old analysis
useScreeningThreshold = true;      % Use a fixed screening threshold for evaluation
infection_threshold = 0.4;         % Baseline threshold from old analysis
useSMOTE = true;
k_neighbors = 5;

% Data preparation: use R, G, B, C channels plus augmented ratios.
data = processedcsfdata;
epsilon = 1e-6;
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
augmented_names = {'ratioRC','ratioGC','ratioBC'};
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

% Data balancing via SMOTE.
if useSMOTE
    [features_all, binary_labels, patient_ids, batches] = ...
        smote_oversample(features_all, binary_labels, patient_ids, batches, k_neighbors);
end
fprintf('\nFinal Balanced Dataset: Class 0 = %d, Class 1 = %d\n', ...
    sum(binary_labels==0), sum(binary_labels==1));

%% --- Expanded Parameter Grid ---
% Original grid:
NTreesOptions = [50, 100, 200];           
MinLeafSizeOptions = [1, 5, 10];            
% Expand screening thresholds to include lower values to potentially catch more infections.
ScreeningThresholdOptions = [0.2, 0.3, 0.4, 0.5];  

% New hyperparameters:
p = size(features_all, 2);  % total number of predictors
NumPredictorsOptions = [max(1, round(sqrt(p))), max(1, round(p/2)), p];
MaxNumSplitsOptions = [10, 20, 50];  % maximum number of splits per tree

allResults = [];
expCount = 1;
exitFlag = false;  % Flag to indicate early stopping when >70% accuracy is reached

%% --- Experiment Loop ---
for nt = NTreesOptions
    if exitFlag, break; end
    for ml = MinLeafSizeOptions
        if exitFlag, break; end
        for st = ScreeningThresholdOptions
            if exitFlag, break; end
            for np = NumPredictorsOptions
                if exitFlag, break; end
                for ms = MaxNumSplitsOptions
                    fprintf('\nExperiment %d: NTrees = %d, MinLeafSize = %d, ScreeningThreshold = %.2f, NumPredictors = %d, MaxNumSplits = %d\n', ...
                        expCount, nt, ml, st, np, ms);
                    result = runRFExperiment(nt, ml, st, useScreeningThreshold, infection_threshold, ...
                        features_all, binary_labels, patient_ids, batches, ...
                        useDynamicWindow, min_window_size, max_window_size, variance_threshold, windowLabelingMode, np, ms);
                    % Append parameter details to the result.
                    result.NTrees = nt;
                    result.MinLeafSize = ml;
                    result.ScreeningThreshold = st;
                    result.NumPredictors = np;
                    result.MaxNumSplits = ms;
                    allResults = [allResults; result]; %#ok<AGROW>
                    expCount = expCount + 1;
                    
                    % Check if accuracy exceeds 70%
                    if result.Accuracy > 0.70
                        fprintf('Accuracy %.2f%% exceeds 70%% threshold. Terminating further experiments.\n', result.Accuracy*100);
                        exitFlag = true;
                        break;
                    end
                end
            end
        end
    end
end

%% --- Export Results ---
resultsTable = struct2table(allResults, 'AsArray', true);
writetable(resultsTable, 'rf_expanded_experiment_results.csv');
fprintf('\nAll experiment results exported to rf_expanded_experiment_results.csv\n');
diary off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Helper Function Definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function result = runRFExperiment(NTrees, MinLeafSize, screeningThreshold, useScreeningThreshold, infection_threshold, ...
    features_all, binary_labels, patient_ids, batches, useDynamicWindow, minSize, maxSize, varThreshold, labelingMode, NPred, maxSplits)
    unique_patients = unique(patient_ids);
    n_patients = length(unique_patients);
    all_test_preds  = [];
    all_test_truth  = [];
    all_test_scores = [];
    
    % Preallocate cell arrays for parallel computation.
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
        
        % Standardize features using training set statistics.
        [X_train_scaled, mu, sigma] = zscore(X_train);
        X_test_scaled = (X_test - mu) ./ sigma;
        
        % Train the random forest model using expanded hyperparameters.
        rfModel = trainRandomForestClassifier(X_train_scaled, y_train, NTrees, MinLeafSize, NPred, maxSplits);
        
        % Generate predictions; pass infection_threshold to the prediction function.
        [preds, scores] = predictRandomForestClassifier(rfModel, X_test_scaled, infection_threshold);
        
        test_preds_cell{i} = preds;
        test_truth_cell{i} = y_test;
        test_scores_cell{i} = scores;
    end
    
    % Aggregate predictions and ground truth.
    for i = 1:n_patients
        all_test_preds = [all_test_preds; test_preds_cell{i}];
        all_test_truth = [all_test_truth; test_truth_cell{i}];
        all_test_scores = [all_test_scores; test_scores_cell{i}];
    end
    
    % Determine threshold.
    optimized_threshold = optimizeThreshold(all_test_truth, all_test_scores);
    if useScreeningThreshold
        used_threshold = screeningThreshold;
    else
        used_threshold = optimized_threshold;
    end
    % Recompute binary predictions using the chosen threshold.
    all_test_preds = double(all_test_scores >= used_threshold);
    
    % Compute confusion matrix and performance metrics.
    binary_cm = confusionmat(all_test_truth, all_test_preds);
    [precision, recall, f1] = calculate_metrics(binary_cm);
    accuracy = sum(diag(binary_cm)) / sum(binary_cm(:));
    TN = binary_cm(1,1); FP = binary_cm(1,2);
    FN = binary_cm(2,1); TP = binary_cm(2,2);
    sensitivity = TP / max((TP+FN), 1);
    specificity = TN / max((TN+FP), 1);
    
    result = struct('OptimizedThreshold', optimized_threshold, ...
                    'UsedThreshold', used_threshold, ...
                    'Accuracy', accuracy, ...
                    'F1_Clean', f1(1), ...
                    'F1_Infected', f1(2), ...
                    'Sensitivity', sensitivity, ...
                    'Specificity', specificity, ...
                    'ConfusionMatrix', {binary_cm});
end

function rfModel = trainRandomForestClassifier(X_train, y_train, NTrees, MinLeafSize, NPred, maxSplits)
    y_train_cat = categorical(y_train);
    opts = statset('UseParallel', true);
    % Pass hyperparameters as name-value pairs.
    rfModel = TreeBagger(NTrees, X_train, y_train_cat, 'Method', 'classification', ...
                'MinLeafSize', MinLeafSize, 'NumVariablesToSample', NPred, 'MaxNumSplits', maxSplits, ...
                'OOBPrediction', 'off', 'Options', opts);
end

function [preds, scores] = predictRandomForestClassifier(rfModel, X_test, infection_threshold)
    [~, scoreOut] = predict(rfModel, X_test);
    if iscell(scoreOut)
        % Convert cell array to numeric matrix.
        scoresMat = str2double(scoreOut);
    else
        scoresMat = scoreOut;
    end
    % If only one column is returned, assume it corresponds to the positive class.
    if size(scoresMat,2) == 1
        scores = scoresMat(:,1);
    else
        % Identify the column corresponding to the positive class (assumed label '1').
        classNames = rfModel.ClassNames;
        idxPositive = find(classNames == categorical(1));
        if isempty(idxPositive)
            error('Positive class not found in the model.');
        end
        scores = scoresMat(:, idxPositive);
    end
    % Determine binary predictions using the provided infection_threshold.
    preds = double(scores >= infection_threshold);
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
    bestThreshold = 0.5; % default threshold
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
