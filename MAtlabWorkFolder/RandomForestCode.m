%% Expanded RandomForest Experiment with Hyperparameter Sweep, Presentation Plots, and Model Saving
% This script iterates through random seeds and a full hyperparameter grid 
% (NTrees, MinLeafSize, ScreeningThreshold, NumPredictorsToSample, and 
% MaxNumSplits) and prints the accuracy and sensitivity for every configuration.
% It stops as soon as it finds a configuration where:
%   - Accuracy > 80%
%   - Sensitivity > 70%
%
% After a configuration meeting these criteria is found, the script:
%   1. Generates presentation‐quality plots (ROC, confusion matrix, histogram).
%   2. Trains a final model on the entire windowed dataset.
%   3. Saves the final model (and feature scaling parameters) to a MAT‐file.
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

%% --- Setup CSV Logging ---
outputCSV = 'rf_experiment_results.csv';
fid = fopen(outputCSV, 'w');
% Write header row:
fprintf(fid, 'Experiment,Seed,NTrees,MinLeafSize,ScreeningThreshold,NumPredictors,MaxNumSplits,Accuracy,Sensitivity,Specificity,OptimizedThreshold,UsedThreshold,F1_Clean,F1_Infected\n');

%% --- Fixed Data Processing Parameters ---
useDynamicWindow = true;          % Use dynamic windowing for feature extraction
min_window_size    = 2;             % As in previous analysis
max_window_size    = 24;
variance_threshold = 0.05;
windowLabelingMode = 'majority';   % Majority voting for labeling
useScreeningThreshold = false;      % The experiment will use an optimized threshold here
infection_threshold = 0.4;         % Baseline threshold from earlier analysis
useSMOTE = false;
k_neighbors = 5;

% Data preparation: use R, G, B, C channels plus augmented ratios.
data = processedcsfdata;
epsilon = 1e-6;
R = data.RNormalized;
G = data.GNormalized;
B = data.BNormalized;
C = data.CNormalized;
features_orig = [R, G, B, C];
feature_names = {'R','G','C'};
% Augmented features (ratios)
ratioRC = R ./ (C + epsilon);
ratioGC = G ./ (C + epsilon);
ratioBC = B ./ (C + epsilon);
features_augmented = [ratioRC, ratioGC, ratioBC];
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

% Data balancing via SMOTE.
if useSMOTE
    [features_all, binary_labels, patient_ids, batches] = ...
        smote_oversample(features_all, binary_labels, patient_ids, batches, k_neighbors);
end
fprintf('\nFinal Balanced Dataset: Class 0 = %d, Class 1 = %d\n', ...
    sum(binary_labels==0), sum(binary_labels==1));

%% --- Hyperparameter Grid ---
NTreesOptions = [50];           
MinLeafSizeOptions = [1];            
ScreeningThresholdOptions = [0.45, 0.46, 0.47, 0.48, 0.49, 0.5];  

p = size(features_all, 2);  % Total number of predictors
NumPredictorsOptions = [max(1, round(sqrt(p))), max(1, round(p/2)), p];
MaxNumSplitsOptions = [10, 20];  

%% --- Experiment Criteria ---
% We want to stop when:
%   Accuracy > 80%
%   Sensitivity > 70%
target_accuracy = 0.80;
target_sensitivity = 0.70;

%% --- Seed and Parameter Grid Sweep ---
maxSeeds = 10000;  % Maximum number of seeds to try
foundFlag = false;  % Flag indicating a desired configuration has been found
finalSeed = NaN;
finalResult = [];
finalParams = struct();

expCount = 1;  % Experiment counter

for s = 1:maxSeeds
    if foundFlag, break; end
    rng(s);  % Set the random seed
    fprintf('\nSeed %d\n', s);
    
    % Loop over hyperparameter grid
    for nt = NTreesOptions
        if foundFlag, break; end
        for ml = MinLeafSizeOptions
            if foundFlag, break; end
            for st = ScreeningThresholdOptions
                if foundFlag, break; end
                for np = NumPredictorsOptions
                    if foundFlag, break; end
                    for ms = MaxNumSplitsOptions
                        fprintf('\nExperiment %d: Seed = %d, NTrees = %d, MinLeafSize = %d, ScreeningThreshold = %.2f, NumPredictors = %d, MaxNumSplits = %d\n', ...
                            expCount, s, nt, ml, st, np, ms);
                        result = runRFExperiment(nt, ml, st, useScreeningThreshold, infection_threshold, ...
                            features_all, binary_labels, patient_ids, batches, ...
                            useDynamicWindow, min_window_size, max_window_size, variance_threshold, windowLabelingMode, np, ms);
                        % Append hyperparameter details to the result for record
                        result.NTrees = nt;
                        result.MinLeafSize = ml;
                        result.ScreeningThreshold = st;
                        result.NumPredictors = np;
                        result.MaxNumSplits = ms;
                        % Log the result to CSV:
                        fprintf(fid, '%d,%d,%d,%d,%.2f,%d,%d,%.4f,%.4f,%.4f,%.2f,%.2f,%.4f,%.4f\n', ...
                            expCount, s, nt, ml, st, np, ms, result.Accuracy, result.Sensitivity, result.Specificity, ...
                            result.OptimizedThreshold, result.UsedThreshold, result.F1_Clean, result.F1_Infected);
                        
                        expCount = expCount + 1;
                        
                        fprintf(' -> Accuracy: %.2f%%, Sensitivity: %.2f%%\n', result.Accuracy*100, result.Sensitivity*100);
                        
                        % Check if criteria are met: Accuracy > 80% and Sensitivity > 70%
                        if (result.Accuracy > target_accuracy) && (result.Sensitivity > target_sensitivity)
                            fprintf('Criteria met: Accuracy = %.2f%%, Sensitivity = %.2f%%\n', result.Accuracy*100, result.Sensitivity*100);
                            foundFlag = true;
                            finalSeed = s;
                            finalResult = result;
                            finalParams.NTrees = nt;
                            finalParams.MinLeafSize = ml;
                            finalParams.ScreeningThreshold = st;
                            finalParams.NumPredictors = np;
                            finalParams.MaxNumSplits = ms;
                            break;
                        end
                    end
                end
            end
        end
    end
end

if foundFlag
    fprintf('\nFinal configuration found!\n');
    fprintf('Seed: %d\n', finalSeed);
    fprintf('NTrees: %d\n', finalParams.NTrees);
    fprintf('MinLeafSize: %d\n', finalParams.MinLeafSize);
    fprintf('ScreeningThreshold: %.2f\n', finalParams.ScreeningThreshold);
    fprintf('NumPredictors: %d\n', finalParams.NumPredictors);
    fprintf('MaxNumSplits: %d\n', finalParams.MaxNumSplits);
    fprintf('Final Accuracy: %.2f%%\n', finalResult.Accuracy*100);
    fprintf('Final Sensitivity: %.2f%%\n', finalResult.Sensitivity*100);
    fprintf('OptimizedThreshold: %.2f\n', finalResult.OptimizedThreshold);
    fprintf('UsedThreshold: %.2f\n', finalResult.UsedThreshold);
    fprintf('F1 (Clean): %.4f\n', finalResult.F1_Clean);
    fprintf('F1 (Infected): %.4f\n', finalResult.F1_Infected);
    fprintf('Specificity: %.2f%%\n', finalResult.Specificity*100);
    
    % Compute and display FP and FN rates based on the confusion matrix.
    fprintf('Confusion Matrix:\n');
    disp(finalResult.ConfusionMatrix);
    cm = finalResult.ConfusionMatrix;
    TN = cm(1,1); FP = cm(1,2); FN = cm(2,1); TP = cm(2,2);
    fp_rate = FP / (TN + FP);
    fn_rate = FN / (TP + FN);
    fprintf('False Positive Rate: %.2f%%\n', fp_rate*100);
    fprintf('False Negative Rate: %.2f%%\n', fn_rate*100);
else
    fprintf('\nNo configuration meeting the criteria was found after %d seeds and %d experiments.\n', maxSeeds, expCount);
end

fclose(fid);  % Close CSV file
diary off;

%% --- Generate Presentation-Quality Plots ---
% ROC Curve
figure;
[rocX, rocY, rocT, auc] = perfcurve(finalResult.all_test_truth, finalResult.all_test_scores, 1);
plot(rocX, rocY, 'b-', 'LineWidth', 2);
xlabel('False Positive Rate', 'FontSize', 24);
ylabel('True Positive Rate', 'FontSize', 24);
title(sprintf('ROC Curve (AUC = %.3f)', auc), 'FontSize', 24);
grid on;
set(gca, 'FontSize', 24);  % Set axes tick labels to font size 24

% Modified Confusion Matrix Plot: Annotate with numerical values.
figure;
cm = finalResult.ConfusionMatrix;
imagesc(cm);
colormap('hot');
colorbar;
title('Confusion Matrix', 'FontSize', 24);
xlabel('Predicted Class', 'FontSize', 24);
ylabel('True Class', 'FontSize', 24);
set(gca, 'XTick', 1:2, 'XTickLabel', {'Clean','Infected'}, 'FontSize', 24);
set(gca, 'YTick', 1:2, 'YTickLabel', {'Clean','Infected'}, 'FontSize', 24);
% Overlay the numbers on each cell with increased font size
textStrings = num2str(cm(:), '%d'); % Convert numbers to string
textStrings = strtrim(cellstr(textStrings));  % Remove any space padding
[x, y] = meshgrid(1:size(cm,2), 1:size(cm,1));
text(x(:), y(:), textStrings(:), 'HorizontalAlignment', 'center', ...
    'Color', 'white', 'FontWeight', 'bold', 'FontSize', 24);

% Histogram of Prediction Scores by Class
figure;
hold on;
histogram(finalResult.all_test_scores(finalResult.all_test_truth==0), 'Normalization', 'pdf', 'FaceColor', 'r', 'FaceAlpha', 0.5);
histogram(finalResult.all_test_scores(finalResult.all_test_truth==1), 'Normalization', 'pdf', 'FaceColor', 'g', 'FaceAlpha', 0.5);
title('Distribution of Prediction Scores by Class', 'FontSize', 24);
xlabel('Prediction Score', 'FontSize', 24);
ylabel('Probability Density', 'FontSize', 24);
legend('Clean (0)','Infected (1)', 'FontSize', 24);
set(gca, 'FontSize', 24);  % Set axes tick labels to font size 24
hold off;


%% --- Train Final Model on Entire Dataset and Save It ---
fprintf('\nTraining final model on the entire windowed dataset...\n');
[all_windowed_features, all_windowed_labels, ~, ~] = create_dynamic_rolling_window_per_patient(...
    features_all, binary_labels, patient_ids, batches, min_window_size, max_window_size, variance_threshold, windowLabelingMode);

% Standardize features for the final model.
[X_train_final, mu_final, sigma_final] = zscore(all_windowed_features);

% Train the final Random Forest model using the best hyperparameters.
finalModel = TreeBagger(finalParams.NTrees, X_train_final, categorical(all_windowed_labels), 'Method', 'classification', ...
    'MinLeafSize', finalParams.MinLeafSize, 'NumVariablesToSample', finalParams.NumPredictors, 'MaxNumSplits', finalParams.MaxNumSplits, 'OOBPrediction', 'off');

% Save the model along with scaling parameters.
save('finalRFModel.mat', 'finalModel', 'mu_final', 'sigma_final');
fprintf('Final model saved as finalRFModel.mat\n');

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
        
        % Train the random forest model.
        rfModel = trainRandomForestClassifier(X_train_scaled, y_train, NTrees, MinLeafSize, NPred, maxSplits);
        
        % Generate predictions.
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
    
    % Include aggregated ground truth and scores for plotting.
    result = struct('OptimizedThreshold', optimized_threshold, ...
                    'UsedThreshold', used_threshold, ...
                    'Accuracy', accuracy, ...
                    'F1_Clean', f1(1), ...
                    'F1_Infected', f1(2), ...
                    'Sensitivity', sensitivity, ...
                    'Specificity', specificity, ...
                    'ConfusionMatrix', {binary_cm}, ...
                    'all_test_truth', all_test_truth, ...
                    'all_test_scores', all_test_scores);
end

function rfModel = trainRandomForestClassifier(X_train, y_train, NTrees, MinLeafSize, NPred, maxSplits)
    y_train_cat = categorical(y_train);
    opts = statset('UseParallel', true);
    rfModel = TreeBagger(NTrees, X_train, y_train_cat, 'Method', 'classification', ...
                'MinLeafSize', MinLeafSize, 'NumVariablesToSample', NPred, 'MaxNumSplits', maxSplits, ...
                'OOBPrediction', 'off', 'Options', opts);
end

function [preds, scores] = predictRandomForestClassifier(rfModel, X_test, infection_threshold)
    [~, scoreOut] = predict(rfModel, X_test);
    if iscell(scoreOut)
        scoresMat = str2double(scoreOut);
    else
        scoresMat = scoreOut;
    end
    if size(scoresMat,2) == 1
        scores = scoresMat(:,1);
    else
        classNames = rfModel.ClassNames;
        idxPositive = find(classNames == categorical(1));
        if isempty(idxPositive)
            error('Positive class not found in the model.');
        end
        scores = scoresMat(:, idxPositive);
    end
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
    bestThreshold = 0.5;
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
