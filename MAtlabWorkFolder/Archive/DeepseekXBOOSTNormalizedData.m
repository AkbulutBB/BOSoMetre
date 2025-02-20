%% Main Script
clc;
clearvars -except cleaneddata

if ~exist('cleaneddata', 'var')
    error('cleaneddata is not found in the workspace.');
end

data = cleaneddata;

% Use the "batch" column for ordering
batches = data.Batch;

% Extract features and create ratios
epsilon = 1e-6;  % Prevent division by zero
R = data.RPercNormalized;
G = data.GPercNormalized;
B = data.BPercNormalized;
C = data.CPercNormalized;
RC = R ./ (C + epsilon);
GC = G ./ (C + epsilon);
BC = B ./ (C + epsilon);

features = [R, G, B, C, RC, GC, BC];

% Extract binary labels and patient IDs
binary_labels = data.infClassIDSA;
patient_ids = data.InStudyID;

% Remove rows with NaN/Inf
valid_rows = ~any(isnan(features) | isinf(features), 2);
features = features(valid_rows, :);
binary_labels = binary_labels(valid_rows);
patient_ids = patient_ids(valid_rows);
batches = batches(valid_rows);

% Handle class imbalance
fprintf('\nBalancing the Dataset...\n');
class0_idx = find(binary_labels == 0);
class1_idx = find(binary_labels == 1);
max_samples = max(length(class0_idx), length(class1_idx));
rng(42);
class0_oversampled = class0_idx(randi(length(class0_idx), max_samples, 1));
class1_oversampled = class1_idx(randi(length(class1_idx), max_samples, 1));
balanced_idx = [class0_oversampled; class1_oversampled];
features = features(balanced_idx, :);
binary_labels = binary_labels(balanced_idx);
patient_ids = patient_ids(balanced_idx);
batches = batches(balanced_idx);
fprintf('Final Balanced Dataset: Class 0 = %d, Class 1 = %d\n', sum(binary_labels==0), sum(binary_labels==1));

% Leave-One-Patient-Out Cross-Validation with multiple window sizes
window_sizes = [6, 8, 12, 24];
results = struct();
unique_patients = unique(patient_ids);
n_patients = length(unique_patients);

for w = 1:length(window_sizes)
    window_size = window_sizes(w);
    fprintf('\nPerforming LOOCV - Window Size: %d\n', window_size);
    
    % Create rolling windows and also output a purity flag for each window
    [windowed_features, windowed_labels, windowed_patient_ids, ~, windowed_is_pure] = ...
        create_rolling_window_per_patient(features, binary_labels, patient_ids, batches, window_size);
    
    % Accumulate test predictions and ground truth from pure windows only
    all_test_preds = [];
    all_test_truth = [];
    
    for i = 1:n_patients
        fprintf('Processing patient %d/%d\n', i, n_patients);
        test_idx = (windowed_patient_ids == unique_patients(i));
        % Keep only pure test windows (i.e. windows where all blocks have the same original label)
        pure_test_idx = test_idx & windowed_is_pure;
        
        if sum(pure_test_idx) == 0
            fprintf('Patient %d has no pure windows. Skipping this patient in testing.\n', unique_patients(i));
            continue;
        end
        
        % Use all windows from other patients for training (including impure ones)
        train_idx = ~test_idx;
        X_train = windowed_features(train_idx, :);
        y_train = windowed_labels(train_idx);
        X_test = windowed_features(pure_test_idx, :);
        y_test = windowed_labels(pure_test_idx);
        
        % Standardize training and test sets using training statistics
        [X_train_scaled, mu, sigma] = zscore(X_train);
        X_test_scaled = (X_test - mu) ./ sigma;
        
        % Train a bagged ensemble classifier
        % (Consider tuning hyperparameters or trying alternative models to improve accuracy.)
        model = fitcensemble(X_train_scaled, y_train, 'Method', 'Bag');
        preds = predict(model, X_test_scaled);
        
        % Accumulate predictions and corresponding ground truth
        all_test_preds = [all_test_preds; preds];
        all_test_truth = [all_test_truth; y_test];
    end
    
    % Evaluate performance metrics on the accumulated test set
    binary_cm = confusionmat(all_test_truth, all_test_preds);
    [precision, recall, f1] = calculate_metrics(binary_cm);
    accuracy = sum(diag(binary_cm)) / sum(binary_cm(:));
    
    % Compute additional metrics:
    % Assuming the confusion matrix order is: 
    %    row 1 = Actual Clean, row 2 = Actual Infected,
    %    col 1 = Predicted Clean, col 2 = Predicted Infected.
    TN = binary_cm(1,1);
    FP = binary_cm(1,2);
    FN = binary_cm(2,1);
    TP = binary_cm(2,2);
    
    sensitivity = TP / max((TP + FN), 1);   % Recall for infected (true positive rate)
    specificity = TN / max((TN + FP), 1);     % True negative rate for clean
    FP_rate = FP / max((TN + FP), 1);
    FN_rate = FN / max((TP + FN), 1);
    
    results(w).window_size = window_size;
    results(w).accuracy = accuracy;
    results(w).f1 = f1;
    results(w).confusion_matrix = binary_cm;
    results(w).sensitivity = sensitivity;
    results(w).specificity = specificity;
    results(w).FP_rate = FP_rate;
    results(w).FN_rate = FN_rate;
    
    % Print performance metrics in the terminal for easy copy-pasting
    fprintf('Window Size: %d | F1 (Clean): %.4f | F1 (Infected): %.4f | Sensitivity: %.4f | Specificity: %.4f | FP Rate: %.4f | FN Rate: %.4f\n', ...
            window_size, f1(1), f1(2), sensitivity, specificity, FP_rate, FN_rate);
    
    figure;
    cm_chart = confusionchart(binary_cm, {'Clean', 'Infected'});
    cm_chart.Title = sprintf('Confusion Matrix (Window Size: %d)', window_size);
end

% Performance trends across window sizes
fprintf('\nPerformance Comparison:\n');
accuracy_values = arrayfun(@(x) x.accuracy, results);
f1_class0 = arrayfun(@(x) x.f1(1), results);
f1_class1 = arrayfun(@(x) x.f1(2), results);
sensitivities = arrayfun(@(x) x.sensitivity, results);
specificities = arrayfun(@(x) x.specificity, results);
FP_rates = arrayfun(@(x) x.FP_rate, results);
FN_rates = arrayfun(@(x) x.FN_rate, results);

figure;
plot(window_sizes, accuracy_values, '-o', 'LineWidth', 2);
hold on;
plot(window_sizes, f1_class0, '-s', 'LineWidth', 2);
plot(window_sizes, f1_class1, '-d', 'LineWidth', 2);
plot(window_sizes, sensitivities, '-^', 'LineWidth', 2);
plot(window_sizes, specificities, '-v', 'LineWidth', 2);
plot(window_sizes, FP_rates, '-*', 'LineWidth', 2);
plot(window_sizes, FN_rates, '-x', 'LineWidth', 2);
hold off;
xlabel('Window Size');
ylabel('Score');
legend('Accuracy','F1 Clean','F1 Infected', 'Sensitivity','Specificity','FP Rate','FN Rate', 'Location', 'best');
title('Performance Trends');
grid on;

% Diagnostic analysis and patient visualization remain unchanged
analyze_model_behavior(features, binary_labels, patient_ids, batches, 24);
visualize_patient_data(features, binary_labels, patient_ids);

%% Helper Functions

function [precision, recall, f1] = calculate_metrics(cm)
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
    [rows, cols] = size(data);
    windowed_data = zeros(rows - window_size + 1, cols * window_size);
    for i = 1:(rows - window_size + 1)
        windowed_data(i, :) = reshape(data(i:i+window_size-1, :)', 1, []);
    end
end

function [windowed_features, windowed_labels, windowed_patient_ids, windowed_batches, windowed_is_pure] = create_rolling_window_per_patient(features, binary_labels, patient_ids, batches, window_size)
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
            % Determine the union label: label as infected (1) if any block is infected
            if any(curr_labels(j:j+window_size-1) == 1)
                label = 1;
            else
                label = 0;
            end
            % Check whether the window is pure (i.e., all blocks have the same label)
            if all(curr_labels(j:j+window_size-1) == curr_labels(j))
                is_pure = true;
            else
                is_pure = false;
            end
            
            batch_val = curr_batches(j+window_size-1);
            windowed_features = [windowed_features; window];
            windowed_labels = [windowed_labels; label];
            windowed_patient_ids = [windowed_patient_ids; curr_patient];
            windowed_batches = [windowed_batches; batch_val];
            windowed_is_pure = [windowed_is_pure; is_pure];
        end
    end
end

function analyze_model_behavior(features, binary_labels, patient_ids, batches, window_size)
    [windowed_features, windowed_labels, ~, ~, ~] = create_rolling_window_per_patient(features, binary_labels, patient_ids, batches, window_size);
    [windowed_features_scaled, ~, ~] = zscore(windowed_features);
    model = fitcensemble(windowed_features_scaled, windowed_labels, 'Method', 'Bag');
    imp = predictorImportance(model);
    
    % Reshape importance values to reflect the 7 features across the window
    imp_reshaped = reshape(imp, 7, window_size);  
    feature_names = {'R','G','B','C','R/C', 'G/C','B/C'};
    
    figure;
    heatmap(feature_names, cellstr("Batch-" + string(0:window_size-1)), imp_reshaped');
    title(sprintf('Feature Importance (Window Size: %d)', window_size));
    
    figure;
    for feat = 1:7  
        subplot(4,2,feat);
        [feature_windows, ~, ~, ~, ~] = create_rolling_window_per_patient(features(:,feat), binary_labels, patient_ids, batches, window_size);
        clean_windows = feature_windows(windowed_labels == 0, :);
        infected_windows = feature_windows(windowed_labels == 1, :);
        plot(0:window_size-1, mean(clean_windows,1), 'b', 'LineWidth', 2);
        hold on;
        plot(0:window_size-1, mean(infected_windows,1), 'r', 'LineWidth', 2);
        title(feature_names{feat});
        legend('Clean','Infected');
        grid on;
    end
    sgtitle(sprintf('Feature Patterns (Window Size: %d)', window_size));
end

function visualize_patient_data(features, binary_labels, patient_ids)
    feature_names = {'R','G','B','C','R/C','G/C','B/C'};
    
    figure('Position', [100 100 1200 800]);
    for i = 1:7
        subplot(4,2,i);
        clean_data = features(binary_labels == 0, i);
        infected_data = features(binary_labels == 1, i);
        boxplot([clean_data; infected_data], [zeros(size(clean_data)); ones(size(infected_data))], ...
                'Colors', 'br', 'Labels', {'Clean','Infected'});
        title(feature_names{i});
        grid on;
    end
    sgtitle('Feature Distributions by Infection Status');
    
    figure;
    subplot(2,1,1);
    [~, scores] = pca(zscore(features));
    scatter(scores(:,1), scores(:,2), 30, binary_labels, 'filled');
    title('PCA Projection');
    colorbar;
    
    subplot(2,1,2);
    imagesc(corr(features));
    colorbar;
    title('Feature Correlation Matrix');
    xticks(1:7);
    yticks(1:7);
    xticklabels(feature_names);
    yticklabels(feature_names);
    
    figure;
    unique_patients = unique(patient_ids);
    for p = 1:length(unique_patients)
        subplot(ceil(length(unique_patients)/2), 2, p);
        patient_data = features(patient_ids == unique_patients(p), :);
        plot(patient_data, 'LineWidth', 1.5);
        title(sprintf('Patient %d', unique_patients(p)));
        legend(feature_names, 'Location', 'eastoutside');
        grid on;
    end
    sgtitle('Patient Trajectories');
end
