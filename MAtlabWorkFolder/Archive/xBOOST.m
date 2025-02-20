% Winsorization function
function winsorized_data = winsorize_features(data, lower_percentile, upper_percentile)
    winsorized_data = data;
    for col = 1:size(data, 2)
        valid_data = data(:, col);
        valid_data = valid_data(~isnan(valid_data));
        lower_bound = prctile(valid_data, lower_percentile);
        upper_bound = prctile(valid_data, upper_percentile);
        winsorized_data(:, col) = max(min(data(:, col), upper_bound), lower_bound);
    end
end

% Metrics calculation function
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

% Function to create rolling window features
function windowed_data = create_rolling_window(data, window_size)
    [rows, cols] = size(data);
    windowed_data = zeros(rows - window_size + 1, cols * window_size);
    for i = 1:(rows - window_size + 1)
        windowed_data(i, :) = reshape(data(i:i+window_size-1, :)', 1, []);
    end
end

% ========================
% PCA + LOOCV + Winsorization + Rolling Window + XGBoost + Visualization
% ========================
clc;
clearvars -except cleaneddata

if ~exist('cleaneddata', 'var')
    error('cleaneddata is not found in the workspace.');
end

data = cleaneddata;

% Extract features
features = [data.RPercNormalized, data.GPercNormalized, data.BPercNormalized, data.CPercNormalized];

% =====================
% Winsorization (1-99 percentile)
% =====================
fprintf('\nApplying Winsorization (1-99 percentile)...\n');
winsor_level = [1, 99];
features = winsorize_features(features, winsor_level(1), winsor_level(2));

% Extract labels and patient IDs
binary_labels = data.infClassIDSA;
patient_ids = data.InStudyID;

% Remove NaN values
valid_rows = ~any(isnan(features) | isinf(features), 2);
features = features(valid_rows, :);
binary_labels = binary_labels(valid_rows);
patient_ids = patient_ids(valid_rows);

% =====================
% Handle Class Imbalance: Oversampling Minority Class
% =====================
fprintf('\nBalancing the Dataset...\n');
class0_idx = find(binary_labels == 0);
class1_idx = find(binary_labels == 1);

% Determine class sizes
max_samples = max(length(class0_idx), length(class1_idx));

% Oversample minority class
rng(42);
class0_oversampled = class0_idx(randi(length(class0_idx), max_samples, 1));
class1_oversampled = class1_idx(randi(length(class1_idx), max_samples, 1));

% Combine balanced dataset
balanced_idx = [class0_oversampled; class1_oversampled];
features = features(balanced_idx, :);
binary_labels = binary_labels(balanced_idx);
patient_ids = patient_ids(balanced_idx);

fprintf('Final Balanced Dataset: Class 0 = %d, Class 1 = %d\n', sum(binary_labels == 0), sum(binary_labels == 1));

% Get unique patient IDs
unique_patients = unique(patient_ids);
n_patients = length(unique_patients);

% =====================
% Leave-One-Patient-Out Cross-Validation for Different Window Sizes
% =====================
window_sizes = [1, 4, 6, 8, 12, 24];
results = struct();

for w = 1:length(window_sizes)
    window_size = window_sizes(w);
    fprintf('\nPerforming Binary Classification (LOOCV) - Rolling Window Size: %d\n', window_size);
    
    % Create rolling window features
    windowed_features = create_rolling_window(features, window_size);
    windowed_labels = binary_labels(window_size:end);
    windowed_patient_ids = patient_ids(window_size:end);
    
    binary_predictions = zeros(length(windowed_labels), 1);
    
    for i = 1:n_patients
        fprintf('Processing patient %d of %d\n', i, n_patients);
        
        % Train-test split
        test_idx = windowed_patient_ids == unique_patients(i);
        train_idx = ~test_idx;
        
        X_train = windowed_features(train_idx, :);
        X_test = windowed_features(test_idx, :);
        y_train = windowed_labels(train_idx);
        
        % Standardize features
        [X_train_scaled, mu, sigma] = zscore(X_train);
        X_test_scaled = (X_test - mu) ./ sigma;
        
        % Train XGBoost model
        model = fitcensemble(X_train_scaled, y_train, 'Method', 'Bag');
        
        % Predict on test set
        binary_predictions(test_idx) = predict(model, X_test_scaled);
    end
    
    % =====================
    % Evaluate Performance
    % =====================
    binary_cm = confusionmat(windowed_labels, binary_predictions);
    [precision, recall, f1] = calculate_metrics(binary_cm);
    accuracy = sum(diag(binary_cm)) / sum(binary_cm(:));
    
    fprintf('\n=== Results for Rolling Window XGBoost (Window Size: %d) ===\n', window_size);
    fprintf('Accuracy: %.4f\n', accuracy);
    fprintf('Precision (Class 0, Class 1): %.4f, %.4f\n', precision(1), precision(2));
    fprintf('Recall (Class 0, Class 1): %.4f, %.4f\n', recall(1), recall(2));
    fprintf('F1 Score (Class 0, Class 1): %.4f, %.4f\n', f1(1), f1(2));
    
    % Store results for visualization
    results(w).window_size = window_size;
    results(w).accuracy = accuracy;
    results(w).f1 = f1;
    results(w).confusion_matrix = binary_cm;
    
    % =====================
    % Confusion Matrix Visualization
    % =====================
    figure;
    cm_chart = confusionchart(binary_cm, {'Clean', 'Infected'});
    cm_chart.Title = sprintf('Confusion Matrix (Window Size: %d)', window_size);
    cm_chart.ColumnSummary = 'column-normalized';
    cm_chart.RowSummary = 'row-normalized';
end

% =====================
% Compare Window Sizes
% =====================
fprintf('\nComparison of Different Window Sizes:\n');
accuracy_values = zeros(1, length(window_sizes));
f1_values_class0 = zeros(1, length(window_sizes));
f1_values_class1 = zeros(1, length(window_sizes));

for w = 1:length(window_sizes)
    accuracy_values(w) = results(w).accuracy;
    f1_values_class0(w) = results(w).f1(1);
    f1_values_class1(w) = results(w).f1(2);
    
    fprintf('Window Size %d - Accuracy: %.4f, F1 Score: %.4f, %.4f\n', ...
        results(w).window_size, results(w).accuracy, results(w).f1(1), results(w).f1(2));
end

% =====================
% Performance Trend Graph
% =====================
figure;
plot(window_sizes, accuracy_values, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Accuracy');
hold on;
plot(window_sizes, f1_values_class0, '-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'F1 Score (Class 0)');
plot(window_sizes, f1_values_class1, '-d', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'F1 Score (Class 1)');
hold off;

xlabel('Rolling Window Size');
ylabel('Performance');
title('Accuracy & F1-Score Trends');
legend('Location', 'southeast');
grid on;
