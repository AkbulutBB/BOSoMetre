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

% ========================
% LOOCV + Balancing + Winsorization + Optimized RBF SVM (No PCA)
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
% Handle Class Imbalance: Limited Oversampling
% =====================
fprintf('\nBalancing the Dataset...\n');
class0_idx = find(binary_labels == 0);
class1_idx = find(binary_labels == 1);

% Limit oversampling (max 1.5x of minority class)
max_samples = min(ceil(1.5 * min(length(class0_idx), length(class1_idx))), max(length(class0_idx), length(class1_idx)));

% Randomly oversample (but limit max duplication to avoid overfitting)
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
% Leave-One-Patient-Out Cross-Validation
% =====================
fprintf('\nPerforming Binary Classification (LOOCV)\n');
binary_predictions = zeros(length(binary_labels), 1);
best_C = 1; best_gamma = 0.1;  % Default hyperparams

% Define grid search parameters
C_values = [0.1, 1, 10, 100];  % Regularization
gamma_values = [0.01, 0.1, 1, 10];  % Kernel Scale

for i = 1:n_patients
    fprintf('Processing patient %d of %d\n', i, n_patients);
    
    % Train-test split
    test_idx = patient_ids == unique_patients(i);
    train_idx = ~test_idx;
    
    X_train = features(train_idx, :);
    X_test = features(test_idx, :);
    y_train = binary_labels(train_idx);
    
    % Standardize features
    [X_train_scaled, mu, sigma] = zscore(X_train);
    X_test_scaled = (X_test - mu) ./ sigma;

    % =====================
    % Grid Search for RBF SVM
    % =====================
    best_acc = 0;
    for C = C_values
        for gamma = gamma_values
            model = fitcsvm(X_train_scaled, y_train, ...
                'KernelFunction', 'rbf', ...
                'BoxConstraint', C, ...
                'KernelScale', gamma, ...
                'Standardize', true);
            
            % LOOCV within training set
            cv_model = crossval(model, 'KFold', 5);
            acc = 1 - kfoldLoss(cv_model);
            
            if acc > best_acc
                best_acc = acc;
                best_C = C;
                best_gamma = gamma;
            end
        end
    end

    % Train best SVM on full training set
    final_model = fitcsvm(X_train_scaled, y_train, ...
        'KernelFunction', 'rbf', ...
        'BoxConstraint', best_C, ...
        'KernelScale', best_gamma, ...
        'Standardize', true);
    
    % Predict on test set
    binary_predictions(test_idx) = predict(final_model, X_test_scaled);
end

% =====================
% Evaluate Performance
% =====================
fprintf('\n=== Results for Optimized RBF SVM (Winsorized + Balanced) ===\n');
binary_cm = confusionmat(binary_labels, binary_predictions);
[precision, recall, f1] = calculate_metrics(binary_cm);
accuracy = sum(diag(binary_cm)) / sum(binary_cm(:));

fprintf('Final Hyperparameters: C = %.2f, Gamma = %.2f\n', best_C, best_gamma);
fprintf('Accuracy: %.4f\n', accuracy);
fprintf('Precision (Class 0, Class 1): %.4f, %.4f\n', precision(1), precision(2));
fprintf('Recall (Class 0, Class 1):    %.4f, %.4f\n', recall(1), recall(2));
fprintf('F1 Score (Class 0, Class 1):  %.4f, %.4f\n', f1(1), f1(2));

% =====================
% Plot Confusion Matrix
% =====================
figure;
confusionchart(binary_cm, {'Clean', 'Infected'}, 'Title', 'Optimized RBF SVM - Binary Classification');

% Print final class distribution
fprintf('\nFinal Balanced Class Distribution (Binary):\n');
binary_dist = histcounts(binary_labels, 'BinMethod', 'integers');
fprintf('Class 0 (Clean): %d (%.1f%%)\n', binary_dist(1), 100 * binary_dist(1) / sum(binary_dist));
fprintf('Class 1 (Infected): %d (%.1f%%)\n', binary_dist(2), 100 * binary_dist(2) / sum(binary_dist));
