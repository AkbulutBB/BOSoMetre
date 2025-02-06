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

% Feature augmentation function
function augmented_features = augment_features(features)
    R = features(:,1); G = features(:,2); B = features(:,3); C = features(:,4);
    RG_ratio = R ./ G; RB_ratio = R ./ B; RC_ratio = R ./ C;
    GB_ratio = G ./ B; GC_ratio = G ./ C; BC_ratio = B ./ C;
    augmented_features = [features, RG_ratio, RB_ratio, RC_ratio, GB_ratio, GC_ratio, BC_ratio];
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

% =====================
% Main Analysis Script (LOOCV with PCA)
% =====================
clc;
clearvars -except cleaneddata

if ~exist('cleaneddata', 'var')
    error('cleaneddata is not found in the workspace.');
end

data = cleaneddata;

% Define classifiers
classifiers = {
    'Simple SVM', ...
    'RBF SVM', ...
    'LDA', ...
    'Simple KNN', ...
    'Subspace KNN', ...
    'Ensemble KNN'
};

% Extract features
features = [data.RPercNormalized, data.GPercNormalized, data.BPercNormalized, data.CPercNormalized];

% Apply winsorization
winsor_level = [1, 99];
features = winsorize_features(features, winsor_level(1), winsor_level(2));

% Apply feature augmentation (ratios)
features = augment_features(features);

% Remove rows with NaN or Inf values
valid_rows = ~any(isnan(features) | isinf(features), 2);
features = features(valid_rows, :);
binary_labels = data.infClassIDSA(valid_rows);
patient_ids = data.InStudyID(valid_rows);

% =====================
% PCA: Dimensionality Reduction
% =====================
fprintf('\nPerforming PCA on Feature Set...\n');

% Standardize features before PCA
features_standardized = zscore(features);

% Perform PCA
[coeff, score, ~, ~, explained] = pca(features_standardized);

% Determine how many components to keep (95% variance threshold)
cumulative_variance = cumsum(explained);
num_components = find(cumulative_variance >= 95, 1);

fprintf('Selected %d principal components to explain %.2f%% of the variance.\n', num_components, cumulative_variance(num_components));

% Reduce feature set to the chosen principal components
features_pca = score(:, 1:num_components);

% =====================
% Leave-One-Patient-Out Cross-Validation (LOOCV)
% =====================
fprintf('\nPerforming Binary Classification (LOOCV)\n');

% Get unique patient IDs
unique_patients = unique(patient_ids);
n_patients = length(unique_patients);
n_classifiers = length(classifiers);
binary_predictions = zeros(length(binary_labels), n_classifiers);

% LOOCV process
for i = 1:n_patients
    fprintf('Processing patient %d of %d\n', i, n_patients);
    
    % Split train and test
    test_idx = patient_ids == unique_patients(i);
    train_idx = ~test_idx;
    
    X_train = features_pca(train_idx, :);
    X_test = features_pca(test_idx, :);
    y_train = binary_labels(train_idx);
    
    % Standardize features
    [X_train_scaled, mu, sigma] = zscore(X_train);
    X_test_scaled = (X_test - mu) ./ sigma;
    
    % Train and predict with each classifier
    for j = 1:n_classifiers
        try
            switch classifiers{j}
                case 'Simple SVM'
                    model = fitclinear(X_train_scaled, y_train, 'Learner', 'svm', 'Regularization', 'ridge');
                case 'RBF SVM'
                    model = fitcsvm(X_train_scaled, y_train, 'KernelFunction', 'rbf', 'KernelScale', 'auto', 'Standardize', true);
                case 'LDA'
                    model = fitcdiscr(X_train_scaled, y_train, 'DiscrimType', 'linear');
                case 'Simple KNN'
                    model = fitcknn(X_train_scaled, y_train, 'NumNeighbors', 10, 'Distance', 'euclidean');
                case 'Subspace KNN'
                    model = fitcensemble(X_train_scaled, y_train, 'Method', 'Subspace', 'NumLearningCycles', 60, 'Learners', templateKNN('NumNeighbors', 3));
                case 'Ensemble KNN'
                    knn1 = fitcknn(X_train_scaled, y_train, 'NumNeighbors', 10, 'Distance', 'euclidean');
                    knn2 = fitcknn(X_train_scaled, y_train, 'NumNeighbors', 10, 'Distance', 'cityblock');
                    knn3 = fitcknn(X_train_scaled, y_train, 'NumNeighbors', 10, 'Distance', 'cosine');
                    
                    pred1 = predict(knn1, X_test_scaled);
                    pred2 = predict(knn2, X_test_scaled);
                    pred3 = predict(knn3, X_test_scaled);
                    binary_predictions(test_idx, j) = mode([pred1, pred2, pred3], 2);
                    continue;
            end
            if ~strcmp(classifiers{j}, 'Ensemble KNN')
                binary_predictions(test_idx, j) = predict(model, X_test_scaled);
            end
        catch ME
            fprintf('Classification failed for %s: %s\n', classifiers{j}, ME.message);
            binary_predictions(test_idx, j) = mode(y_train);
        end
    end
end

% =====================
% Display Results
% =====================
fprintf('\n=== Results for Binary Classification (LOOCV with PCA) ===\n');

for j = 1:n_classifiers
    fprintf('\n=== %s ===\n', classifiers{j});
    
    binary_cm = confusionmat(binary_labels, binary_predictions(:,j));
    [precision, recall, f1] = calculate_metrics(binary_cm);
    accuracy = sum(diag(binary_cm)) / sum(binary_cm(:));
    
    fprintf('Accuracy: %.4f\n', accuracy);
    fprintf('Precision (Class 0, Class 1): %.4f, %.4f\n', precision(1), precision(2));
    fprintf('Recall (Class 0, Class 1):    %.4f, %.4f\n', recall(1), recall(2));
    fprintf('F1 Score (Class 0, Class 1):  %.4f, %.4f\n', f1(1), f1(2));
    
    % Display confusion matrix
    figure('Position', [100, 100, 600, 500]);
    confusionchart(binary_cm, {'Clean', 'Infected'}, ...
        'Title', sprintf('%s - Binary Classification (PCA)', classifiers{j}));
end

% =====================
% Print Class Distribution
% =====================
fprintf('\nClass Distribution (Binary):\n');
binary_dist = histcounts(binary_labels, 'BinMethod', 'integers');
fprintf('Class 0 (Clean): %d (%.1f%%)\n', binary_dist(1), 100 * binary_dist(1) / sum(binary_dist));
fprintf('Class 1 (Infected): %d (%.1f%%)\n', binary_dist(2), 100 * binary_dist(2) / sum(binary_dist));

% =====================
% PCA Variance Explained Plot
% =====================
figure;
plot(1:length(cumulative_variance), cumulative_variance, '-o');
xlabel('Number of Principal Components');
ylabel('Cumulative Variance Explained');
title('PCA: Variance Explained');
grid on;
