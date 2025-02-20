%% Helper Functions

% Function to calculate precision, recall, and F1 score from a confusion matrix
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

% A non-patient-specific rolling window function (kept here for reference)
function windowed_data = create_rolling_window(data, window_size)
    [rows, cols] = size(data);
    windowed_data = zeros(rows - window_size + 1, cols * window_size);
    for i = 1:(rows - window_size + 1)
        windowed_data(i, :) = reshape(data(i:i+window_size-1, :)', 1, []);
    end
end

% New function: Create rolling window features per patient using batch ordering.
% Assumes "batches" is a numeric vector that orders the observations.
function [windowed_features, windowed_labels, windowed_patient_ids, windowed_batches] = create_rolling_window_per_patient(features, binary_labels, patient_ids, batches, window_size)
    windowed_features = [];
    windowed_labels = [];
    windowed_patient_ids = [];
    windowed_batches = [];
    unique_patients = unique(patient_ids);
    for p = 1:length(unique_patients)
        curr_patient = unique_patients(p);
        % Get indices for current patient and sort by the batch column
        idx = find(patient_ids == curr_patient);
        [sorted_batches, sortOrder] = sort(batches(idx));
        idx_sorted = idx(sortOrder);
        curr_features = features(idx_sorted, :);
        curr_labels = binary_labels(idx_sorted);
        curr_batches = batches(idx_sorted);
        n_rows = size(curr_features, 1);
        if n_rows < window_size
            continue;  % Skip patients with too few observations
        end
        for j = 1:(n_rows - window_size + 1)
            window = reshape(curr_features(j:j+window_size-1, :)', 1, []);
            label = curr_labels(j+window_size-1);  % Use the label from the last observation in the window
            batch_val = curr_batches(j+window_size-1);
            windowed_features = [windowed_features; window];
            windowed_labels = [windowed_labels; label];
            windowed_patient_ids = [windowed_patient_ids; curr_patient];
            windowed_batches = [windowed_batches; batch_val];
        end
    end
end

% Diagnostic analysis function using per-patient rolling windows with batch ordering
function analyze_model_behavior(features, binary_labels, patient_ids, batches, window_size)
    % Create per-patient rolling window features (for all features)
    [windowed_features, windowed_labels, ~, windowed_batches] = create_rolling_window_per_patient(features, binary_labels, patient_ids, batches, window_size);
    % Standardize the windowed features
    [windowed_features_scaled, ~, ~] = zscore(windowed_features);
    % Train an ensemble model using the Bag method (similar to XGBoost style)
    model = fitcensemble(windowed_features_scaled, windowed_labels, 'Method', 'Bag');
    % Get feature importance scores
    imp = predictorImportance(model);
    % Reshape importance scores assuming 4 features and the given window size
    imp_reshaped = reshape(imp, 4, window_size);
    
    % Plot a heatmap of feature importance across the window
    figure;
    heatmap({'RPercNormalized','GPercNormalized','BPercNormalized','CPercNormalized'}, ...
            cellstr("Batch-" + string(0:window_size-1)), imp_reshaped');
    title(sprintf('Feature Importance Across Batches (Window Size: %d)', window_size));
    xlabel('Features');
    ylabel('Batch Index within Window');
    
    % Plot average patterns for each feature for clean and infected classes
    figure;
    feature_names = {'RPercNormalized','GPercNormalized','BPercNormalized','CPercNormalized'};
    for feat = 1:4
        subplot(2,2,feat);
        % Create per-patient windows for a single feature
        [feature_windows, ~, ~, batch_windows] = create_rolling_window_per_patient(features(:,feat), binary_labels, patient_ids, batches, window_size);
        % Separate windows by class using the labels from the full windowing
        clean_windows = feature_windows(windowed_labels == 0, :);
        infected_windows = feature_windows(windowed_labels == 1, :);
        % Average the patterns across windows
        clean_pattern = mean(clean_windows, 1);
        infected_pattern = mean(infected_windows, 1);
        x_axis = 0:(window_size-1);
        plot(x_axis, clean_pattern, 'b-', 'LineWidth', 2, 'DisplayName', 'Clean');
        hold on;
        plot(x_axis, infected_pattern, 'r-', 'LineWidth', 2, 'DisplayName', 'Infected');
        hold off;
        title(sprintf('%s Pattern', feature_names{feat}));
        xlabel('Batch Index within Window');
        ylabel('Value');
        legend('Location', 'best');
        grid on;
    end
    sgtitle(sprintf('Average Patterns by Batch (Window Size: %d)', window_size));
end

%% Main Script

clc;
clearvars -except cleaneddata

if ~exist('cleaneddata', 'var')
    error('cleaneddata is not found in the workspace.');
end

data = cleaneddata;

% Use the "batch" column for ordering (ensure that this field is numeric)
batches = data.Batch;

% Extract normalized features
features = [data.RPercNormalized, data.GPercNormalized, data.BPercNormalized, data.CPercNormalized];

% Extract binary labels and patient IDs
binary_labels = data.infClassIDSA;
patient_ids = data.InStudyID;

% Remove rows with NaN or Inf values
valid_rows = ~any(isnan(features) | isinf(features), 2);
features = features(valid_rows, :);
binary_labels = binary_labels(valid_rows);
patient_ids = patient_ids(valid_rows);
batches = batches(valid_rows);

% =====================
% Handle Class Imbalance: Oversampling Minority Class
% =====================
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

unique_patients = unique(patient_ids);
n_patients = length(unique_patients);

% =====================
% Leave-One-Patient-Out Cross-Validation for Multiple Window Sizes
% Window sizes (in hours/samples): 6, 8, 12, 24
% =====================
window_sizes = [6, 8, 12, 24];
results = struct();

for w = 1:length(window_sizes)
    window_size = window_sizes(w);
    fprintf('\nPerforming Binary Classification (LOOCV) - Window Size: %d\n', window_size);
    % Create per-patient rolling window features using batch ordering
    [windowed_features, windowed_labels, windowed_patient_ids, windowed_batches] = ...
        create_rolling_window_per_patient(features, binary_labels, patient_ids, batches, window_size);
    
    binary_predictions = zeros(length(windowed_labels), 1);
    
    for i = 1:n_patients
        fprintf('Processing patient %d of %d\n', i, n_patients);
        test_idx = (windowed_patient_ids == unique_patients(i));
        train_idx = ~test_idx;
        
        X_train = windowed_features(train_idx, :);
        X_test = windowed_features(test_idx, :);
        y_train = windowed_labels(train_idx);
        
        % Standardize the training features and apply the same scaling to test features
        [X_train_scaled, mu, sigma] = zscore(X_train);
        X_test_scaled = (X_test - mu) ./ sigma;
        
        % Train an ensemble model (Bag method)
        model = fitcensemble(X_train_scaled, y_train, 'Method', 'Bag');
        binary_predictions(test_idx) = predict(model, X_test_scaled);
    end
    
    % Evaluate performance
    binary_cm = confusionmat(windowed_labels, binary_predictions);
    [precision, recall, f1] = calculate_metrics(binary_cm);
    accuracy = sum(diag(binary_cm)) / sum(binary_cm(:));
    
    fprintf('\n=== Results for Window Size %d ===\n', window_size);
    fprintf('Accuracy: %.4f\n', accuracy);
    fprintf('Precision (Class 0, Class 1): %.4f, %.4f\n', precision(1), precision(2));
    fprintf('Recall (Class 0, Class 1): %.4f, %.4f\n', recall(1), recall(2));
    fprintf('F1 Score (Class 0, Class 1): %.4f, %.4f\n', f1(1), f1(2));
    
    results(w).window_size = window_size;
    results(w).accuracy = accuracy;
    results(w).f1 = f1;
    results(w).confusion_matrix = binary_cm;
    
    % Visualize the confusion matrix
    figure;
    cm_chart = confusionchart(binary_cm, {'Clean', 'Infected'});
    cm_chart.Title = sprintf('Confusion Matrix (Window Size: %d)', window_size);
    cm_chart.ColumnSummary = 'column-normalized';
    cm_chart.RowSummary = 'row-normalized';
end

fprintf('\nComparison of Different Window Sizes:\n');
accuracy_values = zeros(1, length(window_sizes));
f1_values_class0 = zeros(1, length(window_sizes));
f1_values_class1 = zeros(1, length(window_sizes));

for w = 1:length(window_sizes)
    accuracy_values(w) = results(w).accuracy;
    f1_values_class0(w) = results(w).f1(1);
    f1_values_class1(w) = results(w).f1(2);
    fprintf('Window Size %d - Accuracy: %.4f, F1 Score: %.4f, %.4f\n', results(w).window_size, results(w).accuracy, results(w).f1(1), results(w).f1(2));
end

figure;
plot(window_sizes, accuracy_values, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Accuracy');
hold on;
plot(window_sizes, f1_values_class0, '-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'F1 Score (Class 0)');
plot(window_sizes, f1_values_class1, '-d', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'F1 Score (Class 1)');
hold off;
xlabel('Window Size (hours)');
ylabel('Performance');
title('Accuracy & F1-Score Trends');
legend('Location', 'southeast');
grid on;

% Run the diagnostic analysis using a window size of 24 (per-patient, batch-ordered)
analyze_model_behavior(features, binary_labels, patient_ids, batches, 24);

%% Patient Visualization Functions

function visualize_patient_data(features, binary_labels, patient_ids)
    feature_names = {'RPercNormalized','GPercNormalized','BPercNormalized','CPercNormalized'};
    
    % 1. Boxplots: Feature Distributions by Patient and Infection Status
    figure('Position', [100 100 1200 800]);
    for i = 1:4
        subplot(2,2,i);
        clean_data = features(binary_labels == 0, i);
        clean_patients = patient_ids(binary_labels == 0);
        boxplot(clean_data, clean_patients, 'Positions', 1:length(unique(clean_patients)), 'Colors', 'b', 'Symbol', '');
        hold on;
        infected_data = features(binary_labels == 1, i);
        infected_patients = patient_ids(binary_labels == 1);
        boxplot(infected_data, infected_patients, 'Positions', 1:length(unique(infected_patients)), 'Colors', 'r', 'Symbol', '');
        title([feature_names{i} ' Distribution']);
        xlabel('Patient ID');
        ylabel('Value');
        grid on;
    end
    sgtitle('Feature Distributions by Patient (Blue=Clean, Red=Infected)');
    
    % 2. Feature Importance and Correlation Analysis
    figure('Position', [100 100 800 600]);
    [features_scaled, ~, ~] = zscore(features);
    model = fitcensemble(features_scaled, binary_labels, 'Method', 'Bag');
    importance = predictorImportance(model);
    subplot(2,1,1);
    bar(importance);
    title('Feature Importance Scores');
    xticks(1:4);
    xticklabels(feature_names);
    ylabel('Importance Score');
    grid on;
    subplot(2,1,2);
    correlation_matrix = corr(features);
    imagesc(correlation_matrix);
    colorbar;
    title('Feature Correlation Matrix');
    xticks(1:4);
    yticks(1:4);
    xticklabels(feature_names);
    yticklabels(feature_names);
    
    % 3. Patient Trajectory Visualization
    figure('Position', [100 100 1200 600]);
    unique_patients = unique(patient_ids);
    for p = 1:length(unique_patients)
        patient_idx = patient_ids == unique_patients(p);
        patient_features = features(patient_idx, :);
        patient_labels = binary_labels(patient_idx);
        subplot(ceil(length(unique_patients)/2), 2, p);
        plot(patient_features, 'LineWidth', 1.5);
        hold on;
        if any(patient_labels == 1)
            ylimits = ylim;
            infected_idx = find(patient_labels == 1);
            for j = 1:length(infected_idx)
                patch([infected_idx(j)-0.5, infected_idx(j)+0.5, infected_idx(j)+0.5, infected_idx(j)-0.5], ...
                      [ylimits(1), ylimits(1), ylimits(2), ylimits(2)], 'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
            end
        end
        title(['Patient ' num2str(unique_patients(p))]);
        xlabel('Time Point');
        ylabel('Value');
        legend(feature_names, 'Location', 'best');
        grid on;
    end
    sgtitle('Patient Trajectories (Red Background = Infected Period)');
end

% Usage example:
visualize_patient_data(features, binary_labels, patient_ids);
