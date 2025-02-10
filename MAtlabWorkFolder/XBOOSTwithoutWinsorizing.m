% Helper Functions
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

function dt = parse_datetime(datestr)
    % Format: 240930_0050 (YYMMDD_HHMM)
    yr = str2double('20' + datestr(1:2));
    mn = str2double(datestr(3:4));
    dy = str2double(datestr(5:6));
    hr = str2double(datestr(8:9));
    mt = str2double(datestr(10:11));
    dt = datetime(yr, mn, dy, hr, mt, 0);
end

function analyze_model_behavior(features, binary_labels, datetimes, window_size)
    % Create windowed features
    windowed_features = create_rolling_window(features, window_size);
    windowed_labels = binary_labels(window_size:end);
    
    % Standardize all features
    [windowed_features_scaled, ~, ~] = zscore(windowed_features);
    
    % Train a model on all data for analysis
    model = fitcensemble(windowed_features_scaled, windowed_labels, 'Method', 'Bag');
    
    % Get feature importance scores
    imp = predictorImportance(model);
    
    % Reshape importance scores to match original features and time windows
    imp_reshaped = reshape(imp, 4, window_size);
    
    % Plot feature importance heatmap
    figure;
    heatmap({'RPerc', 'GPerc', 'BPerc', 'CPerc'}, ...
            cellstr("T-" + string((0:window_size-1))), imp_reshaped');
    title(sprintf('Feature Importance Across Time Windows (Window Size: %d)', window_size));
    xlabel('Features');
    ylabel('Time Steps');
    
    % Plot temporal patterns for each class
    figure;
    for feat = 1:4
        subplot(2,2,feat);
        
        % Create windows for both features and timestamps
        feature_windows = create_rolling_window(features(:,feat), window_size);
        time_windows = create_rolling_window(datenum(datetimes), window_size);
        
        % Separate data by class
        clean_windows = feature_windows(windowed_labels==0, :);
        clean_times = time_windows(windowed_labels==0, :);
        infected_windows = feature_windows(windowed_labels==1, :);
        infected_times = time_windows(windowed_labels==1, :);
        
        % Calculate mean patterns for each class
        clean_pattern = mean(clean_windows, 1);
        clean_times_mean = datetime(mean(clean_times, 1), 'ConvertFrom', 'datenum');
        infected_pattern = mean(infected_windows, 1);
        infected_times_mean = datetime(mean(infected_times, 1), 'ConvertFrom', 'datenum');
        
        % Plot patterns with real timestamps
        plot(clean_times_mean, clean_pattern, 'b-', 'LineWidth', 2, 'DisplayName', 'Clean');
        hold on;
        plot(infected_times_mean, infected_pattern, 'r-', 'LineWidth', 2, 'DisplayName', 'Infected');
        hold off;
        
        feature_names = {'RPerc', 'GPerc', 'BPerc', 'CPerc'};
        title(sprintf('%s Temporal Pattern', feature_names{feat}));
        xlabel('Time');
        ylabel('Value');
        legend('Location', 'best');
        grid on;
        
        % Format x-axis to show time properly
        xtickformat('dd-MMM HH:mm')
        xtickangle(45)
    end
    sgtitle(sprintf('Average Temporal Patterns (Window Size: %d)', window_size));
end

% ========================
% Main Script
% ========================
clc;
clearvars -except AllCleanedDataWithAllPatients

if ~exist('AllCleanedDataWithAllPatients', 'var')
    error('AllCleanedDataWithAllPatients is not found in the workspace.');
end

data = AllCleanedDataWithAllPatients;

% Parse datetime information
timestamps = data.DateTime;  % adjust column name if needed
datetimes = arrayfun(@(x) parse_datetime(char(x)), timestamps, 'UniformOutput', false);
datetimes = [datetimes{:}]';  % Convert cell array to datetime array

% Extract features using raw percentage values
features = [data.RPerc, data.GPerc, data.BPerc, data.CPerc];

% Extract labels and patient IDs
binary_labels = data.infClassIDSA;
patient_ids = data.InStudyID;

% Remove NaN values
valid_rows = ~any(isnan(features) | isinf(features), 2);
features = features(valid_rows, :);
binary_labels = binary_labels(valid_rows);
patient_ids = patient_ids(valid_rows);
datetimes = datetimes(valid_rows);

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
datetimes = datetimes(balanced_idx);

fprintf('Final Balanced Dataset: Class 0 = %d, Class 1 = %d\n', sum(binary_labels == 0), sum(binary_labels == 1));

% Get unique patient IDs
unique_patients = unique(patient_ids);
n_patients = length(unique_patients);

% =====================
% Leave-One-Patient-Out Cross-Validation for Different Window Sizes
% =====================
window_sizes = [24];
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

% Run the diagnostic analysis for the best performing window size (24)
analyze_model_behavior(features, binary_labels, datetimes, 24);

function visualize_patient_data(features, binary_labels, patient_ids)
    % Create multiple visualizations to understand the data patterns
    feature_names = {'RPerc', 'GPerc', 'BPerc', 'CPerc'};
    
    % 1. Feature Distributions by Patient and Infection Status
    figure('Position', [100 100 1200 800]);
    
    for i = 1:4
        subplot(2,2,i);
        
        % Plot clean samples
        clean_data = features(binary_labels == 0, i);
        clean_patients = patient_ids(binary_labels == 0);
        boxplot(clean_data, clean_patients, 'Positions', 1:length(unique(clean_patients)), ...
                'Colors', 'b', 'Symbol', '');
        hold on;
        
        % Plot infected samples
        infected_data = features(binary_labels == 1, i);
        infected_patients = patient_ids(binary_labels == 1);
        boxplot(infected_data, infected_patients, 'Positions', 1:length(unique(infected_patients)), ...
                'Colors', 'r', 'Symbol', '');
        
        title([feature_names{i} ' Distribution']);
        xlabel('Patient ID');
        ylabel('Value');
        grid on;
    end
    sgtitle('Feature Distributions by Patient (Blue=Clean, Red=Infected)');
    
    % 2. Feature Importance Analysis
    figure('Position', [100 100 800 600]);
    
    % Standardize features
    [features_scaled, ~, ~] = zscore(features);
    
    % Train a model for importance analysis
    model = fitcensemble(features_scaled, binary_labels, 'Method', 'Bag');
    importance = predictorImportance(model);
    
    subplot(2,1,1);
    bar(importance);
    title('Feature Importance Scores');
    xticks(1:4);
    xticklabels(feature_names);
    ylabel('Importance Score');
    grid on;
    
    % 3. Feature Correlation Analysis
    subplot(2,1,2);
    correlation_matrix = corr(features);
    imagesc(correlation_matrix);
    colorbar;
    title('Feature Correlation Matrix');
    xticks(1:4);
    yticks(1:4);
    xticklabels(feature_names);
    yticklabels(feature_names);
    
    % 4. Patient Trajectory Visualization
    figure('Position', [100 100 1200 600]);
    unique_patients = unique(patient_ids);
    
    for p = 1:length(unique_patients)
        patient_idx = patient_ids == unique_patients(p);
        patient_features = features(patient_idx, :);
        patient_labels = binary_labels(patient_idx);
        
        % Create patient subplot
        subplot(ceil(length(unique_patients)/2), 2, p);
        
        % Plot all features for this patient
        plot(patient_features, 'LineWidth', 1.5);
        hold on;
        
        % Highlight infection status
        infected_regions = patient_labels == 1;
        if any(infected_regions)
            % Add red background for infected periods
            ylimits = ylim;
            x = 1:length(patient_labels);
            infected_idx = find(infected_regions);
            
            % Create patches for infected regions
            for i = 1:length(infected_idx)
                patch([infected_idx(i)-0.5 infected_idx(i)+0.5 infected_idx(i)+0.5 infected_idx(i)-0.5], ...
                      [ylimits(1) ylimits(1) ylimits(2) ylimits(2)], ...
                      'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
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