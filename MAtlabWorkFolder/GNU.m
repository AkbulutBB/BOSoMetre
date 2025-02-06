% =========================================================================
% GNU-Compatible Machine Learning Models: Naïve Bayes, SVM, Logistic Regression
% LOOCV + Rolling Window for Time-Series Classification
% =========================================================================

clc;
clearvars -except cleaneddata;

if ~exist('cleaneddata', 'var')
    error('cleaneddata is not found in the workspace.');
end

data = cleaneddata;

% =========================================================================
% Define Rolling Window Sizes & Other Parameters
% =========================================================================
window_sizes = [6, 12, 24];  % Test different rolling window sizes
winsor_level = [1, 99];  % Winsorization at 1-99 percentile

% Extract Features & Labels
features = [data.RPercNormalized, data.GPercNormalized, data.BPercNormalized, data.CPercNormalized];
binary_labels = data.infClassIDSA;
patient_ids = data.InStudyID;

% Apply Winsorization
features = winsorize_features(features, winsor_level(1), winsor_level(2));

% =========================================================================
% Define Models (GNU-Compatible)
% =========================================================================
models = {'Naïve Bayes', 'SVM (Linear)', 'Logistic Regression'};
n_models = length(models);

% Store results for different window sizes
results = struct();

% =========================================================================
% Iterate Over Different Rolling Window Sizes
% =========================================================================
for ws = 1:length(window_sizes)
    window_size = window_sizes(ws);
    fprintf('\n=== Processing Rolling Window Size: %d ===\n', window_size);

    % Convert to Rolling Windows
    [X, Y, PatientIDs] = create_rolling_window(features, binary_labels, patient_ids, window_size);

    % Standardize Features (Z-score Normalization)
    X = normalize(X, 2);

    % Reshape into tabular format for models: [Samples, Features]
    X = reshape(X, size(X,1), []);

    % Convert Labels to Categorical
    Y = categorical(Y);

    % Get Unique Patient IDs for LOOCV
    unique_patients = unique(PatientIDs);
    n_patients = length(unique_patients);

    % Initialize Predictions Storage
    predictions = cell(n_models, 1);
    for m = 1:n_models
        predictions{m} = categorical(zeros(size(Y)));
    end

    fprintf('\nPerforming LOOCV for GNU-Compatible Models...\n');

    % =========================================================================
    % Leave-One-Patient-Out Cross-Validation (LOOCV)
    % =========================================================================
    for i = 1:n_patients
        fprintf('Processing patient %d of %d\n', i, n_patients);
        test_idx = PatientIDs == unique_patients(i);
        train_idx = ~test_idx;

        X_train = X(train_idx, :);
        Y_train = Y(train_idx);
        X_test = X(test_idx, :);

        % Train Naïve Bayes
        nb_model = fitcnb(X_train, Y_train);
        predictions{1}(test_idx) = predict(nb_model, X_test);

        % Train Linear SVM
        svm_model = fitcsvm(X_train, Y_train, 'KernelFunction', 'linear');
        predictions{2}(test_idx) = predict(svm_model, X_test);

        % Train Logistic Regression
        logit_model = fitclinear(X_train, Y_train, 'Learner', 'logistic');
        predictions{3}(test_idx) = predict(logit_model, X_test);
    end

    % =========================================================================
    % Evaluate Model Performance
    % =========================================================================
    fprintf('\nEvaluating Model Performance...\n');

    for m = 1:n_models
        fprintf('\n=== Results for %s (Window Size: %d) ===\n', models{m}, window_size);

        conf_mat = confusionmat(Y, predictions{m});
        [precision, recall, f1] = calculate_metrics(conf_mat);
        accuracy = sum(diag(conf_mat)) / sum(conf_mat(:));

        fprintf('Accuracy: %.4f\n', accuracy);
        fprintf('Precision (Class 0, Class 1): %.4f, %.4f\n', precision(1), precision(2));
        fprintf('Recall (Class 0, Class 1): %.4f, %.4f\n', recall(1), recall(2));
        fprintf('F1 Score (Class 0, Class 1): %.4f, %.4f\n', f1(1), f1(2));

        % Store results
        results(ws).window_size = window_size;
        results(ws).model(m).name = models{m};
        results(ws).model(m).accuracy = accuracy;
        results(ws).model(m).precision = precision;
        results(ws).model(m).recall = recall;
        results(ws).model(m).f1_score = f1;

        % Plot Confusion Matrix
        figure;
        confusionchart(conf_mat, {'Clean', 'Infected'}, ...
            'Title', sprintf('%s Classification (Window Size: %d)', models{m}, window_size));
    end
end

% =========================================================================
% Helper Functions
% =========================================================================

% -----------------------
% Winsorization Function
% -----------------------
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

% -----------------------
% Rolling Window Function
% -----------------------
function [windowed_X, windowed_Y, windowed_PatientIDs] = create_rolling_window(data, labels, patient_ids, window_size)
    num_samples = size(data, 1) - window_size + 1;
    num_features = size(data, 2);
    windowed_X = zeros(num_samples, window_size, num_features);
    windowed_Y = zeros(num_samples, 1);
    windowed_PatientIDs = zeros(num_samples, 1);
    
    for i = 1:num_samples
        windowed_X(i, :, :) = data(i:i+window_size-1, :);
        windowed_Y(i) = labels(i+window_size-1); % Assign last value as target
        windowed_PatientIDs(i) = patient_ids(i+window_size-1);
    end
end

% -----------------------
% Metrics Calculation
% -----------------------
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
