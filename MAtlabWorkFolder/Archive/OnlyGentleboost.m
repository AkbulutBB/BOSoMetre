%% GentleBoost with 12- and 24-Block Windows: Model Training, Threshold Tuning, and Hyperparameter Optimization
clc;
clearvars -except cleaneddatamad

if ~exist('cleaneddatamad', 'var')
    error('cleaneddatamad is not found in the workspace.');
end

%% User-Defined Parameters
% Use GentleBoost only
model_method = 'GentleBoost';
% Enable hyperparameter tuning (this may increase training time)
performHyperparameterTuning = false;
% Use SMOTE for oversampling (set to true if you want to balance the dataset synthetically)
useSMOTE = true;
% Decision threshold for labeling an instance as infected (class 1)
% (Lowering the threshold below 0.5 gives a slight preference to infected.)
infection_threshold = 0.35;
% Define the window sizes of interest: 12 and 24 blocks.
window_sizes = [12, 24];

%% Data Preparation and Feature Selection
data = cleaneddatamad;

% Use the "batch" column for ordering (to account for inter-device variability)
batches = data.Batch;

% Extract original features.
% We remove the ratio features (R/C, G/C, B/C) because CPerc is sometimes close to 0.
epsilon = 1e-6;
R = data.RPercNormalized;
G = data.GPercNormalized;
B = data.BPercNormalized;
C = data.CPercNormalized;

% Use only the raw features R, G, B, and C.
features_orig = [R, G, B, C];
feature_names = {'R','G','B','C'};
features = features_orig;  % Final feature matrix

% Extract binary labels and patient IDs
binary_labels = data.infClassIDSA;
patient_ids = data.InStudyID;

% Remove rows with NaN or Inf values
valid_rows = ~any(isnan(features) | isinf(features), 2);
features = features(valid_rows, :);
binary_labels = binary_labels(valid_rows);
patient_ids = patient_ids(valid_rows);
batches = batches(valid_rows);

%% Data Balancing with SMOTE Oversampling
fprintf('\nBalancing the Dataset...\n');
if useSMOTE
    k_neighbors = 5;  % Number of neighbors for SMOTE (default)
    [features, binary_labels, patient_ids, batches] = smote_oversample(features, binary_labels, patient_ids, batches, k_neighbors);
else
    % Random oversampling (fallback)
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
end
fprintf('Final Balanced Dataset: Class 0 = %d, Class 1 = %d\n', sum(binary_labels==0), sum(binary_labels==1));

%% Leave-One-Patient-Out Cross-Validation with GentleBoost
results = struct();
result_idx = 1;
unique_patients = unique(patient_ids);
n_patients = length(unique_patients);

for w = 1:length(window_sizes)
    window_size = window_sizes(w);
    fprintf('\nPerforming LOOCV with Window Size: %d\n', window_size);
    
    % Create rolling windows with purity flag (only windows with homogeneous labels are used for testing)
    [windowed_features, windowed_labels, windowed_patient_ids, ~, windowed_is_pure] = ...
        create_rolling_window_per_patient(features, binary_labels, patient_ids, batches, window_size);
    
    all_test_preds = [];
    all_test_truth = [];
    
    for i = 1:n_patients
        fprintf('Processing patient %d/%d\n', i, n_patients);
        test_idx = (windowed_patient_ids == unique_patients(i));
        % Use only pure windows for testing
        pure_test_idx = test_idx & windowed_is_pure;
        if sum(pure_test_idx) == 0
            fprintf('Patient %d has no pure windows. Skipping testing for this patient.\n', unique_patients(i));
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
        
        % Train GentleBoost classifier with hyperparameter tuning
        if performHyperparameterTuning
            model = fitcensemble(X_train_scaled, y_train, 'Method', model_method, ...
                'OptimizeHyperparameters', 'all', ...
                'HyperparameterOptimizationOptions', struct('ShowPlots', false, 'Verbose', 0));
        else
            model = fitcensemble(X_train_scaled, y_train, 'Method', model_method);
        end
        
        % Obtain predicted probabilities and adjust decision threshold
        [~, scores] = predict(model, X_test_scaled);
        % Assume column 2 corresponds to probability for infected (class 1)
        preds_adjusted = double(scores(:,2) >= infection_threshold);
        
        all_test_preds = [all_test_preds; preds_adjusted];
        all_test_truth = [all_test_truth; y_test];
    end
    
    % Evaluate performance on the aggregated test set
    binary_cm = confusionmat(all_test_truth, all_test_preds);
    [precision, recall, f1] = calculate_metrics(binary_cm);
    accuracy = sum(diag(binary_cm)) / sum(binary_cm(:));
    
    % Compute additional metrics
    TN = binary_cm(1,1); FP = binary_cm(1,2);
    FN = binary_cm(2,1); TP = binary_cm(2,2);
    sensitivity = TP / max((TP + FN), 1);
    specificity = TN / max((TN + FP), 1);
    FP_rate = FP / max((TN + FP), 1);
    FN_rate = FN / max((TP + FN), 1);
    
    results(result_idx).model = model_method;
    results(result_idx).window_size = window_size;
    results(result_idx).accuracy = accuracy;
    results(result_idx).f1 = f1;
    results(result_idx).confusion_matrix = binary_cm;
    results(result_idx).sensitivity = sensitivity;
    results(result_idx).specificity = specificity;
    results(result_idx).FP_rate = FP_rate;
    results(result_idx).FN_rate = FN_rate;
    
    fprintf('GentleBoost, Window Size: %d | Accuracy: %.2f | F1 (Clean): %.4f | F1 (Infected): %.4f | Sensitivity: %.4f | Specificity: %.4f | FP Rate: %.4f | FN Rate: %.4f\n', ...
        window_size, accuracy, f1(1), f1(2), sensitivity, specificity, FP_rate, FN_rate);
    
    result_idx = result_idx + 1;
end

% Export aggregated performance metrics for later review
metricsTable = struct2table(results);
writetable(metricsTable, 'gentleboost_performance_metrics.csv');
fprintf('\nPerformance metrics exported to gentleboost_performance_metrics.csv\n');

%% Diagnostic Analysis: Variable Importance
% We run diagnostic analysis (using GentleBoost with Bagging method internally)
chosen_window_size = 24;  % You may change this as desired
analyze_model_behavior(features, binary_labels, patient_ids, batches, chosen_window_size, feature_names);

%% Patient Visualization
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
            % Label by union rule: if any block in the window is infected, label as infected.
            if any(curr_labels(j:j+window_size-1) == 1)
                label = 1;
            else
                label = 0;
            end
            % Check if the window is pure (all blocks have the same label)
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

function analyze_model_behavior(features, binary_labels, patient_ids, batches, window_size, feature_names)
    [windowed_features, windowed_labels, ~, ~, ~] = create_rolling_window_per_patient(features, binary_labels, patient_ids, batches, window_size);
    [windowed_features_scaled, ~, ~] = zscore(windowed_features);
    % Train a GentleBoost model (using Bagging internally for stability)
    model = fitcensemble(windowed_features_scaled, windowed_labels, 'Method', 'Bag');
    imp = predictorImportance(model);
    
    num_features = length(feature_names);
    imp_reshaped = reshape(imp, num_features, window_size);
    avg_importance = mean(imp_reshaped, 2);
    
    figure;
    heatmap(feature_names, cellstr("T-" + string(0:window_size-1)), imp_reshaped');
    title(sprintf('Variable Importance (Window Size: %d)', window_size));
    
    importance_table = table(feature_names(:), avg_importance, 'VariableNames', {'Feature', 'AverageImportance'});
    csv_filename = sprintf('variable_importance_window%d.csv', window_size);
    writetable(importance_table, csv_filename);
    fprintf('Variable importance exported to %s\n', csv_filename);
end

function visualize_patient_data(features, binary_labels, patient_ids)
    % We are now only using the raw features: R, G, B, and C.
    feature_names = {'R','G','B','C'};
    
    % 1. Boxplots: Feature Distributions by Infection Status
    figure('Position', [100 100 1200 800]);
    for i = 1:length(feature_names)
        subplot(4,2,i);
        clean_data = features(binary_labels == 0, i);
        infected_data = features(binary_labels == 1, i);
        boxplot([clean_data; infected_data], [zeros(size(clean_data)); ones(size(infected_data))], ...
                'Colors', 'br', 'Labels', {'Clean','Infected'});
        title(feature_names{i});
        grid on;
    end
    sgtitle('Feature Distributions by Infection Status');
    
    % 2. PCA Projection and Feature Correlation
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
    xticks(1:length(feature_names));
    yticks(1:length(feature_names));
    xticklabels(feature_names);
    yticklabels(feature_names);
    
    % 3. Patient Trajectory Visualization with Infected Areas Highlighted
    figure('Position', [100 100 1200 800]);
    unique_patients = unique(patient_ids);
    nPatients = length(unique_patients);
    
    for p = 1:nPatients
        subplot(ceil(nPatients/2), 2, p);
        idx = (patient_ids == unique_patients(p));
        patient_data = features(idx, :);
        patient_labels = binary_labels(idx);
        num_timepoints = size(patient_data, 1);
        
        % Plot the patient trajectories for each feature
        h = plot(patient_data, 'LineWidth', 1.5);
        hold on;
        
        % Get the current y-axis limits (so patches span the entire vertical range)
        yl = ylim;
        
        % For each time point, if the sample is infected, add a red patch
        for t = 1:num_timepoints
            if patient_labels(t) == 1
                % Create a patch for time point t (with a half-index padding)
                patch([t-0.5, t+0.5, t+0.5, t-0.5], [yl(1) yl(1) yl(2) yl(2)], ...
                      'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
            end
        end
        hold off;
        
        title(sprintf('Patient %d', unique_patients(p)));
        legend(h, feature_names, 'Location', 'eastoutside');
        grid on;
    end
    sgtitle('Patient Trajectories with Infected Regions Highlighted');
end


function [features_new, labels_new, patient_ids_new, batches_new] = smote_oversample(features, labels, patient_ids, batches, k)
    % SMOTE oversampling as described in Chawla et al. (2002)
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
