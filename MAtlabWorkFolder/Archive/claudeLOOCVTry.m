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
        
        if (tp + fp) == 0
            precision(i) = 0;
        else
            precision(i) = tp / (tp + fp);
        end
        
        if (tp + fn) == 0
            recall(i) = 0;
        else
            recall(i) = tp / (tp + fn);
        end
        
        if (precision(i) + recall(i)) == 0
            f1(i) = 0;
        else
            f1(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
        end
    end
end

% Confusion matrix plotting function
function plot_confusion_matrix(cm, classes, title_text)
    figure('Position', [100, 100, 600, 500]);
    imagesc(cm);
    colormap('winter');
    colorbar;
    
    % Add numbers to cells
    [r,c] = size(cm);
    for i = 1:r
        for j = 1:c
            text(j, i, num2str(cm(i,j)), 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', 'FontSize', 12, 'FontWeight', 'bold');
        end
    end
    
    title(title_text, 'FontSize', 14);
    xlabel('Predicted Class');
    ylabel('True Class');
    
    % Add class labels
    xticks(1:length(classes));
    yticks(1:length(classes));
    xticklabels(classes);
    yticklabels(classes);
    xtickangle(45);
    
    % Add percentage text
    for i = 1:r
        for j = 1:c
            percentage = 100 * cm(i,j) / sum(cm(:));
            text(j, i-0.2, sprintf('%.1f%%', percentage), ...
                'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', 'black');
        end
    end
end

% Main analysis code
clc;
clearvars -except BOSoMetreDataTrimmedDataForTraining

if ~exist('BOSoMetreDataTrimmedDataForTraining', 'var')
    error('BOSoMetreDataTrimmedDataForTraining is not found in the workspace.');
end
data = BOSoMetreDataTrimmedDataForTraining;

%% Exclude Specific Patients
excludePatients = [9, 17, 18, 19];
data = data(~ismember(data.InStudyID, excludePatients), :);

% Handle missing values first
valid_idx = ~isnan(data.infClassIDSA);
data = data(valid_idx, :);

% Parse datetime with correct format
timestamps = datetime(data.DateTime, 'InputFormat', 'yyMMdd_HHmm');

% Extract and prepare base features
base_features = [data.RPercNormalized, data.GPercNormalized, data.BPercNormalized, data.CPercNormalized];
base_features = winsorize_features(base_features, 1, 99);

% Add engineered features
features = [base_features, ...
           base_features(:,1)./base_features(:,2), ...  % R/G ratio
           base_features(:,1)./base_features(:,3), ...  % R/B ratio
           base_features(:,2)./base_features(:,3), ...  % G/B ratio
           base_features(:,1)./base_features(:,4), ...  % R/C ratio
           base_features(:,2)./base_features(:,4), ...  % G/C ratio
           base_features(:,3)./base_features(:,4)];     % B/C ratio

% Add time-based features
temporal_features = [
    hour(timestamps), ...        % Hour of day
    day(timestamps), ...         % Day of month
    month(timestamps), ...       % Month
    weekday(timestamps)         % Day of week
];

% Calculate rolling statistics for each patient
unique_patients = unique(data.InStudyID);
window_size = 5;  % 5-point rolling window
rolling_features = zeros(size(base_features));
rate_of_change = zeros(size(base_features));
acceleration = zeros(size(base_features));

for i = 1:length(unique_patients)
    patient_idx = data.InStudyID == unique_patients(i);
    patient_data = base_features(patient_idx, :);
    patient_times = timestamps(patient_idx);
    
    % Sort data by time for each patient
    [sorted_times, sort_idx] = sort(patient_times);
    patient_data = patient_data(sort_idx, :);
    
    for j = 1:size(patient_data, 2)
        % Calculate rolling mean
        rolling_features(patient_idx, j) = movmean(patient_data(:, j), window_size, 'omitnan');
        
        % Calculate rate of change (first derivative)
        rate_of_change(patient_idx, j) = [0; diff(patient_data(:, j))];
        
        % Calculate acceleration (second derivative)
        acceleration(patient_idx, j) = [0; diff(rate_of_change(patient_idx, j))];
    end
end

% Combine all features
features = [features, temporal_features, rolling_features, rate_of_change, acceleration];

% Remove any infinite values and handle NaN
features(isinf(features)) = NaN;
features = fillmissing(features, 'nearest');  % Fill remaining NaN values

binary_labels = data.infClassIDSA;
tri_labels = data.triClassIDSA;
patient_ids = data.InStudyID;

% Define classifiers
classifiers = {
    'Simple SVM', ...
    'RBF SVM', ...
    'LDA', ...
    'Simple KNN', ...
    'Subspace KNN', ...
    'Ensemble KNN'
};

n_classifiers = length(classifiers);
k = 10;  % number of neighbors for KNN

% Initialize storage
binary_predictions = zeros(length(binary_labels), n_classifiers);
tri_predictions = zeros(length(tri_labels), n_classifiers);

% Process classifications
for classification_type = 1:2
    if classification_type == 1
        fprintf('\nProcessing Binary Classification\n');
        labels_use = binary_labels;
        patient_ids_use = patient_ids;
        unique_patients_use = unique(patient_ids);
        predictions = binary_predictions;
    else
        fprintf('\nProcessing Triphasic Classification\n');
        labels_use = tri_labels;
        patient_ids_use = patient_ids;
        unique_patients_use = unique(patient_ids);
        predictions = tri_predictions;
    end
    
    % LOOCV
    for i = 1:length(unique_patients_use)
        fprintf('Processing patient %d of %d\n', i, length(unique_patients_use));
        
        % Split data
        test_idx = patient_ids_use == unique_patients_use(i);
        train_idx = ~test_idx;
        
        X_train = features(train_idx,:);
        X_test = features(test_idx,:);
        y_train = labels_use(train_idx);
        
        % Standardize features
        [X_train_scaled, mu, sigma] = zscore(X_train);
        X_test_scaled = (X_test - mu) ./ sigma;
        
        % Train and predict with each classifier
        for j = 1:n_classifiers
            try
                switch classifiers{j}
                    case 'Simple SVM'
                        if classification_type == 1
                            model = fitclinear(X_train_scaled, y_train, ...
                                'Learner', 'svm', ...
                                'Regularization', 'ridge');
                        else
                            model = fitcecoc(X_train_scaled, y_train, ...
                                'Learner', 'linear');
                        end
                            
                    case 'RBF SVM'
                        if classification_type == 1
                            model = fitcsvm(X_train_scaled, y_train, ...
                                'KernelFunction', 'rbf', ...
                                'KernelScale', 'auto', ...
                                'Standardize', true);
                        else
                            model = fitcecoc(X_train_scaled, y_train, ...
                                'Learner', templateSVM('KernelFunction', 'rbf'));
                        end
                            
                    case 'LDA'
                        model = fitcdiscr(X_train_scaled, y_train, ...
                            'DiscrimType', 'linear');
                            
                    case 'Simple KNN'
                        model = fitcknn(X_train_scaled, y_train, ...
                            'NumNeighbors', k, ...
                            'Distance', 'euclidean');
                            
                    case 'Subspace KNN'
                        model = fitcensemble(X_train_scaled, y_train, ...
                            'Method', 'Subspace', ...
                            'NumLearningCycles', 60, ...
                            'Learners', templateKNN('NumNeighbors', 3));
                            
                    case 'Ensemble KNN'
                        % Combine multiple KNN models
                        knn1 = fitcknn(X_train_scaled, y_train, 'NumNeighbors', k, 'Distance', 'euclidean');
                        knn2 = fitcknn(X_train_scaled, y_train, 'NumNeighbors', k, 'Distance', 'cityblock');
                        knn3 = fitcknn(X_train_scaled, y_train, 'NumNeighbors', k, 'Distance', 'cosine');
                        
                        pred1 = predict(knn1, X_test_scaled);
                        pred2 = predict(knn2, X_test_scaled);
                        pred3 = predict(knn3, X_test_scaled);
                        
                        ensemble_preds = mode([pred1, pred2, pred3], 2);
                        predictions(test_idx, j) = ensemble_preds;
                        continue;
                end
                
                % Predict for non-ensemble methods
                if ~strcmp(classifiers{j}, 'Ensemble KNN')
                    predictions(test_idx, j) = predict(model, X_test_scaled);
                end
                
            catch ME
                fprintf('Classification failed for %s: %s\n', classifiers{j}, ME.message);
                predictions(test_idx, j) = mode(y_train);
            end
        end
    end
    
    % Store results
    if classification_type == 1
        binary_predictions = predictions;
    else
        tri_predictions = predictions;
    end
end

% Evaluate results
for j = 1:n_classifiers
    fprintf('\n\n=== Results for %s ===\n', classifiers{j});
    
    % Binary Classification Results
    fprintf('\nBinary Classification Results:\n');
    binary_cm = confusionmat(binary_labels, binary_predictions(:,j));
    binary_accuracy = sum(diag(binary_cm))/sum(binary_cm(:));
    [binary_precision, binary_recall, binary_f1] = calculate_metrics(binary_cm);
    
    fprintf('Accuracy: %.4f\n', binary_accuracy);
    fprintf('Class-wise metrics:\n');
    fprintf('Precision: %.4f, %.4f\n', binary_precision);
    fprintf('Recall: %.4f, %.4f\n', binary_recall);
    fprintf('F1 Score: %.4f, %.4f\n', binary_f1);
    
    % Plot binary confusion matrix
    plot_confusion_matrix(binary_cm, {'Clean', 'Infected'}, ...
        sprintf('Binary Confusion Matrix - %s', classifiers{j}));
    
    % Triphasic Classification Results
    fprintf('\nTriphasic Classification Results:\n');
    tri_cm = confusionmat(tri_labels, tri_predictions(:,j));
    tri_accuracy = sum(diag(tri_cm))/sum(tri_cm(:));
    [tri_precision, tri_recall, tri_f1] = calculate_metrics(tri_cm);
    
    fprintf('Accuracy: %.4f\n', tri_accuracy);
    fprintf('Class-wise metrics:\n');
    if length(tri_precision) >= 3
        fprintf('Precision: %.4f, %.4f, %.4f\n', tri_precision(1), tri_precision(2), tri_precision(3));
        fprintf('Recall: %.4f, %.4f, %.4f\n', tri_recall(1), tri_recall(2), tri_recall(3));
        fprintf('F1 Score: %.4f, %.4f, %.4f\n', tri_f1(1), tri_f1(2), tri_f1(3));
    else
        fprintf('Precision: %.4f, %.4f\n', tri_precision(1), tri_precision(2));
        fprintf('Recall: %.4f, %.4f\n', tri_recall(1), tri_recall(2));
        fprintf('F1 Score: %.4f, %.4f\n', tri_f1(1), tri_f1(2));
    end
    
    % Plot triphasic confusion matrix
    plot_confusion_matrix(tri_cm, {'Clean', 'Infected', 'Unsure'}, ...
        sprintf('Triphasic Confusion Matrix - %s', classifiers{j}));
end

% Print class distribution
fprintf('\nClass Distribution:\n');
fprintf('Binary Classification:\n');
binary_dist = histcounts(binary_labels, 'BinMethod', 'integers');
fprintf('Class 0 (Clean): %d (%.1f%%)\n', binary_dist(1), 100*binary_dist(1)/sum(binary_dist));
fprintf('Class 1 (Infected): %d (%.1f%%)\n', binary_dist(2), 100*binary_dist(2)/sum(binary_dist));

fprintf('\nTriphasic Classification:\n');
tri_dist = histcounts(tri_labels, 'BinMethod', 'integers');
for i = 1:length(tri_dist)
    switch i
        case 1
            label = 'Clean';
        case 2
            label = 'Infected';
        case 3
            label = 'Unsure';
    end
    fprintf('Class %d (%s): %d (%.1f%%)\n', i-1, label, tri_dist(i), 100*tri_dist(i)/sum(tri_dist));
end