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

% ... Previous helper functions remain the same ...

% Main analysis code
clc;
clearvars -except BOSoMetreDataTrimmedDataForTraining

if ~exist('BOSoMetreDataTrimmedDataForTraining', 'var')
    error('BOSoMetreDataTrimmedDataForTraining is not found in the workspace.');
end
data = BOSoMetreDataTrimmedDataForTraining;

% Define classifiers
classifiers = {
    'Simple SVM', ...
    'RBF SVM', ...
    'LDA', ...
    'Simple KNN', ...
    'Subspace KNN', ...
    'Ensemble KNN'
};

% Define winsorization levels
winsor_levels = [
    [1, 99];   % 1-99 percentiles
];

% Store results for each winsorization level
results = struct();

for w = 1:size(winsor_levels, 1)
    fprintf('\nTesting winsorization level %d-%d percentiles\n', ...
        winsor_levels(w,1), winsor_levels(w,2));
    
    % Extract and prepare features with current winsorization level
    features = [data.RPercNormalized, data.GPercNormalized, data.BPercNormalized, data.CPercNormalized];
    features = winsorize_features(features, winsor_levels(w,1), winsor_levels(w,2));
    
    % Remove any infinite values and handle NaN
    features(isinf(features)) = NaN;
    features = fillmissing(features, 'nearest');
    
    binary_labels = data.infClassIDSA;
    tri_labels = data.triClassIDSA;
    patient_ids = data.InStudyID;
    
    % Get unique patient IDs
    unique_patients = unique(patient_ids);
    n_patients = length(unique_patients);
    
    % Initialize storage for each classifier
    n_classifiers = length(classifiers);
    binary_predictions = zeros(length(binary_labels), n_classifiers);
    tri_predictions = zeros(length(tri_labels), n_classifiers);
    
    % Process classifications
    for classification_type = 1:2
        if classification_type == 1
            fprintf('\nProcessing Binary Classification\n');
            labels_use = binary_labels;
            predictions = binary_predictions;
        else
            fprintf('\nProcessing Triphasic Classification\n');
            labels_use = tri_labels;
            predictions = tri_predictions;
        end
        
        % LOOCV
        for i = 1:n_patients
            fprintf('Processing patient %d of %d\n', i, n_patients);
            
            % Split data
            test_idx = patient_ids == unique_patients(i);
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
                                'NumNeighbors', 10, ...
                                'Distance', 'euclidean');
                                
                        case 'Subspace KNN'
                            model = fitcensemble(X_train_scaled, y_train, ...
                                'Method', 'Subspace', ...
                                'NumLearningCycles', 60, ...
                                'Learners', templateKNN('NumNeighbors', 3));
                                
                        case 'Ensemble KNN'
                            % Combine multiple KNN models
                            knn1 = fitcknn(X_train_scaled, y_train, 'NumNeighbors', 10, 'Distance', 'euclidean');
                            knn2 = fitcknn(X_train_scaled, y_train, 'NumNeighbors', 10, 'Distance', 'cityblock');
                            knn3 = fitcknn(X_train_scaled, y_train, 'NumNeighbors', 10, 'Distance', 'cosine');
                            
                            pred1 = predict(knn1, X_test_scaled);
                            pred2 = predict(knn2, X_test_scaled);
                            pred3 = predict(knn3, X_test_scaled);
                            
                            predictions(test_idx, j) = mode([pred1, pred2, pred3], 2);
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
    
    % Store results for this winsorization level
    results(w).percentiles = winsor_levels(w,:);
    results(w).binary_predictions = binary_predictions;
    results(w).tri_predictions = tri_predictions;
end

% Display comparative results
fprintf('\n=== Comparative Results for Different Winsorization Levels ===\n');

for w = 1:length(results)
    fprintf('\n\nResults for %d-%d percentile winsorization:\n', ...
        results(w).percentiles(1), results(w).percentiles(2));
    
    for j = 1:length(classifiers)
        fprintf('\n=== %s ===\n', classifiers{j});
        
        % Binary Classification Results
        binary_cm = confusionmat(binary_labels, results(w).binary_predictions(:,j));
        [precision, recall, f1] = calculate_metrics(binary_cm);
        accuracy = sum(diag(binary_cm))/sum(binary_cm(:));
        
        fprintf('Binary Classification:\n');
        fprintf('Accuracy: %.4f\n', accuracy);
        fprintf('Precision: %.4f, %.4f\n', precision);
        fprintf('Recall: %.4f, %.4f\n', recall);
        fprintf('F1 Score: %.4f, %.4f\n', f1);
        
        % Plot confusion matrix
        plot_confusion_matrix(binary_cm, {'Clean', 'Infected'}, ...
            sprintf('%s - Binary (%d-%d percentiles)', ...
            classifiers{j}, results(w).percentiles(1), results(w).percentiles(2)));
        
        % Triphasic Classification Results
        tri_cm = confusionmat(tri_labels, results(w).tri_predictions(:,j));
        [precision, recall, f1] = calculate_metrics(tri_cm);
        accuracy = sum(diag(tri_cm))/sum(tri_cm(:));
        
        fprintf('\nTriphasic Classification:\n');
        fprintf('Accuracy: %.4f\n', accuracy);
        if length(precision) >= 3
            fprintf('Precision: %.4f, %.4f, %.4f\n', precision(1), precision(2), precision(3));
            fprintf('Recall: %.4f, %.4f, %.4f\n', recall(1), recall(2), recall(3));
            fprintf('F1 Score: %.4f, %.4f, %.4f\n', f1(1), f1(2), f1(3));
        else
            fprintf('Precision: %.4f, %.4f\n', precision(1), precision(2));
            fprintf('Recall: %.4f, %.4f\n', recall(1), recall(2));
            fprintf('F1 Score: %.4f, %.4f\n', f1(1), f1(2));
        end
        
        % Plot confusion matrix
        plot_confusion_matrix(tri_cm, {'Clean', 'Infected', 'Unsure'}, ...
            sprintf('%s - Triphasic (%d-%d percentiles)', ...
            classifiers{j}, results(w).percentiles(1), results(w).percentiles(2)));
    end
end

% Print final class distribution
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