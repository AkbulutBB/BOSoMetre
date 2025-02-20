% CSF Analysis with Dynamic Class Balancing
clc; clearvars -except BOSoMetreDataCSFDataForTraining

% Data Loading and Validation
if ~exist('BOSoMetreDataCSFDataForTraining', 'var')
    error('Dataset not found. Load BOSoMetreDataCSFDataForTraining first.');
end

data = BOSoMetreDataCSFDataForTraining;

%% Data Preparation
% Exclude patients
excludePatients = [9, 17, 18, 19];
data = data(~ismember(data.InStudyID, excludePatients), :);

% Voltage compensation
nominalVoltage = 2.52;
voltages = str2double(data.VoltageAverage);
voltages(isnan(voltages) | voltages <= 0) = nominalVoltage;
voltageCompensation = nominalVoltage ./ voltages;

% Create compensated features
RPerc_comp = data.RPerc .* voltageCompensation;
GPerc_comp = data.GPerc .* voltageCompensation;
BPerc_comp = data.BPerc .* voltageCompensation;
CPerc_comp = data.CPerc .* voltageCompensation;

% Feature engineering
features = [RPerc_comp, GPerc_comp, BPerc_comp, CPerc_comp, ...
    RPerc_comp./(GPerc_comp + eps), ...
    BPerc_comp./(CPerc_comp + eps)];

binary_labels = data.infClassIDSA;
patient_ids = data.InStudyID;

% Clean data
valid_idx = all(~isnan(features) & ~isinf(features), 2) & ~isnan(binary_labels);
features = features(valid_idx, :);
binary_labels = double(binary_labels(valid_idx)); % Ensure double type
patient_ids = patient_ids(valid_idx);

% Standardize features
features = zscore(features);

%% Model Configuration
classifiers = {
    struct('name','SVM-Lin', 'fun',@fitcsvm, 'params',{{'KernelFunction','linear'}}), 
    struct('name','SVM-RBF', 'fun',@fitcsvm, 'params',{{'KernelFunction','rbf'}}),
    struct('name','GBM', 'fun',@fitcensemble, 'params',{{'Method','GentleBoost','NumLearningCycles',100}}),
    struct('name','RF', 'fun',@(X,y,w)TreeBagger(150,X,y,'Method','classification','MinLeafSize',5,'Weights',w), 'params',{}),
    struct('name','XGB', 'fun',@fitcensemble, 'params',{{'Method','LogitBoost','NumLearningCycles',100}})
};

n_classifiers = length(classifiers);
all_predictions = zeros(length(binary_labels), n_classifiers);

%% LOOCV Implementation with Per-Fold Balancing
for j = 1:n_classifiers
    fprintf('\nProcessing %s\n', classifiers{j}.name);
    predictions = zeros(size(binary_labels));
    
    for i = 1:length(unique(patient_ids))
        fprintf('Patient %d of %d\n', i, length(unique(patient_ids)));
        
        % Patient-wise split
        test_idx = (patient_ids == unique(patient_ids(i)));
        train_idx = ~test_idx;
        
        X_train = features(train_idx, :);
        X_test = features(test_idx, :);
        y_train = binary_labels(train_idx);
        
        try
            % Calculate class weights for current fold
            [class_counts, ~] = histcounts(y_train);
            sample_weights = ones(size(y_train));
            sample_weights(y_train == 0) = 1/(class_counts(1)/numel(y_train));
            sample_weights(y_train == 1) = 1/(class_counts(2)/numel(y_train));
            
            % Train model with weights
            if contains(classifiers{j}.name, {'RF', 'SVM'})
                model = classifiers{j}.fun(X_train, y_train, sample_weights, classifiers{j}.params{:});
            else
                model = classifiers{j}.fun(X_train, y_train, classifiers{j}.params{:}, 'Weights', sample_weights);
            end
            
            % Generate predictions
            if strcmp(classifiers{j}.name, 'RF')
                pred = str2double(predict(model, X_test));
            else
                pred = predict(model, X_test);
            end
            
            % Convert to double and validate size
            pred = double(pred);
            if numel(pred) ~= sum(test_idx)
                error('Prediction size mismatch');
            end
            
            predictions(test_idx) = pred;
            
        catch ME
            fprintf('Error: %s\n', ME.message);
            predictions(test_idx) = mode(y_train);
        end
    end
    
    all_predictions(:, j) = predictions;
    
    % Evaluation
    fprintf('\n=== Results for %s ===\n', classifiers{j}.name);
    cm = confusionmat(binary_labels, predictions);
    [precision, recall, f1] = calculate_metrics(cm);
    
    fprintf('Class 0 Recall: %.4f\n', recall(1));
    fprintf('Class 1 Precision: %.4f\n', precision(2));
    fprintf('Confusion Matrix:\n');
    disp(cm);
end

%% Ensemble with Type-Safe Predictions
ensemble_preds = double(mode(double(all_predictions), 2));

% Final Evaluation
fprintf('\n=== Ensemble Results ===\n');
cm = confusionmat(binary_labels, ensemble_preds);
[precision, recall, f1] = calculate_metrics(cm);
fprintf('Class 0 Recall: %.4f\n', recall(1));
fprintf('Class 1 Precision: %.4f\n', precision(2));
fprintf('Confusion Matrix:\n');
disp(cm);

% Helper function
function [precision, recall, f1] = calculate_metrics(cm)
    precision = diag(cm) ./ (sum(cm, 1)' + eps);
    recall = diag(cm) ./ (sum(cm, 2) + eps);
    f1 = 2 * (precision .* recall) ./ (precision + recall + eps);
    precision(isnan(precision)) = 0;
    recall(isnan(recall)) = 0;
    f1(isnan(f1)) = 0;
end