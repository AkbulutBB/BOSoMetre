% CSF Analysis with advanced models
clc;
clearvars -except BOSoMetreDataTrimmedDataForTraining

if ~exist('BOSoMetreDataTrimmedDataForTraining', 'var')
    error('BOSoMetreDataTrimmedDataForTraining is not found in the workspace.');
end
data = BOSoMetreDataTrimmedDataForTraining;

%% Exclude Specific Patients
excludePatients = [9, 17, 18, 19];
data = data(~ismember(data.InStudyID, excludePatients), :);

% Extract and prepare features with voltage compensation
nominalVoltage = 2.52;
voltages = str2double(data.VoltageAverage);
voltages(isnan(voltages) | voltages <= 0) = nominalVoltage;
voltageCompensation = nominalVoltage ./ voltages;

% Create voltage-compensated features
RPerc_comp = data.RPerc .* voltageCompensation;
GPerc_comp = data.GPerc .* voltageCompensation;
BPerc_comp = data.BPerc .* voltageCompensation;
CPerc_comp = data.CPerc .* voltageCompensation;

% Create channel ratios
RG_ratio = RPerc_comp ./ GPerc_comp;
RB_ratio = RPerc_comp ./ BPerc_comp;
GB_ratio = GPerc_comp ./ BPerc_comp;
RC_ratio = RPerc_comp ./ CPerc_comp;
GC_ratio = GPerc_comp ./ CPerc_comp;
BC_ratio = BPerc_comp ./ CPerc_comp;

% Combine features
features = [RPerc_comp, GPerc_comp, BPerc_comp, CPerc_comp];
% features = [RPerc_comp, GPerc_comp, BPerc_comp, CPerc_comp, ...
%            RG_ratio, RB_ratio, GB_ratio, RC_ratio, GC_ratio, BC_ratio];
binary_labels = data.infClassIDSA;
patient_ids = data.InStudyID;

% Remove invalid data
valid_idx = all(~isnan(features) & ~isinf(features), 2) & ~isnan(binary_labels);
features = features(valid_idx, :);
binary_labels = binary_labels(valid_idx);
patient_ids = patient_ids(valid_idx);

% Standardize features
features = zscore(features);

% Get unique patient IDs
unique_patients = unique(patient_ids);
n_patients = length(unique_patients);

% Define classifiers
classifiers = {
    'Linear SVM', ...
    'RBF SVM', ...
    'Deep Neural Network', ...
    'Gradient Boosting', ...
    'Random Forest', ...
    'Subspace KNN', ...
    'XGBoost Ensemble'
};

n_classifiers = length(classifiers);
all_predictions = zeros(length(binary_labels), n_classifiers);

% LOOCV for each classifier
for j = 1:n_classifiers
    fprintf('\nProcessing %s\n', classifiers{j});
    predictions = zeros(size(binary_labels));
    
    for i = 1:n_patients
        fprintf('Patient %d of %d\n', i, n_patients);
        
        % Split data
        test_idx = patient_ids == unique_patients(i);
        train_idx = ~test_idx;
        
        X_train = features(train_idx, :);
        X_test = features(test_idx, :);
        y_train = binary_labels(train_idx);
        
        try
            switch classifiers{j}
                case 'Linear SVM'
                    model = fitcecoc(X_train, y_train, 'Learners', templateSVM('Standardize', false, 'KernelFunction', 'linear'));
                    
                case 'RBF SVM'
                    model = fitcecoc(X_train, y_train, 'Learners', templateSVM('Standardize', false, 'KernelFunction', 'rbf', 'KernelScale', 'auto'));
                    
                case 'Deep Neural Network'
                    layers = [
                        featureInputLayer(size(X_train, 2))
                        fullyConnectedLayer(64)
                        batchNormalizationLayer
                        reluLayer
                        dropoutLayer(0.3)
                        fullyConnectedLayer(32)
                        batchNormalizationLayer
                        reluLayer
                        dropoutLayer(0.2)
                        fullyConnectedLayer(2)
                        softmaxLayer
                        classificationLayer];
                    
                    options = trainingOptions('adam', ...
                        'MaxEpochs', 50, ...
                        'MiniBatchSize', 32, ...
                        'Shuffle', 'every-epoch', ...
                        'Verbose', false);
                    
                    model = trainNetwork(X_train, categorical(y_train), layers, options);
                    
                case 'Gradient Boosting'
                    model = fitcensemble(X_train, y_train, 'Method', 'GentleBoost', ...
                        'NumLearningCycles', 100, 'LearnRate', 0.1);
                    
                case 'Random Forest'
                    model = TreeBagger(100, X_train, y_train, 'Method', 'classification', ...
                        'MinLeafSize', 5, 'NumPredictorsToSample', 'all');
                    
                case 'Subspace KNN'
                      model = fitcensemble(X_train, y_train, ...
                        'Method', 'Subspace', ...
                        'NumLearningCycles', 60, ... % 60 learners
                        'Learners', templateKNN('NumNeighbors', 10), ... % 10 neighbors
                        'NPredToSample', 3); % Use 3 features per learner
                      
                case 'XGBoost Ensemble'
                % For two-class problems, we cannot use 'AdaBoostM2'.
                % We can substitute AdaBoostM1 or GentleBoost, etc.
                model = fitcensemble(X_train, y_train, ...
                    'Method', 'AdaBoostM1', ...  % Replaced AdaBoostM2
                    'NumLearningCycles', 100, ...
                    'LearnRate', 0.1, ...
                    'Learners', templateTree('MaxNumSplits', 20));

            end
            
            % Make predictions
            if strcmp(classifiers{j}, 'Deep Neural Network')
                pred = predict(model, X_test);
                [~, pred] = max(pred, [], 2);
                pred = pred - 1;  % Convert to 0/1
            elseif strcmp(classifiers{j}, 'Random Forest')
                pred = str2double(predict(model, X_test));
            else
                pred = predict(model, X_test);
            end
            predictions(test_idx) = pred;
            
        catch ME
            fprintf('Error with %s: %s\n', classifiers{j}, ME.message);
            predictions(test_idx) = mode(y_train);
        end
    end
    
    all_predictions(:, j) = predictions;
    
    % Evaluate Results
    fprintf('\n=== Results for %s ===\n', classifiers{j});
    cm = confusionmat(binary_labels, predictions);
    accuracy = sum(diag(cm))/sum(cm(:));
    [precision, recall, f1] = calculate_metrics(cm);
    
    fprintf('Accuracy: %.4f\n', accuracy);
    fprintf('Confusion Matrix:\n');
    disp(cm);
    fprintf('Class-wise metrics:\n');
    fprintf('Precision: %.4f, %.4f\n', precision);
    fprintf('Recall: %.4f, %.4f\n', recall);
    fprintf('F1 Score: %.4f, %.4f\n', f1);
    
    % Plot ROC curve
    [X,Y,~,AUC] = perfcurve(binary_labels, predictions, 1);
    figure;
    plot(X,Y);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(sprintf('ROC Curve for %s (AUC = %.3f)', classifiers{j}, AUC));
end

% Create ensemble of all models
ensemble_predictions = mode(all_predictions, 2);
fprintf('\n=== Results for Ensemble of All Models ===\n');
cm = confusionmat(binary_labels, ensemble_predictions);
accuracy = sum(diag(cm))/sum(cm(:));
[precision, recall, f1] = calculate_metrics(cm);

fprintf('Accuracy: %.4f\n', accuracy);
fprintf('Confusion Matrix:\n');
disp(cm);
fprintf('Class-wise metrics:\n');
fprintf('Precision: %.4f, %.4f\n', precision);
fprintf('Recall: %.4f, %.4f\n', recall);
fprintf('F1 Score: %.4f, %.4f\n', f1);

% Helper function to calculate metrics
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