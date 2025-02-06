% =========================================================================
% CNN-Based Binary Classification using Rolling Window + LOOCV
% Optimized for Low Computational Cost (Deployable on Raspberry Pi)
% =========================================================================

clc;
clearvars -except cleaneddata;

if ~exist('cleaneddata', 'var')
    error('cleaneddata is not found in the workspace.');
end

data = cleaneddata;

% =========================================================================
% Define Rolling Window Size & Other Parameters
% =========================================================================
window_size = 12;  % Adjust to 24 for longer-term pattern detection
winsor_level = [1, 99];  % Winsorization at 1-99 percentile

% =========================================================================
% Preprocess Data: Extract Features, Apply Winsorization, Create Rolling Windows
% =========================================================================
fprintf('\nPreprocessing Data...\n');

% Extract Features & Labels
features = [data.RPercNormalized, data.GPercNormalized, data.BPercNormalized, data.CPercNormalized];
binary_labels = data.infClassIDSA;
patient_ids = data.InStudyID;

% Apply Winsorization
features = winsorize_features(features, winsor_level(1), winsor_level(2));

% Convert to Rolling Windows
[X, Y, PatientIDs] = create_rolling_window(features, binary_labels, patient_ids, window_size);

% Standardize Features (Z-score Normalization)
X = normalize(X, 2);

% Reshape for CNN Input: [Samples, Time Steps, Features]
X = permute(X, [1, 3, 2]);  

% Convert Labels to Categorical
Y = categorical(Y);

% Get Unique Patient IDs for LOOCV
unique_patients = unique(PatientIDs);
n_patients = length(unique_patients);

% =========================================================================
% Define Optimized CNN Model
% =========================================================================
fprintf('\nDefining CNN Model...\n');

layers = [
    sequenceInputLayer(window_size, 'MinLength', window_size)  % Ensure input is properly handled
    convolution1dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(1, 'Stride', 1)  % Minimized pooling to prevent sequence shortening
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false);

% =========================================================================
% Perform LOOCV (Leave-One-Patient-Out Cross-Validation)
% =========================================================================
fprintf('\nPerforming LOOCV...\n');
predictions = categorical(zeros(size(Y)));

for i = 1:n_patients
    fprintf('Processing patient %d of %d\n', i, n_patients);
    test_idx = PatientIDs == unique_patients(i);
    train_idx = ~test_idx;
    
    X_train = X(train_idx, :, :);
    Y_train = Y(train_idx);
    X_test = X(test_idx, :, :);
    
    % Debugging: Check Input Dimensions Before Training
    fprintf('Training Data Size: %s\n', mat2str(size(X_train)));

    % Train CNN Model
    net = trainNetwork(X_train, Y_train, layers, options);
    
    % Predict on Test Data
    Y_pred = classify(net, X_test);
    predictions(test_idx) = Y_pred;
end

% =========================================================================
% Evaluate Model Performance
% =========================================================================
fprintf('\nEvaluating Model Performance...\n');
conf_mat = confusionmat(Y, predictions);
[precision, recall, f1] = calculate_metrics(conf_mat);
accuracy = sum(diag(conf_mat)) / sum(conf_mat(:));

fprintf('\nCNN LOOCV Results (Rolling Window = %d)\n', window_size);
fprintf('Accuracy: %.4f\n', accuracy);
fprintf('Precision (Class 0, Class 1): %.4f, %.4f\n', precision(1), precision(2));
fprintf('Recall (Class 0, Class 1): %.4f, %.4f\n', recall(1), recall(2));
fprintf('F1 Score (Class 0, Class 1): %.4f, %.4f\n', f1(1), f1(2));

% =========================================================================
% Plot Confusion Matrix
% =========================================================================
figure;
confusionchart(conf_mat, {'Clean', 'Infected'}, 'Title', 'CNN Classification');

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
