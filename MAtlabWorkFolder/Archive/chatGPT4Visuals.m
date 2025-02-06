% Step 1: Load the data
filePath = 'G:\My Drive\Neuropapers\BBA -BOSoMetre\MAtlabWorkFolder\LongBosometreData.csv'; % Replace with your file path
data = readtable(filePath);

% Display the variable names
disp('Variable Names:');
disp(data.Properties.VariableNames);

% Display the first few rows
disp('First few rows of the data:');
disp(head(data));

% Extract relevant data
timestamps = data.DurationInHours; % Time points
sensorData = [data.R, data.G, data.B, data.C]; % Sensor readings
labels = data.BinaryInfected; % Infection status
patientIDs = data.InStudyID; % Patient identifiers

% Convert labels to categorical if they are not already
labels = categorical(labels);

% Check data size
disp(['Total samples: ', num2str(length(timestamps))]);

% Feature Engineering
% Add derived features: Ratios and Total Intensity
colorIntensity = sum(sensorData(:, 1:3), 2); % Total intensity (R + G + B)
colorRatios = [sensorData(:, 1) ./ (sensorData(:, 2) + 1e-6), ... % R/G
               sensorData(:, 2) ./ (sensorData(:, 3) + 1e-6), ... % G/B
               sensorData(:, 3) ./ (sensorData(:, 1) + 1e-6)];   % B/R

% Combine all features: Individual channels, turbidity (C), and derived features
X = [sensorData, colorIntensity, colorRatios];

% Handle missing labels (if any)
hasLabel = ~ismissing(labels);
X = X(hasLabel, :);
y = labels(hasLabel);
timestamps = timestamps(hasLabel);
patientIDs = patientIDs(hasLabel);

% Normalize features
[X, mu, sigma] = zscore(X);

% Split data into training and testing using Leave-One-Patient-Out Cross-Validation
uniquePatients = unique(patientIDs);

% Initialize performance metrics
accuracies = [];
confusionMatrices = [];

% Collect predictions and true labels for overall performance
allTrueLabels = [];
allPredictedLabels = [];

for i = 1:length(uniquePatients)
    testPatient = uniquePatients(i);
    trainIdx = patientIDs ~= testPatient;
    testIdx = patientIDs == testPatient;
    
    X_train = X(trainIdx, :);
    y_train = y(trainIdx);
    X_test = X(testIdx, :);
    y_test = y(testIdx);
    
    % Train multiclass model
    model = fitcecoc(X_train, y_train);
    
    % Predict on test data
    y_pred = predict(model, X_test);
    
    % Evaluate performance
    accuracy = sum(y_pred == y_test) / length(y_test);
    accuracies = [accuracies; accuracy];
    
    % Confusion matrix
    classNames = categories(y);
    confMat = confusionmat(y_test, y_pred, 'Order', classNames);
    confusionMatrices(:, :, i) = confMat;
    
    % Collect for overall performance
    allTrueLabels = [allTrueLabels; y_test];
    allPredictedLabels = [allPredictedLabels; y_pred];
    
    % Display performance for this patient
    disp(['Patient ', char(testPatient), ' Accuracy: ', num2str(accuracy)]);
end

% Overall performance
meanAccuracy = mean(accuracies);
disp(['Leave-One-Patient-Out Mean Accuracy: ', num2str(meanAccuracy)]);

% Overall confusion matrix
overallConfMat = confusionmat(allTrueLabels, allPredictedLabels, 'Order', classNames);
disp('Overall Confusion Matrix:');
disp(array2table(overallConfMat, 'VariableNames', cellstr(classNames), 'RowNames', cellstr(classNames)));

% Compute overall metrics
numClasses = length(classNames);
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1Score = zeros(numClasses, 1);

for c = 1:numClasses
    tp = overallConfMat(c, c);
    fp = sum(overallConfMat(:, c)) - tp;
    fn = sum(overallConfMat(c, :)) - tp;
    precision(c) = tp / (tp + fp);
    recall(c) = tp / (tp + fn);
    f1Score(c) = 2 * (precision(c) * recall(c)) / (precision(c) + recall(c));
end

% Display metrics
for c = 1:numClasses
    disp(['Class ', char(classNames(c)), ':']);
    disp(['  Precision: ', num2str(precision(c))]);
    disp(['  Recall: ', num2str(recall(c))]);
    disp(['  F1-Score: ', num2str(f1Score(c))]);
end

% Plotting predictions vs actual labels per patient (optional)
% For each patient, plot timestamps vs actual and predicted labels

for i = 1:length(uniquePatients)
    patientID = uniquePatients(i);
    idx = patientIDs == patientID;
    patientTimestamps = timestamps(idx);
    patientLabels = y(idx);
    patientPredictions = allPredictedLabels(idx);
    
    figure;
    plot(patientTimestamps, double(patientLabels), 'bo-', 'DisplayName', 'Actual Labels');
    hold on;
    plot(patientTimestamps, double(patientPredictions), 'rx-', 'DisplayName', 'Predicted Labels');
    xlabel('Duration in Hours');
    ylabel('Label (Numeric)');
    title(['Patient ', char(patientID)]);
    legend;
    hold off;
end

% Optional: Feature importance using predictor importance estimates
% This requires Statistics and Machine Learning Toolbox

% Since fitcecoc doesn't directly support predictor importance,
% we can train a bagged tree ensemble for this purpose
modelTree = TreeBagger(100, X, y, 'OOBPredictorImportance', 'on', 'Method', 'classification');

% Get importance scores
importance = modelTree.OOBPermutedPredictorDeltaError;

% Plot feature importance
figure;
bar(importance);
xlabel('Feature Index');
ylabel('Importance');
title('Feature Importance');
xticklabels({'R', 'G', 'B', 'C', 'Intensity', 'R/G', 'G/B', 'B/R'});
xtickangle(45);

% Advanced Model (Optional): Using LSTM for time-series data
% This section requires Deep Learning Toolbox

% Prepare data for LSTM
% Reshape data into sequences per patient
sequences = {};
sequenceLabels = {};

for i = 1:length(uniquePatients)
    idx = patientIDs == uniquePatients(i);
    % Extract the sequence of features for the patient
    sequences{end+1} = X(idx, :)'; % Size: [Features x TimeSteps]
    % Extract the sequence of labels for the patient
    labels_i = y(idx)'; % Transpose to make it a row vector
    % Ensure labels are categorical and match the time steps
    sequenceLabels{end+1} = labels_i;
end

% Convert labels in sequenceLabels to categorical row vectors
for i = 1:length(sequenceLabels)
    sequenceLabels{i} = categorical(sequenceLabels{i});
end

% Verify that sequences and labels have matching lengths and correct orientation
for i = 1:length(sequenceLabels)
    seqLength = size(sequences{i}, 2); % Number of time steps
    labelLength = length(sequenceLabels{i}); % Should be equal to seqLength
    if seqLength ~= labelLength
        error(['Mismatch in sequence and label length for sequence ', num2str(i)]);
    end
    % Check if labels are row vectors
    if size(sequenceLabels{i}, 1) ~= 1
        error(['Labels for sequence ', num2str(i), ' are not row vectors']);
    end
end

% Define LSTM network architecture with OutputMode 'sequence'
inputSize = size(X, 2);
numHiddenUnits = 50;
numClasses = numel(classNames);

layers = [
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

% Set training options
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 1, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', 0, ...
    'Plots', 'training-progress');

% Train LSTM network
net = trainNetwork(sequences, sequenceLabels, layers, options);

% Predict using the trained LSTM network
predictedLabelsLSTM = classify(net, sequences, 'MiniBatchSize', 1);

% Concatenate true labels and predictions
allTrueLabelsLSTM = [];
allPredictedLabelsLSTM = [];

for i = 1:length(sequenceLabels)
    allTrueLabelsLSTM = [allTrueLabelsLSTM; sequenceLabels{i}(:)];
end

for i = 1:length(predictedLabelsLSTM)
    allPredictedLabelsLSTM = [allPredictedLabelsLSTM; predictedLabelsLSTM{i}(:)];
end

% Ensure that the concatenated labels are categorical arrays
allTrueLabelsLSTM = categorical(allTrueLabelsLSTM);
allPredictedLabelsLSTM = categorical(allPredictedLabelsLSTM);

% Confusion matrix for LSTM model
confMatLSTM = confusionmat(allTrueLabelsLSTM, allPredictedLabelsLSTM, 'Order', classNames);

% Display LSTM performance
disp('LSTM Model Confusion Matrix:');
disp(array2table(confMatLSTM, 'VariableNames', cellstr(classNames), 'RowNames', cellstr(classNames)));

% Compute overall metrics for LSTM model
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1Score = zeros(numClasses, 1);

for c = 1:numClasses
    tp = confMatLSTM(c, c);
    fp = sum(confMatLSTM(:, c)) - tp;
    fn = sum(confMatLSTM(c, :)) - tp;
    precision(c) = tp / (tp + fp);
    recall(c) = tp / (tp + fn);
    f1Score(c) = 2 * (precision(c) * recall(c)) / (precision(c) + recall(c));
end

% Display LSTM metrics
for c = 1:numClasses
    disp(['LSTM Class ', char(classNames(c)), ':']);
    disp(['  Precision: ', num2str(precision(c))]);
    disp(['  Recall: ', num2str(recall(c))]);
    disp(['  F1-Score: ', num2str(f1Score(c))]);
end

% Add to your existing code for visualizations

% 1. Confusion Matrix Heatmap for LSTM
figure;
heatmap({'0 (Not Infected)', '1 (Infected)'}, ...
        {'0 (Not Infected)', '1 (Infected)'}, ...
        confMatLSTM, ...
        'Title', 'LSTM Confusion Matrix', ...
        'XLabel', 'Predicted', ...
        'YLabel', 'Actual', ...
        'Colormap', parula, ...
        'ColorbarVisible', true);

% 2. Confidence Scores (Prediction Probabilities)
% Obtain confidence scores for LSTM predictions
confidenceScores = cellfun(@(seq) predict(net, seq), sequences, 'UniformOutput', false);

% Combine confidence scores into a single matrix
allConfidenceScores = [];
for i = 1:length(confidenceScores)
    allConfidenceScores = [allConfidenceScores; confidenceScores{i}];
end

% Plot confidence scores
figure;
plot(1:size(allConfidenceScores, 1), allConfidenceScores(:, 1), 'r-', 'DisplayName', 'Class 0 Confidence');
hold on;
plot(1:size(allConfidenceScores, 1), allConfidenceScores(:, 2), 'b-', 'DisplayName', 'Class 1 Confidence');
xlabel('Samples');
ylabel('Confidence');
legend;
title('Confidence Scores for LSTM Predictions');
hold off;

% 3. Temporal Trends for Each Patient
for i = 1:length(uniquePatients)
    patientID = uniquePatients(i);
    idx = patientIDs == patientID;
    patientTimestamps = timestamps(idx);
    patientLabels = y(idx);
    patientPredictions = allPredictedLabelsLSTM(idx);

    figure;
    plot(patientTimestamps, double(patientLabels), 'bo-', 'DisplayName', 'Actual Labels');
    hold on;
    plot(patientTimestamps, double(patientPredictions), 'rx-', 'DisplayName', 'Predicted Labels');
    xlabel('Duration in Hours');
    ylabel('Label (Numeric)');
    title(['Patient ', char(patientID), ' Predictions Over Time']);
    legend;
    hold off;
end

% 4. ROC Curve and AUC for LSTM Model
% Compute ROC curve and AUC for binary classification
[Xroc, Yroc, T, AUC] = perfcurve(allTrueLabelsLSTM, allConfidenceScores(:, 2), '1');

% Plot ROC Curve
figure;
plot(Xroc, Yroc, 'b-', 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve for LSTM (AUC = ', num2str(AUC), ')']);
grid on;

