% Clear workspace and close all figures
clear; close all; clc;

% Step 1: Load the data
filePath = 'G:\My Drive\Neuropapers\Githubstuff\BOSoMetre\MAtlabWorkFolder\FixedBosometreData.csv';
data = readtable(filePath);

% Step 2: Correct Patient IDs
if iscell(data.InStudyID)
    data.InStudyID = str2double(data.InStudyID);
end
data.InStudyID = int32(data.InStudyID);

% Handle missing or invalid IDs
if any(isnan(data.InStudyID))
    disp('Warning: Missing Patient IDs detected. Replacing with placeholder -1.');
    data.InStudyID = fillmissing(data.InStudyID, 'constant', -1);
end

% Debugging: Verify the corrected patient IDs
disp('Corrected Unique Patient IDs:');
disp(unique(data.InStudyID));

% Extract relevant data
timestamps = data.DurationInHours;
sensorData = [data.RPerc, data.GPerc, data.BPerc, data.CPerc];
labels = data.ExtraBinaryInfected;
patientIDs = data.InStudyID;

% Convert labels to categorical if they are not already
labels = categorical(labels);

% Check data size
disp(['Total samples: ', num2str(length(timestamps))]);

%% Enhanced Feature Engineering
% Basic color features
colorIntensity = sum(sensorData(:, 1:3), 2);
colorRatios = [sensorData(:, 1) ./ (sensorData(:, 2) + 1e-6), ...
               sensorData(:, 2) ./ (sensorData(:, 3) + 1e-6), ...
               sensorData(:, 3) ./ (sensorData(:, 1) + 1e-6)];

% Add temporal features (rolling means and deltas)
windowSize = 3;
rollingMeans = movmean(sensorData, windowSize);
deltas = diff([zeros(1, size(sensorData, 2)); sensorData], 1);

% Add interaction terms
interactions = [sensorData(:,1).*sensorData(:,2), ...
               sensorData(:,2).*sensorData(:,3), ...
               sensorData(:,3).*sensorData(:,1)];

% Combine all features
X = [sensorData, colorIntensity, colorRatios, rollingMeans, deltas, interactions];

% Handle missing labels
hasLabel = ~ismissing(labels);
X = X(hasLabel, :);
y = labels(hasLabel);
timestamps = timestamps(hasLabel);
patientIDs = patientIDs(hasLabel);

% Extract class names
classNames = categories(y);

% Calculate class weights for imbalance
classWeights = 1 ./ countcats(y);
classWeights = classWeights / mean(classWeights);

%% Prepare sequences for LSTM
uniquePatients = unique(patientIDs);
disp('Unique Patients After Processing:');
disp(uniquePatients);

%% Initialize performance metrics
allTrueLabelsLSTM = categorical([]);
allPredictedLabelsLSTM = categorical([]);
allScoresLSTM = [];

%% LSTM Model with Leave-One-Patient-Out Cross-Validation
for i = 1:length(uniquePatients)
    % Outer loop: Test patient
    testPatient = uniquePatients(i);
    trainPatients = uniquePatients(uniquePatients ~= testPatient);
    
    % Prepare training sequences and labels
    sequencesTrain = {};
    sequenceLabelsTrain = {};
    for j = 1:length(trainPatients)
        patientID = trainPatients(j);
        idx = patientIDs == patientID;
        
        % Normalize features per patient
        X_patient = X(idx, :);
        X_patient = normalize(X_patient, 'center', 'scale');
        
        sequencesTrain{end+1} = X_patient';
        sequenceLabelsTrain{end+1} = y(idx)';
    end
    
    % Prepare validation data (20% of training data)
    numTrainSequences = length(sequencesTrain);
    numValidation = round(0.2 * numTrainSequences);
    validationIndices = randperm(numTrainSequences, numValidation);
    trainIndices = setdiff(1:numTrainSequences, validationIndices);
    
    % Split validation data
    validationSequences = sequencesTrain(validationIndices);
    validationLabels = sequenceLabelsTrain(validationIndices);
    sequencesTrain = sequencesTrain(trainIndices);
    sequenceLabelsTrain = sequenceLabelsTrain(trainIndices);
    
    % Prepare test sequence
    testIdx = patientIDs == testPatient;
    X_test = X(testIdx, :);
    X_test = normalize(X_test, 'center', 'scale');
    testSequence = {X_test'};
    testLabels = {y(testIdx)'};
    
    % Define improved LSTM architecture
    inputSize = size(X, 2);
    numClasses = numel(classNames);
    numHiddenUnits = 25;
    
    layers = [
        sequenceInputLayer(inputSize)
        lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
        dropoutLayer(0.3)
        lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
        dropoutLayer(0.3)
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer('Classes', classNames, 'ClassWeights', classWeights)];
    
    % Set training options with improved validation
    options = trainingOptions('adam', ...
        'MaxEpochs', 50, ...
        'MiniBatchSize', 1, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', {validationSequences, validationLabels}, ...
        'ValidationFrequency', 30, ...
        'ValidationPatience', 8, ...
        'InitialLearnRate', 0.001, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 5, ...
        'LearnRateDropFactor', 0.2, ...
        'GradientThreshold', 1, ...
        'Verbose', 1, ...
        'Plots', 'training-progress');
    
    % Train the network
    net = trainNetwork(sequencesTrain, sequenceLabelsTrain, layers, options);
    
    % Test the network
    [predLabels, scores] = classify(net, testSequence, 'MiniBatchSize', 1);
    
    % Store results for overall performance analysis
    currentTrueLabels = testLabels{1}(:);
    currentPredLabels = predLabels{1}(:);
    currentScores = scores{1}';
    
    % Concatenate results
    allTrueLabelsLSTM = [allTrueLabelsLSTM; currentTrueLabels];
    allPredictedLabelsLSTM = [allPredictedLabelsLSTM; currentPredLabels];
    
    if isempty(allScoresLSTM)
        allScoresLSTM = currentScores;
    else
        allScoresLSTM = [allScoresLSTM; currentScores];
    end
    
    fprintf('Processed patient %d of %d\n', i, length(uniquePatients));
end

%% Performance Analysis
% Calculate and display confusion matrix
figure('Name', 'Confusion Matrix');
cm = confusionchart(allTrueLabelsLSTM, allPredictedLabelsLSTM);
cm.Normalization = 'row-normalized';
cm.Title = 'LSTM Model Confusion Matrix (%)';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

% Calculate and plot ROC curve
figure('Name', 'ROC Curve');
[X,Y,T,AUC] = perfcurve(allTrueLabelsLSTM, allScoresLSTM(:,2), '1');
plot(X,Y)
grid on
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title(sprintf('ROC Curve (AUC = %.3f)', AUC))

% Calculate overall metrics
accuracy = sum(allPredictedLabelsLSTM == allTrueLabelsLSTM) / length(allTrueLabelsLSTM);
fprintf('Overall Accuracy: %.2f%%\n', accuracy * 100);

% Calculate sensitivity and specificity
trueLabels = double(allTrueLabelsLSTM == '1');
predLabels = double(allPredictedLabelsLSTM == '1');
sensitivity = sum(trueLabels & predLabels) / sum(trueLabels);
specificity = sum(~trueLabels & ~predLabels) / sum(~trueLabels);

fprintf('Sensitivity: %.2f%%\n', sensitivity * 100);
fprintf('Specificity: %.2f%%\n', specificity * 100);

% Find optimal threshold from ROC curve
[~,~,T,~] = perfcurve(allTrueLabelsLSTM, allScoresLSTM(:,2), '1', 'XCrit', 'fpr', 'YCrit', 'tpr');
optimalIdx = find(abs(X + Y - 1) == min(abs(X + Y - 1)), 1);
optimalThreshold = T(optimalIdx);
fprintf('Optimal Decision Threshold: %.3f\n', optimalThreshold);