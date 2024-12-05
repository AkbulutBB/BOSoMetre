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

%% Feature Engineering
% Add derived features: Ratios and Total Intensity
colorIntensity = sum(sensorData(:, 1:3), 2);
colorRatios = [sensorData(:, 1) ./ (sensorData(:, 2) + 1e-6), ...
               sensorData(:, 2) ./ (sensorData(:, 3) + 1e-6), ...
               sensorData(:, 3) ./ (sensorData(:, 1) + 1e-6)];

% Combine all features
X = [sensorData, colorIntensity, colorRatios];

% Handle missing labels
hasLabel = ~ismissing(labels);
X = X(hasLabel, :);
y = labels(hasLabel);
timestamps = timestamps(hasLabel);
patientIDs = patientIDs(hasLabel);

% Normalize features
[X, mu, sigma] = zscore(X);

% Extract class names
classNames = categories(y);

%% Prepare sequences for LSTM
uniquePatients = unique(patientIDs);
disp('Unique Patients After Processing:');
disp(uniquePatients);

sequences = {};
sequenceLabels = {};
sequencePatientIDs = {};

for i = 1:length(uniquePatients)
    patientID = uniquePatients(i);
    idx = patientIDs == patientID;
    sequences{end+1} = X(idx, :)';
    labels_i = y(idx)';
    sequenceLabels{end+1} = labels_i;
    sequencePatientIDs{end+1} = patientID;
end

% Verify sequences and labels have matching lengths
for i = 1:length(sequenceLabels)
    seqLength = size(sequences{i}, 2);
    labelLength = length(sequenceLabels{i});
    if seqLength ~= labelLength
        error(['Mismatch in sequence and label length for sequence ', num2str(i)]);
    end
end

%% Initialize performance metrics
allTrueLabelsLSTM = categorical([]);
allPredictedLabelsLSTM = categorical([]);
allScoresLSTM = [];

%% LSTM Model with Fixed Hidden Units (25)
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
        sequencesTrain{end+1} = X(idx, :)';
        sequenceLabelsTrain{end+1} = y(idx)';
    end
    
    % Prepare test sequence
    testIdx = patientIDs == testPatient;
    testSequence = {X(testIdx, :)'};
    testLabels = {y(testIdx)'};
    
    % Define LSTM network architecture with 25 hidden units
    inputSize = size(X, 2);
    numClasses = numel(classNames);
    layers = [
        sequenceInputLayer(inputSize)
        lstmLayer(25, 'OutputMode', 'sequence')
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];
    
    % Set aside some training data for validation (20%)
    numTrainSequences = length(sequencesTrain);
    numValidation = round(0.2 * numTrainSequences);
    validationIndices = randperm(numTrainSequences, numValidation);
    trainIndices = setdiff(1:numTrainSequences, validationIndices);

    % Prepare validation data
    validationSequences = sequencesTrain(validationIndices);
    validationLabels = sequenceLabelsTrain(validationIndices);
    
    % Update training data
    sequencesTrain = sequencesTrain(trainIndices);
    sequenceLabelsTrain = sequenceLabelsTrain(trainIndices);
    
    % Set training options with validation
    options = trainingOptions('adam', ...
        'MaxEpochs', 50, ...
        'MiniBatchSize', 1, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', {validationSequences, validationLabels}, ...
        'ValidationFrequency', 30, ...
        'ValidationPatience', 5, ...
        'Verbose', 1, ...
        'Plots', 'training-progress');
    
    % Train the network
    net = trainNetwork(sequencesTrain, sequenceLabelsTrain, layers, options);
    
    % Test the network
    [predLabels, scores] = classify(net, testSequence, 'MiniBatchSize', 1);
    
    % Store results for overall performance analysis
    currentTrueLabels = testLabels{1}(:);
    currentPredLabels = predLabels{1}(:);
    currentScores = scores{1}';  % Transpose the scores matrix
    
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
% Calculate and display confusion matrix as percentages
figure('Name', 'Confusion Matrix');
cm = confusionchart(allTrueLabelsLSTM, allPredictedLabelsLSTM);
cm.Normalization = 'row-normalized';  % Show percentages by row
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

% Calculate overall accuracy
accuracy = sum(allPredictedLabelsLSTM == allTrueLabelsLSTM) / length(allTrueLabelsLSTM);
fprintf('Overall Accuracy: %.2f%%\n', accuracy * 100);

% Calculate sensitivity and specificity
trueLabels = double(allTrueLabelsLSTM == '1');
predLabels = double(allPredictedLabelsLSTM == '1');
sensitivity = sum(trueLabels & predLabels) / sum(trueLabels);
specificity = sum(~trueLabels & ~predLabels) / sum(~trueLabels);

fprintf('Sensitivity: %.2f%%\n', sensitivity * 100);
fprintf('Specificity: %.2f%%\n', specificity * 100);