% Clear workspace and close all figures
clear; close all; clc;

% Step 1: Load the data
filePath = 'G:\My Drive\Neuropapers\Githubstuff\BOSoMetre\MAtlabWorkFolder\LongBosometreData.csv'; % Replace with your file path
data = readtable(filePath);

% Step 2: Correct Patient IDs
if iscell(data.InStudyID)
    data.InStudyID = str2double(data.InStudyID); % Convert text to numeric
end
data.InStudyID = int32(data.InStudyID); % Ensure integer values

% Handle missing or invalid IDs
if any(isnan(data.InStudyID))
    disp('Warning: Missing Patient IDs detected. Replacing with placeholder -1.');
    data.InStudyID = fillmissing(data.InStudyID, 'constant', -1);
end

% Debugging: Verify the corrected patient IDs
disp('Corrected Unique Patient IDs:');
disp(unique(data.InStudyID));

% Extract relevant data
timestamps = data.DurationInHours; % Time points
sensorData = [data.RPerc, data.GPerc, data.BPerc, data.CPerc]; % Sensor readings (R, G, B, C)
labels = data.BinaryInfected; % Infection status
patientIDs = data.InStudyID; % Patient identifiers

% Convert labels to categorical if they are not already
labels = categorical(labels);

% Check data size
disp(['Total samples: ', num2str(length(timestamps))]);

%% Feature Engineering
% Add derived features: Ratios and Total Intensity
colorIntensity = sum(sensorData(:, 1:3), 2); % Total intensity (R + G + B)
colorRatios = [sensorData(:, 1) ./ (sensorData(:, 2) + 1e-6), ... % R/G
               sensorData(:, 2) ./ (sensorData(:, 3) + 1e-6), ... % G/B
               sensorData(:, 3) ./ (sensorData(:, 1) + 1e-6)];    % B/R

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

% Extract class names
classNames = categories(y);

% Extract class names
classNames = categories(y);
numClasses = length(classNames); % Determine the number of classes

%% Prepare sequences for LSTM
uniquePatients = unique(patientIDs);
disp('Unique Patients After Processing:');
disp(uniquePatients);

% Initialize storage for all results
allTrueLabelsLSTM = categorical([]);
allPredictedLabelsLSTM = categorical([]);
allScoresLSTM = []; % Initialize as empty to handle dynamic assignment
allPatientIDs = [];

%% LSTM Model with Leave-One-Out Validation
for i = 1:length(uniquePatients)
    % Test patient
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
    
    % Prepare test sequence and labels
    idxTest = patientIDs == testPatient;
    sequenceTest = X(idxTest, :)';
    sequenceLabelsTest = y(idxTest)';
    
    % Define LSTM network
    inputSize = size(X, 2);
    layers = [
        sequenceInputLayer(inputSize)
        lstmLayer(35, 'OutputMode', 'sequence')
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];


    % Set training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 50, ...
        'MiniBatchSize', 1, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', 1);

    % Train the network
    [netFinal, info] = trainNetwork(sequencesTrain, sequenceLabelsTrain, layers, options);
    
    % Test the network on the test patient
    [testPredLabels, testScores] = classify(netFinal, {sequenceTest}, 'MiniBatchSize', 1);
    
    % Check and align dimensions before concatenation
    currentScores = testScores{1};
    if isempty(allScoresLSTM)
        allScoresLSTM = zeros(0, size(currentScores, 2)); % Initialize with correct dimensions
    end
    
    if size(currentScores, 2) ~= size(allScoresLSTM, 2)
        error('Score dimensions mismatch. Ensure consistent model output.');
    end
    
    % Append results
    allTrueLabelsLSTM = [allTrueLabelsLSTM; sequenceLabelsTest(:)];
    allPredictedLabelsLSTM = [allPredictedLabelsLSTM; testPredLabels{1}(:)];
    allScoresLSTM = [allScoresLSTM; currentScores];
    allPatientIDs = [allPatientIDs; repmat(testPatient, length(sequenceLabelsTest), 1)];
    
    % Display patient-specific results
    fprintf('\nPatient %d Results:\n', testPatient);
    fprintf('Validation Loss: %.4f\n', info.TrainingLoss(end));
    fprintf('Test Accuracy: %.2f%%\n', sum(testPredLabels{1} == sequenceLabelsTest) / length(sequenceLabelsTest) * 100);
end

%% Evaluate Overall Performance
allTrueLabelsBinary = double(allTrueLabelsLSTM == '1');

% Ensure labels and scores are aligned
if length(allTrueLabelsBinary) ~= size(allScoresLSTM, 1)
    error('Mismatch between binary labels and predicted scores.');
end

% Extract probabilities for the positive class
infectedClassIdx = find(strcmp(classNames, '1'));
probInfected = allScoresLSTM(:, infectedClassIdx);

% Compute ROC curve
[rocX, rocY, rocT, auc] = perfcurve(allTrueLabelsBinary, probInfected, 1);

% Plot ROC curve
figure;
plot(rocX, rocY, 'b-', 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve (AUC = ', num2str(auc), ')']);
grid on;
