% Step 1: Load the data
filePath = 'G:\My Drive\Neuropapers\Githubstuff\BOSoMetre\MAtlabWorkFolder\LongBosometreData.csv'; % Replace with your file path
data = readtable(filePath);

% Display the variable names
disp('Variable Names:');
disp(data.Properties.VariableNames);

% Display the first few rows
disp('First few rows of the data:');
disp(head(data));

% Extract relevant data
timestamps = data.DurationInHours; % Time points
sensorData = [data.R, data.G, data.B, data.C]; % Sensor readings (R, G, B, C)
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
               sensorData(:, 3) ./ (sensorData(:, 1) + 1e-6)];    % B/R

% Combine all features: Individual channels, turbidity (C), and derived features
% Now X includes R, G, B, C, Intensity, R/G, G/B, B/R
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
    confMat = confusionmat(y_test, y_pred); % Removed 'Order' parameter
    confusionMatrices(:, :, i) = confMat;
    
    % Collect for overall performance
    allTrueLabels = [allTrueLabels; y_test];
    allPredictedLabels = [allPredictedLabels; y_pred];
    
    % Display performance for this patient
    disp(['Patient ', num2str(testPatient), ' Accuracy: ', num2str(accuracy)]);
end

% Overall performance
meanAccuracy = mean(accuracies);
disp(['Leave-One-Patient-Out Mean Accuracy: ', num2str(meanAccuracy)]);

% Overall confusion matrix
overallConfMat = confusionmat(allTrueLabels, allPredictedLabels);
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
    if (tp + fp) == 0
        precision(c) = 0;
    else
        precision(c) = tp / (tp + fp);
    end
    if (tp + fn) == 0
        recall(c) = 0;
    else
        recall(c) = tp / (tp + fn);
    end
    if (precision(c) + recall(c)) == 0
        f1Score(c) = 0;
    else
        f1Score(c) = 2 * (precision(c) * recall(c)) / (precision(c) + recall(c));
    end
end

% Display metrics
for c = 1:numClasses
    disp(['Class ', char(classNames(c)), ':']);
    disp(['  Precision: ', num2str(precision(c))]);
    disp(['  Recall: ', num2str(recall(c))]);
    disp(['  F1-Score: ', num2str(f1Score(c))]);
end

% Plot Confusion Matrix for Traditional Model
figure;
confusionchart(allTrueLabels, allPredictedLabels, 'Title', 'Traditional Model Confusion Matrix', 'Normalization', 'row-normalized');
set(gca, 'FontSize', 12);

% Plotting predictions vs actual labels per patient (optional)
for i = 1:length(uniquePatients)
    patientID = uniquePatients(i);
    idx = patientIDs == patientID;
    patientTimestamps = timestamps(idx);
    patientLabels = y(idx);
    
    % Predict for this patient's data
    patientData = X(idx, :);
    patientPredictions = predict(model, patientData);
    
    figure;
    plot(patientTimestamps, double(patientLabels) - 1, 'bo-', 'LineWidth', 1.5, 'DisplayName', 'Actual Labels');
    hold on;
    plot(patientTimestamps, double(patientPredictions) - 1, 'rx-', 'LineWidth', 1.5, 'DisplayName', 'Predicted Labels');
    xlabel('Duration in Hours');
    ylabel('Infection Status');
    title(['Patient ', num2str(patientID)]);
    legend('Location', 'best');
    ylim([-0.1, 1.1]);
    yticks([0, 1]);
    yticklabels({'Not Infected', 'Infected'});
    grid on;
    hold off;
end

% Optional: Feature importance using predictor importance estimates
% This requires Statistics and Machine Learning Toolbox

% Train a bagged tree ensemble to estimate feature importance
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

% Advanced Model: Using LSTM for time-series data
% Prepare data for LSTM
sequences = {};
sequenceLabels = {};

for i = 1:length(uniquePatients)
    idx = patientIDs == uniquePatients(i);
    % Extract the sequence of features for the patient (including individual channels and derived features)
    sequences{end+1} = X(idx, :)'; % Size: [Features x TimeSteps]
    % Extract the sequence of labels for the patient
    labels_i = y(idx)'; % Row vector
    sequenceLabels{end+1} = labels_i;
end

% Convert labels in sequenceLabels to categorical row vectors
for i = 1:length(sequenceLabels)
    sequenceLabels{i} = categorical(sequenceLabels{i});
end

% Verify sequences and labels have matching lengths
for i = 1:length(sequenceLabels)
    seqLength = size(sequences{i}, 2); % Number of time steps
    labelLength = length(sequenceLabels{i});
    if seqLength ~= labelLength
        error(['Mismatch in sequence and label length for sequence ', num2str(i)]);
    end
    % Check if labels are row vectors
    if size(sequenceLabels{i}, 1) ~= 1
        error(['Labels for sequence ', num2str(i), ' are not row vectors']);
    end
end

% Define LSTM network architecture
inputSize = size(X, 2); % Now includes all features
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
    'MaxEpochs', 100, ...
    'MiniBatchSize', 1, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', 0, ...
    'Plots', 'training-progress');

% Train LSTM network
net = trainNetwork(sequences, sequenceLabels, layers, options);

% Predict using the trained LSTM network
[predictedLabelsLSTM, scoresLSTM] = classify(net, sequences, 'MiniBatchSize', 1);

% Concatenate true labels and predictions
allTrueLabelsLSTM = [];
allPredictedLabelsLSTM = [];
allScoresLSTM = [];

for i = 1:length(sequenceLabels)
    allTrueLabelsLSTM = [allTrueLabelsLSTM; sequenceLabels{i}(:)];
    allPredictedLabelsLSTM = [allPredictedLabelsLSTM; predictedLabelsLSTM{i}(:)];
    allScoresLSTM = [allScoresLSTM; scoresLSTM{i}'];
end

% Ensure that the concatenated labels are categorical arrays
allTrueLabelsLSTM = categorical(allTrueLabelsLSTM);
allPredictedLabelsLSTM = categorical(allPredictedLabelsLSTM);

% Confusion matrix for LSTM model
confMatLSTM = confusionmat(allTrueLabelsLSTM, allPredictedLabelsLSTM);

% Display LSTM performance
disp('LSTM Model Confusion Matrix:');
disp(array2table(confMatLSTM, 'VariableNames', cellstr(classNames), 'RowNames', cellstr(classNames)));

% Compute overall metrics for LSTM model
precisionLSTM = zeros(numClasses, 1);
recallLSTM = zeros(numClasses, 1);
f1ScoreLSTM = zeros(numClasses, 1);

for c = 1:numClasses
    tp = confMatLSTM(c, c);
    fp = sum(confMatLSTM(:, c)) - tp;
    fn = sum(confMatLSTM(c, :)) - tp;
    if (tp + fp) == 0
        precisionLSTM(c) = 0;
    else
        precisionLSTM(c) = tp / (tp + fp);
    end
    if (tp + fn) == 0
        recallLSTM(c) = 0;
    else
        recallLSTM(c) = tp / (tp + fn);
    end
    if (precisionLSTM(c) + recallLSTM(c)) == 0
        f1ScoreLSTM(c) = 0;
    else
        f1ScoreLSTM(c) = 2 * (precisionLSTM(c) * recallLSTM(c)) / (precisionLSTM(c) + recallLSTM(c));
    end
end

% Display LSTM metrics
for c = 1:numClasses
    disp(['LSTM Class ', char(classNames(c)), ':']);
    disp(['  Precision: ', num2str(precisionLSTM(c))]);
    disp(['  Recall: ', num2str(recallLSTM(c))]);
    disp(['  F1-Score: ', num2str(f1ScoreLSTM(c))]);
end

% Plot Confusion Matrix for LSTM Model
figure;
confusionchart(allTrueLabelsLSTM, allPredictedLabelsLSTM, 'Title', 'LSTM Model Confusion Matrix', 'Normalization', 'row-normalized');
set(gca, 'FontSize', 12);

% Plotting Predictions vs Actual Labels Over Time for Each Patient (LSTM)
for i = 1:length(uniquePatients)
    patientID = uniquePatients(i);
    idx = patientIDs == patientID;
    
    % Extract patient data
    patientTimestamps = timestamps(idx);
    patientLabels = y(idx);
    patientSequence = X(idx, :)';
    
    % Get predicted labels for the patient
    [patientPredictions, ~] = classify(net, {patientSequence}, 'MiniBatchSize', 1);
    patientPredictions = patientPredictions{1}';
    
    figure;
    plot(patientTimestamps, double(patientLabels) - 1, 'bo-', 'LineWidth', 1.5, 'DisplayName', 'Actual Labels');
    hold on;
    plot(patientTimestamps, double(patientPredictions) - 1, 'rx-', 'LineWidth', 1.5, 'DisplayName', 'Predicted Labels');
    xlabel('Duration in Hours');
    ylabel('Infection Status');
    title(['Patient ', num2str(patientID), ' - LSTM Predictions']);
    legend('Location', 'best');
    ylim([-0.1, 1.1]);
    yticks([0, 1]);
    yticklabels({'Not Infected', 'Infected'});
    grid on;
    hold off;
end

% Plot ROC Curve for LSTM Model
% Convert class labels to binary (assuming '1' represents 'Infected')
allTrueLabelsBinary = double(allTrueLabelsLSTM == '1');

% Display classNames to verify its contents
disp('Class Names:');
disp(classNames);

% Find index of 'Infected' class in classNames
infectedClassIdx = find(strcmp(classNames, '1'));

% Check if infectedClassIdx is not empty
if isempty(infectedClassIdx)
    error('Class label ''1'' not found in classNames.');
end

% Extract probabilities of the positive class
probInfected = allScoresLSTM(:, infectedClassIdx);

% Compute ROC curve
[rocX, rocY, rocT, auc] = perfcurve(allTrueLabelsBinary, probInfected, 1);

% Plot ROC curve
figure;
plot(rocX, rocY, 'b-', 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['LSTM Model ROC Curve (AUC = ', num2str(auc), ')']);
grid on;

% Plot Precision-Recall Curve
[prX, prY, prT, prAUC] = perfcurve(allTrueLabelsBinary, probInfected, 1, 'xCrit', 'reca', 'yCrit', 'prec');

figure;
plot(prX, prY, 'r-', 'LineWidth', 2);
xlabel('Recall');
ylabel('Precision');
title(['LSTM Model Precision-Recall Curve (AUC = ', num2str(prAUC), ')']);
grid on;
