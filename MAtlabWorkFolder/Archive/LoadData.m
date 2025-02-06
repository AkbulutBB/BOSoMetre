% Step 1: Load the data
filePath = 'G:\My Drive\Neuropapers\BBA -BOSoMetre\MAtlabWorkFolder\LongBosometreData.csv'; % Replace with your file path
data = readtable(filePath);

% Step 2: Display the first few rows
disp('First few rows of the data:');
disp(head(data));

% Step 3: Extract relevant data
timestamps = data.DurationInHours; % Time points
sensorData = [data.R, data.G, data.B, data.C]; % Sensor readings
labels = data.BinaryInfected; % Flags (Binary labels)

% Step 4: Check data size
disp(['Total samples: ', num2str(length(timestamps))]);

% Step 5: Interpolate missing labels
interpolatedLabels = fillmissing(labels, 'nearest');

% Step 6: Feature Engineering
% Add derived features: Ratios and Total Intensity
colorIntensity = sum(sensorData(:, 1:3), 2); % Total intensity (R + G + B)
colorRatios = [sensorData(:, 1) ./ (sensorData(:, 2) + 1e-6), ... % R/G
               sensorData(:, 2) ./ (sensorData(:, 3) + 1e-6), ... % G/B
               sensorData(:, 3) ./ (sensorData(:, 1) + 1e-6)];   % B/R

% Combine all features: Individual channels, turbidity (C), and derived features
X = [sensorData, colorIntensity, colorRatios];

% Step 7: Train-test split
trainIdx = ~isnan(labels); % Rows with known labels
testIdx = isnan(labels);   % Rows with missing labels

X_train = X(trainIdx, :);
y_train = labels(trainIdx);
X_test = X(testIdx, :);

% Normalize features
X_train = normalize(X_train, 'range');
X_test = normalize(X_test, 'range');

% Step 8: Train logistic regression model
model = fitclinear(X_train, y_train, 'Learner', 'logistic');

% Step 9: Predict labels for test data
predictedLabels = predict(model, X_test);

% Step 10: Evaluate on training data
y_pred_train = predict(model, X_train);

% Confusion matrix
confMat = confusionmat(y_train, y_pred_train);
disp('Confusion Matrix (Training Data):');
disp(confMat);

% Accuracy, Precision, Recall, F1-Score
accuracy = sum(diag(confMat)) / sum(confMat(:));
precision = confMat(2, 2) / sum(confMat(:, 2));
recall = confMat(2, 2) / sum(confMat(2, :));
f1_score = 2 * (precision * recall) / (precision + recall);

disp(['Accuracy: ', num2str(accuracy)]);
disp(['Precision: ', num2str(precision)]);
disp(['Recall: ', num2str(recall)]);
disp(['F1-Score: ', num2str(f1_score)]);

% Step 11: Plot original and predicted labels
figure;
plot(timestamps, sensorData(:, 1), 'r-', 'DisplayName', 'R');
hold on;
plot(timestamps, interpolatedLabels, 'k--', 'DisplayName', 'Interpolated Labels');
scatter(timestamps(testIdx), predictedLabels, 50, 'm', 'filled', 'DisplayName', 'Predicted Labels');
xlabel('Time (Hours)');
ylabel('Sensor Readings and Labels');
legend;
title('Predicted vs. Actual Labels');
hold off;

% Step 12: Evaluate predictive power of individual channels
channels = {'R', 'G', 'B', 'C'};
for i = 1:4
    % Train with one channel at a time
    X_train_single = X_train(:, i);
    X_test_single = X_test(:, i);

    % Train logistic regression model
    model_single = fitclinear(X_train_single, y_train, 'Learner', 'logistic');

    % Predict on training data
    y_pred_single = predict(model_single, X_train_single);

    % Compute accuracy
    confMat_single = confusionmat(y_train, y_pred_single);
    accuracy_single = sum(diag(confMat_single)) / sum(confMat_single(:));
    disp(['Accuracy using only ', channels{i}, ': ', num2str(accuracy_single)]);
end
