% Define file paths
filePath = 'G:\My Drive\Neuropapers\Githubstuff\BOSoMetre\MAtlabWorkFolder\IDSAfulltest_dataset.csv';
outputPath = 'G:\My Drive\Neuropapers\Githubstuff\BOSoMetre\MAtlabWorkFolder\8020IDSAData.csv';

% Load the data
data = readtable(filePath);

% Extract features specified in RequiredVariables
colorFeatures = subspaceIDSA.RequiredVariables; % Ensure these match the feature names in your CSV
X = data(:, colorFeatures); % Use table slicing to retain column names
trueLabels = data.infClassIDSA; % True labels (optional)

% Use the predictFcn for predictions
[predictedLabels, scores] = subspaceIDSA.predictFcn(X);
semiha=[predictedLabels, scores]
% Extract confidence levels for the predicted class
confidenceLevels = max(scores, [], 2);

% Replace confidence levels less than 1 with NaN
confidenceLevels(confidenceLevels < 1) = NaN;

% Create a table for the output
outputTable = table(trueLabels, predictedLabels, confidenceLevels, ...
    'VariableNames', {'TrueLabels', 'PredictedLabels', 'ConfidenceLevels'});

% Write the table to a CSV file
writetable(outputTable, outputPath);

% Generate and display confusion matrix
confMat = confusionmat(trueLabels, predictedLabels);
disp('Confusion Matrix:');
disp(confMat);

% Display the classification report
classLabels = unique(trueLabels); % Get unique class labels
disp('Classification Report:');
for i = 1:length(classLabels)
    label = classLabels(i);
    TP = sum(predictedLabels == label & trueLabels == label);
    FP = sum(predictedLabels == label & trueLabels ~= label);
    FN = sum(predictedLabels ~= label & trueLabels == label);
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    fprintf('Class %d: Precision = %.2f, Recall = %.2f\n', label, precision, recall);
end

disp('Classification and confidence levels saved to CSV file with NaN for low confidence.');
