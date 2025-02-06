% Load ambiguous data
filePath = 'G:\My Drive\Neuropapers\Githubstuff\BOSoMetre\MAtlabWorkFolder\ambigousData.csv';
ambiguousData = readtable(filePath);

% Extract specific sensor columns
sensorColumns = {'RPerc', 'GPerc', 'BPerc', 'CPerc'};
features = ambiguousData(:, sensorColumns); % Extract as a table

% Use the predictFcn field to classify the data
[predictions, scores] = binaryModel.predictFcn(features);

% Define thresholds
confidenceThreshold = 0.6; % Minimum confidence required for prediction
separationThreshold = 0.2; % Minimal difference between class proportions

% Calculate confidence and separation
maxScores = max(scores, [], 2); % Maximum proportion of neighbors for any class
scoreSeparation = abs(scores(:, 1) - scores(:, 2)); % Difference in proportions

% Flag low-confidence and low-separation samples
isLowConfidence = maxScores < confidenceThreshold; % Confidence below threshold
isLowSeparation = scoreSeparation < separationThreshold; % Poor separation

% Combine conditions for "Inconclusive" classification
isInconclusive = isLowConfidence | isLowSeparation; % Either condition triggers "Inconclusive"

% Assign "Inconclusive" where conditions are not met
finalPredictions = string(predictions); % Convert numeric predictions to strings
finalPredictions(isInconclusive) = "Inconclusive";

% Map numeric values to readable class names
finalPredictions(finalPredictions == "0") = "Not Infected";
finalPredictions(finalPredictions == "1") = "Infected";

% Ensure predictions are categorical with all possible classes
finalPredictions = categorical(finalPredictions, {'Not Infected', 'Infected', 'Inconclusive'});

% Add predictions to the data table
ambiguousData.Predictions = finalPredictions;

% Save the results to a new CSV file
outputFilename = 'classified_ambiguous_data.csv';
writetable(ambiguousData, outputFilename);

% Display summary
disp('=== Classification Summary ===');
disp(['Results saved to: ', outputFilename]);
disp(' ');

% Display counts for each class
disp('=== Class Counts ===');
classCounts = countcats(finalPredictions);
categoriesList = categories(finalPredictions);
for i = 1:numel(categoriesList)
    fprintf('Class %s: %d samples\n', categoriesList{i}, classCounts(i));
end

% Display ambiguous sample details
disp(' ');
disp('=== Ambiguous Sample Details ===');
disp(['Total ambiguous samples (Inconclusive): ', num2str(sum(finalPredictions == "Inconclusive"))]);
disp(['Low-confidence samples: ', num2str(sum(isLowConfidence))]);
disp(['Low-separation samples: ', num2str(sum(isLowSeparation))]);
