% Step 1: Load the data
filePath = 'G:\My Drive\Neuropapers\Githubstuff\BOSoMetre\MAtlabWorkFolder\LongBosometreData.csv'; % Replace with your file path
data = readtable(filePath);

% Display the variable names
disp('Variable Names:');
disp(data.Properties.VariableNames);

% Display the first few rows
disp('First few rows of the data:');
disp(head(data));

% Extract sensor data and BinaryInfected status
sensorData = [data.R, data.G, data.B, data.C]; % Sensor readings
binaryInfected = data.BinaryInfected; % Infection status
timestamps = data.DurationInHours; % Corresponding timestamps

% Convert labels to categorical if not already
binaryInfected = categorical(binaryInfected);

% Sequence parameters
sequenceLength = 10; % Number of time steps before and after
inputSequences = {};
outputLabels = {};

% Iterate over all timestamps with culture results
for i = 1:height(data)
    if ~isnan(data.BinaryInfected(i)) % Only process rows with known labels
        % Extract the label (BinaryInfected)
        label = data.BinaryInfected(i);
        
        % Find the range of sensor data around this timestamp
        startIdx = max(1, i - sequenceLength);
        endIdx = min(height(data), i + sequenceLength);
        
        % Extract sensor data sequence
        sequence = sensorData(startIdx:endIdx, :)';
        
        % Ensure sequence length matches the expected size
        if size(sequence, 2) == (endIdx - startIdx + 1)
            inputSequences{end+1} = sequence; % Add sequence to inputs
            outputLabels{end+1} = label; % Add label
        end
    end
end


% Convert labels to categorical
outputLabels = categorical(cell2mat(outputLabels)); % Convert cell to categorical

% Split data into training and testing
splitIdx = floor(0.8 * length(inputSequences));
trainSequences = inputSequences(1:splitIdx);
trainLabels = outputLabels(1:splitIdx);
testSequences = inputSequences(splitIdx+1:end);
testLabels = outputLabels(splitIdx+1:end);

% Ensure training data is not empty
if isempty(trainSequences) || isempty(trainLabels)
    error('Training data is empty. Adjust the data split or sequence generation logic.');
end

% Define LSTM network architecture
inputSize = size(sensorData, 2); % Number of sensor features (columns in sensorData)
numHiddenUnits = 50; % Number of hidden units
numClasses = numel(categories(binaryInfected)); % Number of classes (Infected, Not Infected)

layers = [
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits, 'OutputMode', 'last') % Use 'last' output for classification
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

% Train the LSTM model
net = trainNetwork(trainSequences, trainLabels, layers, options);

% Test the LSTM model on unseen data
[predictedLabels, scores] = classify(net, testSequences, 'MiniBatchSize', 1);

% Evaluate performance
accuracy = sum(predictedLabels == testLabels) / numel(testLabels);
disp(['Test Accuracy: ', num2str(accuracy)]);

% Confusion Matrix
confMat = confusionmat(testLabels, predictedLabels);
disp('Confusion Matrix:');
disp(array2table(confMat, ...
    'VariableNames', categories(testLabels), ...
    'RowNames', categories(testLabels)));

% Visualize Predictions vs Actual
for i = 1:length(testSequences)
    sequence = testSequences{i};
    trueLabel = testLabels(i);
    predictedLabel = predictedLabels(i);
    
    figure;
    plot(sequence'); % Plot sensor data sequence
    title(['Sequence ', num2str(i), ': True Label = ', char(trueLabel), ...
           ', Predicted Label = ', char(predictedLabel)]);
    xlabel('Time Step');
    ylabel('Sensor Reading');
    legend({'R', 'G', 'B', 'C'});
    grid on;
end
