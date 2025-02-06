%% Clear Workspace and Load Data
clc;
clearvars -except BOSoMetreDataCSFDataForTraining

if ~exist('BOSoMetreDataCSFDataForTraining', 'var')
    error('BOSoMetreDataCSFDataForTraining is not found in the workspace.');
end
data = BOSoMetreDataCSFDataForTraining;

%% Exclude Specific Patients from Analysis
% Exclude patients with IDs 9, 17, 18, 19 so that they are not used for training or testing.
excludePatients = [9, 17, 18, 19, 21];
data = data(~ismember(data.InStudyID, excludePatients), :);

% Extract features and target variables
features = data{:, {'RPerc', 'GPerc', 'BPerc', 'CPerc'}};
infClass = data.infClassIDSA;       % Binary target (0/1)
triClass = data.triClassIDSA;       % Tri-class target (0/1/2)
patientIDs = unique(data.InStudyID);
numPatients = numel(patientIDs);

% Initialize performance storage
results = struct();

% Define classifiers to use
classifiers = {
    'SVM', @fitcsvm, {'KernelFunction', 'linear'};
    'DecisionTree', @fitctree, {};
    'KNN', @fitcknn, {}
};

% Define classification tasks
tasks = {
    'Binary', infClass;
    'TriClass', triClass
};

% Initialize accuracy matrices
for t = 1:size(tasks,1)
    for c = 1:size(classifiers,1)
        results.(tasks{t,1}).(classifiers{c,1}) = zeros(numPatients, 1);
    end
end

% LOOCV main loop
for i = 1:numPatients
    fprintf('Processing patient %d/%d...\n', i, numPatients);
    
    % Create patient-specific split
    testMask = (data.InStudyID == patientIDs(i));
    X_train = features(~testMask, :);
    X_test = features(testMask, :);
    
    for t = 1:size(tasks,1)
        % Get current task parameters
        taskName = tasks{t,1};
        y = tasks{t,2};
        y_train = y(~testMask);
        y_test = y(testMask);
        
        % Skip if test set contains unseen classes
        if numel(unique(y_train)) < numel(unique(y))
            warning('Skipping fold %d - missing classes in training', i);
            continue;
        end
        
        for c = 1:size(classifiers,1)
            % Train classifier
            if strcmp(taskName, 'TriClass') && strcmp(classifiers{c,1}, 'SVM')
                model = fitcecoc(X_train, y_train);
            else
                model = classifiers{c,2}(X_train, y_train, classifiers{c,3}{:});
            end
            
            % Predict and evaluate
            y_pred = predict(model, X_test);
            acc = sum(y_pred == y_test) / numel(y_test);
            
            % Store results
            results.(taskName).(classifiers{c,1})(i) = acc;
        end
    end
end

% Calculate summary statistics
for t = 1:size(tasks,1)
    taskName = tasks{t,1};
    fprintf('\n%s Classification Results:\n', taskName);
    fprintf('%-12s\tMean\tStd\n', 'Classifier');
    
    for c = 1:size(classifiers,1)
        classifierName = classifiers{c,1};
        accuracies = results.(taskName).(classifierName);
        validAcc = accuracies(accuracies ~= 0);
        
        meanAcc = mean(validAcc);
        stdAcc = std(validAcc);
        
        fprintf('%-12s\t%.2f\t%.2f\n', ...
                classifierName, meanAcc*100, stdAcc*100);
    end
end

Additional visualizations (optional)
Uncomment these to generate boxplots of the results
figure;
for t = 1:size(tasks,1)
    subplot(1,2,t);
    taskName = tasks{t,1};
    accData = [];
    groups = {};
    for c = 1:size(classifiers,1)
        classifierName = classifiers{c,1};
        accData = [accData; results.(taskName).(classifierName)];
        groups = [groups; repmat({classifierName}, numPatients, 1)];
    end
    boxplot(accData, groups);
    title([taskName ' Classification Accuracy']);
    ylabel('Accuracy');
    ylim([0 1]);
end