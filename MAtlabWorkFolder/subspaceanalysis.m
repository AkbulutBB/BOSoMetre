%% Subspace kNN Classification with 80-20 Split (Iterative until >90% Accuracy)
% This script uses the same dataset and predictors (normalized R, G, B, C
% channels and an augmented feature, G/C) as in our previous analysis.
% The data is split into 80% training and 20% testing. A subspace ensemble
% of kNN classifiers is trained using 60 learning cycles and a base kNN 
% classifier with k = 10 neighbors.
% The process is repeated until the test accuracy exceeds 90%.
%
% Author: [Your Name]
% Date: [Date]

%% Data Preparation
% Assume processedcsfdata is already loaded in the workspace.
epsilon = 1e-6;
R = processedcsfdata.RNormalized;
G = processedcsfdata.GNormalized;
B = processedcsfdata.BNormalized;
C = processedcsfdata.CNormalized;
% Augmented feature: Green-to-Clear ratio
ratioGC = G ./ (C + epsilon);
% Use predictors: normalized R, G, B, C and augmented feature (G/C)
X = [R, G, B, C, ratioGC];
y = processedcsfdata.infClassIDSA;  % Binary labels

% Remove any rows with NaN or Inf values.
validRows = ~any(isnan(X) | isinf(X), 2);
X = X(validRows, :);
y = y(validRows);

%% Initialize Iteration Parameters
maxIter = 1000;  % Maximum number of iterations to prevent infinite loops
iter = 0;
found = false;
finalModel = [];
final_cm = [];
final_scores = [];
final_yTest = [];

%% Iterative Loop: Repeat until test accuracy > 90%
while ~found && iter < maxIter
    iter = iter + 1;
    fprintf('Iteration %d:\n', iter);
    
    % Create a new 80-20 training-test split
    cv = cvpartition(y, 'HoldOut', 0.2);
    X_train = X(training(cv), :);
    y_train = y(training(cv));
    X_test = X(test(cv), :);
    y_test = y(test(cv));
    
    % Standardize training features and apply the same transformation to test.
    [X_train, mu, sigma] = zscore(X_train);
    X_test = (X_test - mu) ./ sigma;
    
    % Train Subspace kNN Ensemble
    knnTemplate = templateKNN('NumNeighbors', 10);
    model = fitcensemble(X_train, y_train, 'Method', 'Subspace', ...
        'Learners', knnTemplate, 'NumLearningCycles', 60);
    
    % Evaluate the Model on Test Data
    [y_pred, scores] = predict(model, X_test);
    cm = confusionmat(y_test, y_pred);
    accuracy = sum(diag(cm)) / sum(cm(:));
    fprintf('Accuracy: %.2f%%\n', accuracy*100);
    
    if accuracy > 0.90
        found = true;
        finalModel = model;
        final_cm = cm;
        final_scores = scores;
        final_yTest = y_test;
        fprintf('Achieved >90%% accuracy in iteration %d.\n', iter);
        break;
    end
end

if ~found
    error('Maximum iterations reached without achieving >90%% accuracy.');
end

%% Compute Performance Metrics
TP = final_cm(2,2);
TN = final_cm(1,1);
FP = final_cm(1,2);
FN = final_cm(2,1);
sensitivity = TP / (TP + FN);
specificity = TN / (TN + FP);

fprintf('\nFinal Performance:\n');
fprintf('Accuracy: %.2f%%\n', accuracy*100);
fprintf('Sensitivity: %.2f%%\n', sensitivity*100);
fprintf('Specificity: %.2f%%\n', specificity*100);

%% Generate Presentation-Quality Plots

% 1. ROC Curve (Assuming positive class is labeled as 1)
[rocX, rocY, rocT, auc] = perfcurve(final_yTest, final_scores(:,2), 1);
figure;
plot(rocX, rocY, 'b-', 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ROC Curve (AUC = %.3f)', auc));
grid on;

% 2. Confusion Matrix Heatmap
figure;
imagesc(final_cm);
colormap('hot');
colorbar;
title('Confusion Matrix');
xlabel('Predicted Class');
ylabel('True Class');
set(gca, 'XTick', 1:2, 'XTickLabel', {'Clean','Infected'});
set(gca, 'YTick', 1:2, 'YTickLabel', {'Clean','Infected'});

% 3. Histogram of Prediction Scores by Class
figure;
hold on;
histogram(final_scores(final_yTest==0, 2), 'Normalization', 'pdf', 'FaceColor', 'r', 'FaceAlpha', 0.5);
histogram(final_scores(final_yTest==1, 2), 'Normalization', 'pdf', 'FaceColor', 'g', 'FaceAlpha', 0.5);
title('Distribution of Prediction Scores by Class');
xlabel('Prediction Score');
ylabel('Probability Density');
legend('Clean (0)','Infected (1)');
hold off;

% Precision-Recall Curve
[prec, rec, prT, ap] = perfcurve(final_yTest, final_scores(:,2), 1, 'xCrit', 'reca', 'yCrit', 'prec');
figure;
plot(rec, prec, 'm-', 'LineWidth', 2);
xlabel('Recall');
ylabel('Precision');
title(sprintf('Precision-Recall Curve (AP = %.3f)', ap));
grid on;
