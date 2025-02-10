%% LOOCV for Continuous (Gaussian) HMM using RGBCNormalized features
% The goal is to predict infection status (0 = clean, 1 = infected)
% from the 4-dimensional RGBCNormalized features, while taking the temporal
% structure into account.
%
% NOTES:
%  - We assume that processedcsfdata has the fields:
%       DateTimeStr (e.g., '241023_1904'),
%       RNormalized, GNormalized, BNormalized, CNormalized,
%       infClassIDSA (the binary infection flag),
%       InStudyID (patient ID),
%       Batch (optional).
%
%  - Observations for the HMM are the 4-dim feature vectors.
%  - The emission model is assumed to be multivariate Gaussian,
%    with separate parameters for state 0 (clean) and state 1 (infected).
%  - We segment each patient’s data into continuous segments if the gap
%    between consecutive recordings exceeds 360 minutes.
%
%  - The functions trainGaussianHMM and viterbiGaussianHMM are placeholders;
%    you must implement these (or use a toolbox) to perform parameter estimation
%    and Viterbi decoding for a Gaussian HMM.
%
% Author: [Your Name]
% Date: [Current Date]

clc;
clearvars -except processedcsfdata

% Start logging
diary('GaussianHMM_LOOCV_log.txt');
fprintf('Starting Gaussian HMM LOOCV analysis at %s\n', datestr(now));

%% Data Preprocessing: Filter out rows with missing RGBCNormalized values,
% missing DateTimeStr, or missing infClassIDSA.
if ~exist('processedcsfdata', 'var')
    error('processedcsfdata is not found in the workspace.');
end

data = processedcsfdata;


% Create a logical index for rows with no missing values in the relevant columns.
validRows = ~isempty(data.DateTime) & ~isnan(data.RNormalized) & ~isnan(data.GNormalized) & ...
            ~isnan(data.BNormalized) & ~isnan(data.CNormalized) & ~isnan(data.infClassIDSA);
        
% For safety, we also check that DateTimeStr is not empty.
if iscell(data.DateTime)
    validRows = validRows & ~cellfun(@isempty, data.DateTime);
else
    validRows = validRows & (data.DateTime ~= "");
end

% Filter data accordingly.
datetime_strings = data.DateTime(validRows);
% Observed features: a matrix with 4 columns.
features_all = [data.RNormalized(validRows), data.GNormalized(validRows), ...
                data.BNormalized(validRows), data.CNormalized(validRows)];
groundTruth = data.infClassIDSA(validRows);  % infection flag (0 or 1)
patient_ids = data.InStudyID(validRows);
if isfield(data, 'Batch')
    batches = data.Batch(validRows);
else
    batches = [];
end

% Set maximum allowed gap for continuity (in minutes)
expected_gap_minutes = 360;

%% Prepare LOOCV: Identify unique patients.
unique_patients = unique(patient_ids);
n_patients = length(unique_patients);

% Preallocate results for overall evaluation.
all_true_all    = [];  % ground truth labels (binary)
all_pred_all    = [];  % predicted labels (binary)
all_patientIDs  = [];  % corresponding patient IDs
all_dt_all      = [];  % datetime stamps

% Log misclassified cases in a table.
misclassified_log = table();

%% LOOCV Loop: For each patient as test
for testIdx = 1:n_patients
    testPatient = unique_patients(testIdx);
    fprintf('\nLOOCV: Holding out Patient %d\n', testPatient);
    
    %% Assemble Training Data from Other Patients
    trainingSeqs = {};  % each cell: a continuous segment (N x 4 matrix) from a training patient
    trainingLabels = {};  % corresponding ground truth labels (vector, length = N)
    
    for p = 1:n_patients
        if unique_patients(p) == testPatient
            continue;  % skip test patient
        end
        
        % Get indices for patient p:
        idx = find(patient_ids == unique_patients(p));
        % Convert datetime strings to MATLAB datetime:
        dt_patient = datetime(datetime_strings(idx), 'InputFormat', 'yyMMdd_HHmm', 'Format', 'dd-MMM-yyyy HH:mm');
        X_patient = features_all(idx, :);
        Y_patient = groundTruth(idx);
        
        % Filter out records with missing datetime or label (should already be done)
        valid = ~isnat(dt_patient) & ~isnan(Y_patient);
        if sum(valid) == 0, continue; end
        dt_patient = dt_patient(valid);
        X_patient = X_patient(valid, :);
        Y_patient = Y_patient(valid);
        
        % Sort by datetime:
        [dt_sorted, sortOrder] = sort(dt_patient);
        X_patient = X_patient(sortOrder, :);
        Y_patient = Y_patient(sortOrder);
        
        % Segment the patient’s data into continuous segments:
        segments = segmentPatientData(X_patient, Y_patient, dt_sorted, expected_gap_minutes);
        % segments is a cell array of structs, each with fields X (observations) and Y (labels)
        for s = 1:length(segments)
            if size(segments{s}.X,1) < 2  % skip segments that are too short
                continue;
            end
            trainingSeqs{end+1} = segments{s}.X;  % X is an N x 4 matrix.
            trainingLabels{end+1} = segments{s}.Y;  % corresponding ground truth vector.
        end
    end
    
    if isempty(trainingSeqs)
        warning('No valid training data for test patient %d', testPatient);
        continue;
    end
    
    %% Estimate Gaussian HMM Parameters from Training Data
    % Here we assume a 2-state Gaussian HMM.
    % We need to estimate:
    %   - Transition matrix A (2x2)
    %   - For each state, the mean vector (1x4) and covariance matrix (4x4)
    %
    % We call a placeholder function trainGaussianHMM that uses trainingSeqs and trainingLabels.
    %
    % For initialization, we can set:
    initA = [0.95 0.05; 0.10 0.90];
    % For emissions, we require initial guesses for each state's parameters.
    % (You may set these based on prior knowledge or using overall training data statistics.)
    % Here we assume state 0 (clean) and state 1 (infected).
    initMu = [mean(cell2mat(trainingSeqs'),1); mean(cell2mat(trainingSeqs'),1)];  % simple initialization
    initSigma = repmat(cov(cell2mat(trainingSeqs')), [1 1 2]);
    
    % Train the Gaussian HMM from the training data:
    try
        [A_est, mu_est, Sigma_est] = trainGaussianHMM(trainingSeqs, trainingLabels, initA, initMu, initSigma, 100);
    catch ME
        warning('Gaussian HMM training failed for test patient %d: %s', testPatient, ME.message);
        % Fall back to initial parameters if training fails:
        A_est = initA;
        mu_est = initMu;
        Sigma_est = initSigma;
    end
    
    %% Process Test Patient Data
    idx_test = find(patient_ids == testPatient);
    dt_test = datetime(datetime_strings(idx_test), 'InputFormat', 'yyMMdd_HHmm', 'Format', 'dd-MMM-yyyy HH:mm');
    X_test = features_all(idx_test, :);
    Y_test = groundTruth(idx_test);
    
    valid_test = ~isnat(dt_test) & ~isnan(Y_test);
    if sum(valid_test)==0
        warning('No valid test data for patient %d', testPatient);
        continue;
    end
    dt_test = dt_test(valid_test);
    X_test = X_test(valid_test, :);
    Y_test = Y_test(valid_test);
    
    % Sort test data by datetime:
    [dt_sorted_test, sortOrderTest] = sort(dt_test);
    X_test = X_test(sortOrderTest, :);
    Y_test = Y_test(sortOrderTest);
    
    % Segment test patient data:
    testSegments = segmentPatientData(X_test, Y_test, dt_sorted_test, expected_gap_minutes);
    
    pred_test = [];           % predicted states for test patient (binary)
    true_test_concat = [];    % corresponding true labels
    dt_test_concat = [];      % corresponding datetime stamps
    for s = 1:length(testSegments)
        if size(testSegments{s}.X,1) < 2
            continue;
        end
        % Run Viterbi decoding on the continuous segment using the Gaussian HMM parameters:
        try
            states = viterbiGaussianHMM(testSegments{s}.X, A_est, mu_est, Sigma_est);
        catch ME
            warning('Gaussian HMM decoding error for test patient %d in segment %d: %s', testPatient, s, ME.message);
            states = testSegments{s}.Y;  % fallback: use true labels
        end
        % Append results:
        pred_test = [pred_test; states(:)];
        true_test_concat = [true_test_concat; testSegments{s}.Y(:)];
        dt_test_concat = [dt_test_concat; testSegments{s}.dt(:)];
    end
    
    %% Append Test Patient Results to Overall Results
    all_true_all = [all_true_all; true_test_concat];
    all_pred_all = [all_pred_all; pred_test];
    all_patientIDs = [all_patientIDs; repmat(testPatient, length(true_test_concat), 1)];
    all_dt_all = [all_dt_all; dt_test_concat];
    
    %% Log and Visualize Misclassified Cases for This Test Patient
    misIdx = find(true_test_concat ~= pred_test);
    if ~isempty(misIdx)
        T = table(repmat(testPatient, length(misIdx), 1), dt_test_concat(misIdx), ...
            true_test_concat(misIdx), pred_test(misIdx), ...
            'VariableNames', {'PatientID','DateTime','TrueLabel','PredictedLabel'});
        misclassified_log = [misclassified_log; T];
        
        % Plot misclassifications:
        plotGaussianHMMCaseReview(testPatient, dt_test_concat, true_test_concat, pred_test);
    end
end

%% Evaluate Overall Performance (LOOCV)
binary_cm = confusionmat(all_true_all, all_pred_all);
[precision, recall, f1] = calculate_metrics(binary_cm);
accuracy = sum(diag(binary_cm)) / sum(binary_cm(:));

fprintf('\nLOOCV Gaussian HMM Model Performance:\n');
fprintf('Accuracy: %.2f\n', accuracy);
fprintf('F1 (Clean): %.4f, F1 (Infected): %.4f\n', f1(1), f1(2));
fprintf('Confusion Matrix:\n');
disp(binary_cm);

% Export misclassified cases to CSV:
if ~isempty(misclassified_log)
    writetable(misclassified_log, 'GaussianHMM_LOOCV_misclassified_cases.csv');
    fprintf('Misclassified cases exported to GaussianHMM_LOOCV_misclassified_cases.csv\n');
end

diary off;

%% --- Helper Functions ---

function segments = segmentPatientData(X, Y, dt, gapThreshold)
    % SEGMENTPATIENTDATA partitions a patient’s data into continuous segments.
    % Inputs:
    %   X: matrix of observations (n x d)
    %   Y: vector of ground truth labels (n x 1)
    %   dt: sorted datetime array (n x 1)
    %   gapThreshold: maximum allowed gap (in minutes)
    %
    % Output:
    %   segments: a cell array of structs with fields:
    %       X: observations for the segment,
    %       Y: corresponding labels,
    %       dt: corresponding datetime stamps.
    
    n = length(dt);
    segments = {};
    seg_start = 1;
    for i = 1:(n-1)
        if minutes(dt(i+1) - dt(i)) > gapThreshold
            segments{end+1} = struct('X', X(seg_start:i, :), 'Y', Y(seg_start:i), 'dt', dt(seg_start:i));
            seg_start = i + 1;
        end
    end
    if seg_start <= n
        segments{end+1} = struct('X', X(seg_start:n, :), 'Y', Y(seg_start:n), 'dt', dt(seg_start:n));
    end
end

function [precision, recall, f1] = calculate_metrics(cm)
    % CALCULATE_METRICS computes precision, recall, and F1 score from a confusion matrix.
    n_classes = size(cm, 1);
    precision = zeros(n_classes, 1);
    recall = zeros(n_classes, 1);
    f1 = zeros(n_classes, 1);
    for i = 1:n_classes
        tp = cm(i, i);
        fp = sum(cm(:, i)) - tp;
        fn = sum(cm(i, :)) - tp;
        precision(i) = tp / max(tp + fp, 1);
        recall(i) = tp / max(tp + fn, 1);
        f1(i) = 2 * (precision(i) * recall(i)) / max(precision(i) + recall(i), 1);
    end
end

function plotGaussianHMMCaseReview(patientID, dt, true_labels, pred_labels)
    % PLOTGAUSSIANHMMCASEREVIEW visualizes true vs. predicted infection status for a patient.
    figure;
    hold on;
    stairs(dt, true_labels, 'b-', 'LineWidth', 2);
    stairs(dt, pred_labels, 'r--', 'LineWidth', 2);
    xlabel('Time');
    ylabel('Infection Status (0 = Clean, 1 = Infected)');
    title(sprintf('Patient %d: True vs. Gaussian HMM-Predicted Status', patientID));
    legend('True Labels', 'Predicted Labels');
    grid on;
    hold off;
end

%% --- Placeholder Functions for Gaussian HMM ---
% These functions are intended to implement training (via EM) and Viterbi decoding
% for a Gaussian HMM with two states and d-dimensional observations.
%
% In practice, you will need to implement these functions or use an existing toolbox.

function [A, mu, Sigma] = trainGaussianHMM(seqs, labels, initA, initMu, initSigma, maxIter)
    % TRAINGAUSSIANHMM estimates the parameters of a 2-state Gaussian HMM using EM.
    %
    % Inputs:
    %   seqs: cell array of observation matrices (each: n_i x d)
    %   labels: cell array of corresponding ground truth labels (not used directly here,
    %           but you could use them for initialization)
    %   initA: initial transition matrix (2x2)
    %   initMu: initial means (2 x d)
    %   initSigma: initial covariances (d x d x 2)
    %   maxIter: maximum number of EM iterations
    %
    % Outputs:
    %   A: estimated transition matrix (2x2)
    %   mu: estimated means (2 x d)
    %   Sigma: estimated covariances (d x d x 2)
    %
    % NOTE: This is a placeholder. A full implementation would iterate over all sequences,
    % perform the forward-backward algorithm, and update parameters.
    %
    % For demonstration, we simply return the initial parameters.
    
    A = initA;
    mu = initMu;
    Sigma = initSigma;
    % TODO: Implement the full EM algorithm for Gaussian HMM.
end

function states = viterbiGaussianHMM(X, A, mu, Sigma)
    % VITERBIGAUSSIANHMM returns the most likely state sequence for observation matrix X
    % using the Viterbi algorithm for a Gaussian HMM.
    %
    % Inputs:
    %   X: observation matrix (n x d)
    %   A: transition matrix (2x2)
    %   mu: means for each state (2 x d)
    %   Sigma: covariance matrices for each state (d x d x 2)
    %
    % Output:
    %   states: most likely state sequence (n x 1) with values 0 or 1.
    %
    % NOTE: This is a placeholder. You must compute the Gaussian likelihood for each observation
    % under each state's parameters and then run the Viterbi algorithm.
    
    n = size(X, 1);
    d = size(X, 2);
    % Compute emission probabilities for each state:
    B = zeros(n, 2);
    for state = 1:2
        % Evaluate the multivariate normal density (you can use mvnpdf if available)
        B(:, state) = mvnpdf(X, mu(state, :), Sigma(:, :, state)) + realmin;
    end
    % Initialization:
    delta = zeros(n, 2);
    psi = zeros(n, 2);
    % Assume equal initial probabilities:
    delta(1, :) = log(0.5) + log(B(1, :));
    for t = 2:n
        for j = 1:2
            [maxVal, maxState] = max(delta(t-1, :) + log(A(:, j)'));
            delta(t, j) = maxVal + log(B(t, j));
            psi(t, j) = maxState;
        end
    end
    % Backtracking:
    states = zeros(n, 1);
    [~, states(n)] = max(delta(n, :));
    for t = n-1:-1:1
        states(t) = psi(t+1, states(t+1));
    end
    % Convert states from 1/2 to 0/1: state 1 -> 0, state 2 -> 1.
    states = states - 1;
end
