% Step 1: Load the data
filePath = 'G:\My Drive\Neuropapers\Githubstuff\BOSoMetre\MAtlabWorkFolder\classIDSABOS.csv';
data = readtable(filePath);

% 1. Convert TriDecisionSet to categorical immediately after loading
data.TriDecisionSet = categorical(data.TriDecisionSet);

% 2. Handle missing values
data = fillmissing(data, 'constant', categorical(2), 'DataVariables', 'TriDecisionSet');

% 3. Normalize color percentage features
colorFeatures = {'RPerc', 'GPerc', 'BPerc', 'CPerc'};
data_normalized = data;
for i = 1:length(colorFeatures)
    data_normalized.(colorFeatures{i}) = normalize(data.(colorFeatures{i}));
end

% 4. Remove outliers (optional)
% Using IQR method
for i = 1:length(colorFeatures)
    col = data_normalized.(colorFeatures{i});
    q1 = prctile(col, 25);
    q3 = prctile(col, 75);
    iqr = q3 - q1;
    outlier_mask = col < (q1 - 1.5*iqr) | col > (q3 + 1.5*iqr);
    % Either remove or mark outliers
    % data_normalized(outlier_mask, :) = [];
end

% 5. Calculate correlation matrix
correlationMatrix = corr(data_normalized{:, colorFeatures});

% 6. Create training features matrix
X = data_normalized{:, colorFeatures};
y = data_normalized.TriDecisionSet;

% 7. Visualizations
% Check class distribution
figure;
histogram(data.TriDecisionSet);
title('Distribution of Classes');
xlabel('Class');
ylabel('Frequency');

% Create more meaningful visualizations
figure('Position', [100 100 1200 400]);

% Box plots for each color feature by infection status
subplot(1,4,1);
boxplot(data.RPerc, data.TriDecisionSet);
title('Red % by Infection Status');
xlabel('Class (0=Normal, 1=Infected, 2=Inconclusive)');
ylabel('Red %');

subplot(1,4,2);
boxplot(data.GPerc, data.TriDecisionSet);
title('Green % by Infection Status');
xlabel('Class');
ylabel('Green %');

subplot(1,4,3);
boxplot(data.BPerc, data.TriDecisionSet);
title('Blue % by Infection Status');
xlabel('Class');
ylabel('Blue %');

subplot(1,4,4);
boxplot(data.CPerc, data.TriDecisionSet);
title('Clear % by Infection Status');
xlabel('Class');
ylabel('Clear %');

% Plot histograms with outlier regions highlighted
figure('Position', [100 100 1200 400]);

for i = 1:4
    subplot(1,4,i);
    feat = colorFeatures{i};
    
    % Calculate bounds for each class
    hold on;
    for class = categories(data.TriDecisionSet)'
        class_data = data.(feat)(data.TriDecisionSet == class{1});
        q1 = prctile(class_data, 25);
        q3 = prctile(class_data, 75);
        iqr = q3 - q1;
        lower_bound = q1 - 1.5*iqr;
        upper_bound = q3 + 1.5*iqr;
        
        % Find outliers for this class
        outliers = class_data(class_data < lower_bound | class_data > upper_bound);
        
        % Plot histogram
        histogram(class_data, 30, 'DisplayName', ['Class ' char(class{1})]);
    end
    title([feat ' Distribution']);
    xlabel('Value');
    ylabel('Frequency');
    legend('show');
    hold off;
end

% Add time information to the analysis
figure('Position', [100 100 1200 400]);

for i = 1:4
    subplot(1,4,i);
    feat = colorFeatures{i};
    
    scatter(data.DurationInHours, data.(feat), 25, data.TriDecisionSet, 'filled');
    title([feat ' Over Time']);
    xlabel('Hours');
    ylabel(feat);
    
    colorbar;
end

