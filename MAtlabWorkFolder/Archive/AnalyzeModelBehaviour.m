% Feature importance and diagnostic analysis
function analyze_model_behavior(features, binary_labels, window_size)
    % Create windowed features
    windowed_features = create_rolling_window(features, window_size);
    windowed_labels = binary_labels(window_size:end);
    
    % Standardize all features
    [windowed_features_scaled, ~, ~] = zscore(windowed_features);
    
    % Train a model on all data for analysis
    model = fitcensemble(windowed_features_scaled, windowed_labels, 'Method', 'Bag');
    
    % Get feature importance scores
    imp = predictorImportance(model);
    
    % Reshape importance scores to match original features and time windows
    imp_reshaped = reshape(imp, 4, window_size);
    
    % Plot feature importance heatmap
    figure;
    heatmap({'RPerc', 'GPerc', 'BPerc', 'CPerc'}, ...
            cellstr("T-" + string((0:window_size-1))), imp_reshaped');
    title(sprintf('Feature Importance Across Time Windows (Window Size: %d)', window_size));
    xlabel('Features');
    ylabel('Time Steps');
    
    % Plot temporal patterns for each class
    figure;
    for feat = 1:4
        subplot(2,2,feat);
        feature_data = features(:,feat);
        
        % Calculate mean patterns for each class
        clean_pattern = mean(reshape(feature_data(binary_labels==0), [], window_size), 1);
        infected_pattern = mean(reshape(feature_data(binary_labels==1), [], window_size), 1);
        
        % Plot patterns
        plot(0:window_size-1, clean_pattern, 'b-', 'LineWidth', 2, 'DisplayName', 'Clean');
        hold on;
        plot(0:window_size-1, infected_pattern, 'r-', 'LineWidth', 2, 'DisplayName', 'Infected');
        hold off;
        
        feature_names = {'RPerc', 'GPerc', 'BPerc', 'CPerc'};
        title(sprintf('%s Temporal Pattern', feature_names{feat}));
        xlabel('Time Step');
        ylabel('Value');
        legend('Location', 'best');
        grid on;
    end
    sgtitle(sprintf('Average Temporal Patterns (Window Size: %d)', window_size));
end

% Run analysis for a specific window size (e.g., 24 since it performed best)
analyze_model_behavior(features, binary_labels, 24);