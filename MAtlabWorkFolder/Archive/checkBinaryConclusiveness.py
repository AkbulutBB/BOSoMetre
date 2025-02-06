import pandas as pd

# Load the CSV file
file_path = r'G:\My Drive\Neuropapers\Githubstuff\BOSoMetre\MAtlabWorkFolder\8020IDSAData.csv'
data = pd.read_csv(file_path)

# Create the 'EffectivePrediction' column
data['EffectivePrediction'] = data.apply(
    lambda row: row['PredictedLabels'] if pd.notna(row['ConfidenceLevels']) else float('nan'),
    axis=1
)

# Include NaN as a unique class for TrueLabels
unique_values = data['TrueLabels'].unique()

# Initialize metrics
results = []

# Initialize misclassification counters
misclassifications = {
    'NaN': {'ShouldBe1': 0, 'ShouldBe0': 0},
    '1': {'ShouldBeNaN': 0, 'ShouldBe0': 0},
    '0': {'ShouldBeNaN': 0, 'ShouldBe1': 0},
}

# Iterate through unique values in TrueLabels (including NaN)
for value in unique_values:
    valid_data = data  # Retain all rows

    if pd.isna(value):
        # NaN-specific logic
        tp = ((pd.isna(valid_data['TrueLabels'])) & (pd.isna(valid_data['EffectivePrediction']))).sum()
        fn = ((pd.isna(valid_data['TrueLabels'])) & (~pd.isna(valid_data['EffectivePrediction']))).sum()
        tn = ((~pd.isna(valid_data['TrueLabels'])) & (~pd.isna(valid_data['EffectivePrediction']))).sum()
        fp = ((~pd.isna(valid_data['TrueLabels'])) & (pd.isna(valid_data['EffectivePrediction']))).sum()

        # Count misclassifications involving NaN
        misclassifications['NaN']['ShouldBe1'] += ((pd.isna(valid_data['TrueLabels'])) & (valid_data['EffectivePrediction'] == 1)).sum()
        misclassifications['NaN']['ShouldBe0'] += ((pd.isna(valid_data['TrueLabels'])) & (valid_data['EffectivePrediction'] == 0)).sum()
    else:
        # Non-NaN class logic
        tp = ((valid_data['TrueLabels'] == value) & (valid_data['EffectivePrediction'] == value)).sum()
        fn = ((valid_data['TrueLabels'] == value) & ((valid_data['EffectivePrediction'] != value) & ~pd.isna(valid_data['EffectivePrediction']))).sum()
        tn = ((valid_data['TrueLabels'] != value) & (valid_data['EffectivePrediction'] != value) & ~pd.isna(valid_data['EffectivePrediction'])).sum()
        fp = ((valid_data['TrueLabels'] != value) & (valid_data['EffectivePrediction'] == value)).sum()

        if value == 1:
            # Misclassifications for class 1
            misclassifications['1']['ShouldBeNaN'] += ((valid_data['TrueLabels'] == 1) & (pd.isna(valid_data['EffectivePrediction']))).sum()
            misclassifications['1']['ShouldBe0'] += ((valid_data['TrueLabels'] == 1) & (valid_data['EffectivePrediction'] == 0)).sum()
        elif value == 0:
            # Misclassifications for class 0
            misclassifications['0']['ShouldBeNaN'] += ((valid_data['TrueLabels'] == 0) & (pd.isna(valid_data['EffectivePrediction']))).sum()
            misclassifications['0']['ShouldBe1'] += ((valid_data['TrueLabels'] == 0) & (valid_data['EffectivePrediction'] == 1)).sum()

    # Sensitivity (TP / (TP + FN))
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Specificity (TN / (TN + FP))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Append results for this value
    results.append({
        'Class': value if not pd.isna(value) else 'NaN',
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'True Positives (TP)': tp,
        'False Negatives (FN)': fn,
        'True Negatives (TN)': tn,
        'False Positives (FP)': fp
    })

# Convert results to DataFrame
metrics_df = pd.DataFrame(results)

# Display the metrics
print(metrics_df)

# Save the metrics to a CSV file (optional)
metrics_df.to_csv(r'G:\My Drive\Neuropapers\Githubstuff\BOSoMetre\MAtlabWorkFolder\sensitivity_specificity_metrics.csv', index=False)

# Display misclassification counts
print("\nMisclassification Summary:")
for cls, counts in misclassifications.items():
    print(f"Class {cls}:")
    for key, count in counts.items():
        print(f"  {key}: {count}")
