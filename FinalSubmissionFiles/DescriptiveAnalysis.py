import pandas as pd

def format_age(age_months):
    """
    Converts an age value (in months) to a formatted string.
    
    - If the age is less than 12 months, returns the age in months.
    - Otherwise, converts the age into years and remaining months.
    """
    if age_months < 12:
        return f"{int(age_months)} months"
    else:
        years = int(age_months // 12)
        remaining_months = int(age_months % 12)
        if remaining_months == 0:
            return f"{years} years"
        else:
            return f"{years} years and {remaining_months} months"

def descriptive_statistics_age(df):
    """
    Computes descriptive statistics for the 'Age(months)' column.
    
    Statistics include:
    - Mean
    - Median
    - Standard Deviation
    - Range (Minimum and Maximum)
    - Interquartile Range (IQR)
    
    The function formats age values to report those below 12 months in months,
    and those above in a 'years and months' format.
    """
    age_data = df["Age(months)"]
    
    mean_age = age_data.mean()
    median_age = age_data.median()
    std_age = age_data.std()
    min_age = age_data.min()
    max_age = age_data.max()
    q1 = age_data.quantile(0.25)
    q3 = age_data.quantile(0.75)
    iqr = q3 - q1
    
    print("Descriptive Statistics for Patient Age:")
    print(f"Mean Age: {format_age(mean_age)}")
    print(f"Median Age: {format_age(median_age)}")
    print(f"Standard Deviation: {std_age:.2f} months")
    print(f"Range: {format_age(min_age)} - {format_age(max_age)}")
    print(f"IQR (Interquartile Range): {format_age(iqr)} (25th percentile: {format_age(q1)}, 75th percentile: {format_age(q3)})")
    print()

def descriptive_statistics_numeric(df, column, label):
    """
    Computes descriptive statistics for a numeric column.
    
    The statistics include mean, median, standard deviation, range, and IQR.
    """
    data = df[column]
    
    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()
    min_val = data.min()
    max_val = data.max()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    
    print(f"Descriptive Statistics for {label}:")
    print(f"Mean {label}: {mean_val:.2f}")
    print(f"Median {label}: {median_val:.2f}")
    print(f"Standard Deviation: {std_val:.2f}")
    print(f"Range: {min_val} - {max_val}")
    print(f"IQR (Interquartile Range): {iqr:.2f} (25th percentile: {q1:.2f}, 75th percentile: {q3:.2f})")
    print()

def analyze_categorical(df, column, label):
    """
    Analyzes a categorical column by computing frequency counts and percentages.
    
    Outputs the count and relative frequency for each category.
    """
    counts = df[column].value_counts()
    percentages = df[column].value_counts(normalize=True) * 100

    print(f"Frequency Analysis for {label}:")
    for category in counts.index:
        print(f"{category}: {counts[category]} ({percentages[category]:.2f}%)")
    print()

if __name__ == '__main__':
    # Replace 'patient_data.csv' with the path to your CSV file containing the patient data.
    csv_file = "BOSoMetreData - DataSheetForPaper.csv"
    # Load the dataset from the CSV file.
    df = pd.read_csv(csv_file)
    
    # Descriptive analysis for Age (in months)
    descriptive_statistics_age(df)
    
    # Descriptive analysis for Total Recording Days (numeric)
    descriptive_statistics_numeric(df, "Total Recording Days", "Total Recording Days")
    
    # Frequency analysis for categorical variables:
    # Patient Type
    patient_type_col = "PatientType(1-Inf 2-Bleed 3-PostopBleed 4-AcuteHydrocephalusWithoutInfection)"
    analyze_categorical(df, patient_type_col, "Patient Type")
    
    # EVD Type (expected values: 'ArgiFix' and 'HP')
    analyze_categorical(df, "EVDType", "EVD Type")
    
    # Sex
    analyze_categorical(df, "Sex", "Sex")
