# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:19:32 2024

@author: bbaha
"""

import pandas as pd

# Load the data
file_path = r'G:\My Drive\Neuropapers\BBA -BOSoMetre\PythonWorksheet\ToShorten.csv'  
# Replace with your CSV file path
data = pd.read_csv(file_path)

# Define a function to handle mixed datetime formats
def parse_mixed_datetime(datetime_str):
    try:
        # Try the standard datetime format first
        return pd.to_datetime(datetime_str, format='%Y-%m-%d %H:%M:%S')
    except ValueError:
        try:
            # Try the alternative format YYMMDD_HHMM
            return pd.to_datetime(datetime_str, format='%y%m%d_%H%M')
        except ValueError:
            # If neither works, return NaT
            return pd.NaT

# Apply the function to parse the DateTime column
data['DateTime'] = data['DateTime'].apply(parse_mixed_datetime)

# Identify rows with NaT (failed to parse)
invalid_rows = data[data['DateTime'].isna()]

if not invalid_rows.empty:
    print("Some rows could not be parsed. Saving to invalid_rows.csv for review...")
    invalid_rows.to_csv("invalid_rows.csv", index=False)
else:
    print("All rows were successfully parsed.")

# Proceed to adjust DateTime for each patient
def adjust_datetime(group):
    min_time = group['DateTime'].min()
    group['AdjustedDateTime'] = group['DateTime'] - min_time
    group['DurationInHours'] = group['AdjustedDateTime'].dt.total_seconds() / 3600  # Add duration in hours
    return group

# Group by InStudyID and apply the adjustment
data = data.groupby('InStudyID').apply(adjust_datetime)

# Save the adjusted data
output_file = "adjusted_datetime_with_hours.csv"
data.to_csv(output_file, index=False)
print(f"Adjusted data saved to {output_file}")
