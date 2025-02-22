# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:15:38 2024
@author: bbaha
"""
import pandas as pd
from datetime import timedelta

# Ask for the study ID
study_id = input("Please enter the Study ID: ")
print(f"Processing data for Study ID: {study_id}")

# Load the data from CSV
file_path = r'G:\My Drive\Neuropapers\BBA -BOSoMetre\PatientFiles\PythonShortnered\P23-2024075213-D2.TXT'
data = pd.read_csv(file_path, header=None)

# Assign column names for the original data
data.columns = [
    'InDeviceRawName', 'DateTime', 'R', 'G', 'B', 'C', 
    'RPerc', 'GPerc', 'BPerc', 'CPerc', 'VoltageAverage'
]

# Add the study ID as a new column
data.insert(0, 'InStudyID', study_id)

# Ask if time shift is needed
shift_time = input("Do you want to shift the time for any device? (yes/no): ").lower()

if shift_time == 'yes':
    # Show available devices
    print("\nAvailable devices:")
    print(data['InDeviceRawName'].unique())
    
    # Ask which device to shift
    device_name = input("\nEnter the device name to shift time for: ")
    
    # Create mask for selected device
    mask = data['InDeviceRawName'] == device_name
    if mask.any():
        def convert_datetime(dt_str):
            try:
                # Assuming dt_str is in the YYMMDD_HHMM format
                date_part = dt_str[:6]
                time_part = dt_str[7:]
                # Construct a datetime string in the format 'YYYY/MM/DD HH:MM:SS'
                full_dt = f"20{date_part[:2]}/{date_part[2:4]}/{date_part[4:6]} {time_part[:2]}:{time_part[2:]}:00"
                return pd.to_datetime(full_dt, format='%Y/%m/%d %H:%M:%S')
            except Exception as e:
                print(f"Error converting date: {e}")
                return pd.NaT

        # Convert and modify dates for selected device
        temp_datetime = data.loc[mask, 'DateTime'].apply(convert_datetime)
        time_difference = timedelta(days=36)  # Adjusted shift: 16 days instead of 15
        temp_datetime = temp_datetime + time_difference
        
        # Convert back to desired format: YYMMDD_HHMM
        def datetime_to_original_format(dt):
            return dt.strftime('%y%m%d_%H%M')
        
        data.loc[mask, 'DateTime'] = temp_datetime.apply(datetime_to_original_format)
        print(f"Time shifted for {device_name}")

# Create a batch index for every 60 rows
data['Batch'] = data.index // 60

# Group by Batch
averaged_data = data.groupby('Batch').agg({
    'InStudyID': 'first',
    'InDeviceRawName': 'first',
    'DateTime': 'first',
    'R': 'mean',
    'G': 'mean',
    'B': 'mean',
    'C': 'mean',
    'RPerc': 'mean',
    'GPerc': 'mean',
    'BPerc': 'mean',
    'CPerc': 'mean',
    'VoltageAverage': 'mean'
}).reset_index()

# Save the averaged data to a new CSV file
output_file_path = f'averaged_every_60_study_{study_id}.csv'
averaged_data.to_csv(output_file_path, index=False)
print(f"Data averaged every 60 recordings for Study ID {study_id} saved to {output_file_path}")
