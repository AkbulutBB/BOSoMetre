# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 22:16:45 2024

@author: user
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
input_file = r'G:\My Drive\Neuropapers\Githubstuff\BOSoMetre\MAtlabWorkFolder\classIDSABOS.csv'
data = pd.read_csv(input_file)

# Separate rows with missing values in the 'ExtraBinary' column
missing_values = data[data['infClassIDSA'].isna()]
data_no_missing = data[data['infClassIDSA'].notna()]

# Split the dataset (only rows without missing values)
train_data, test_data = train_test_split(data_no_missing, test_size=0.2, random_state=42)

# Append rows with missing values to the test data
#test_data = pd.concat([test_data, missing_values], ignore_index=True)

# Save the splits to separate files
train_file = 'IDSAtrain_dataset.csv'
test_file = 'IDSAtestnoNA_dataset.csv'

train_data.to_csv(train_file, index=False)
test_data.to_csv(test_file, index=False)

print(f"Training data saved to {train_file}")
print(f"Testing data saved to {test_file}")
