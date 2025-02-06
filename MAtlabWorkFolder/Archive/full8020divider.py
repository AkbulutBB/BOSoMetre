# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 19:20:02 2024

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 7 22:16:45 2024

@author: user
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
input_file = r'G:\My Drive\Neuropapers\Githubstuff\BOSoMetre\MAtlabWorkFolder\classIDSABOS.csv'
data = pd.read_csv(input_file)

# Split the entire dataset, including rows with missing values
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the splits to separate files
train_file = 'IDSAfulltrain_dataset.csv'
test_file = 'IDSAfulltest_dataset.csv'

train_data.to_csv(train_file, index=False)
test_data.to_csv(test_file, index=False)

print(f"Training data saved to {train_file}")
print(f"Testing data saved to {test_file}")
