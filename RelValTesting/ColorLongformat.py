# -*- coding: utf-8 -*-
"""
Sensor Data Analysis and Classification
=======================================

This script performs the following tasks:
1. Loads sensor data from a CSV file.
2. Restructures the data from wide to long format.
3. Visualizes the data using box plots and histograms.
4. Prepares the data for classification.
5. Trains an SVM classifier to differentiate between two fluids.
6. Evaluates the classifier's performance.

Author: Your Name
Date: YYYY-MM-DD
"""

# ===========================
# 1. Import Necessary Libraries
# ===========================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ===========================
# 2. Load Your Data
# ===========================
# Define the file path to your CSV file
file_path = r"C:\Users\bbaha\Downloads\SensorDataTesting.xlsx - Validity Data (2).csv"

# Load the data into a pandas DataFrame
try:
    data = pd.read_csv(file_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"File not found at the specified path: {file_path}")
    # Exit the script if file not found
    exit()

# Display the first few rows of the data
print("\nOriginal Data:")
print(data.head())

# ===========================
# 3. Restructure Data to Long Format
# ===========================
# Define the identifier variables and the value variables
id_vars = ['TestName', 'DateTime']  # Adjust these if your CSV has different column names
value_vars = ['Red', 'Green', 'Blue', 'Clear']  # Columns to melt

# Melt the DataFrame to long format
long_data = pd.melt(data, id_vars=id_vars, value_vars=value_vars,
                    var_name='Color', value_name='Value')

# Display the first few rows of the transformed data
print("\nLong Format Data:")
print(long_data.head())

# ===========================
# 4. Visualize the Data
# ===========================

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# ---- a. Box Plot ----
plt.figure(figsize=(12, 10))
sns.boxplot(x='Color', y='Value', hue='TestName', data=long_data)
plt.title('Box Plot of Sensor Readings by Color and Fluid Type')
plt.xlabel('Color Channel')
plt.ylabel('Sensor Reading')
plt.legend(title='Fluid Type')
plt.tight_layout()
plt.show()

# ---- b. Histogram ----
# Create separate histograms for each color channel
g = sns.FacetGrid(long_data, col="Color", hue="TestName", col_wrap=2, sharex=False, sharey=False)
g.map(sns.histplot, "Value", kde=False, bins=30, alpha=0.6)
g.add_legend()
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Histogram of Sensor Readings by Color and Fluid Type')
plt.show()

# ===========================
# 5. Prepare Data for Classification
# ===========================

# Pivot the long_data back to wide format for classification
wide_data = long_data.pivot_table(index=['TestName', 'DateTime'], columns='Color', values='Value').reset_index()

# Display the first few rows of the wide format data
print("\nWide Format Data for Classification:")
print(wide_data.head())

# Define predictors (features) and response (label)
X = wide_data[['Red', 'Green', 'Blue', 'Clear']]
y = wide_data['TestName']

# Encode labels if they are categorical (e.g., 'Fluid1', 'Fluid2')
# Replace 'Fluid1' and 'Fluid2' with 0 and 1 respectively
label_mapping = {'Fluid1': 0, 'Fluid2': 1}
y = y.map(label_mapping)

# Check for any unmapped labels
if y.isnull().any():
    print("\nWarning: Some labels were not mapped. Please check the 'TestName' column.")
    print(y[y.isnull()])
    # Optionally, exit or handle the unmapped labels
    exit()

# ===========================
# 6. Train a Support Vector Machine (SVM) Classifier
# ===========================

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVM classifier with a linear kernel
classifier = SVC(kernel='linear', probability=True)  # probability=True for probability estimates

# Train the classifier on the training data
classifier.fit(X_train, y_train)
print("\nSVM classifier trained successfully.")

# ===========================
# 7. Evaluate the Classifier
# ===========================

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Generate a classification report
class_report = classification_report(y_test, y_pred, target_names=['Fluid1', 'Fluid2'])
print("\nClassification Report:")
print(class_report)

# Calculate overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Optional: Visualize the Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fluid1', 'Fluid2'],
            yticklabels=['Fluid1', 'Fluid2'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


