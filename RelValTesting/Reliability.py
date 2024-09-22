# First, install the pingouin library if you haven't already
# pip install pingouin

import pandas as pd
import pingouin as pg

# Sample data setup for Red, Green, Blue, and Clear channels
# Assuming you have already read your data from a file or loaded it as a pandas DataFrame

# Example data: Replace this with actual data
data = {
    'Test-Name': ['Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-1','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-2','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3','Rel-Test-3'],
    'Red': [67,65,66,66,65,67,65,65,65,64,67,69,70,68,69,69,69,69,69,70,68,68,69,70,70,68,69,69,70,68,68,68,67,69,68,69,69,69,68,69,69,68,69,70,68,68,69,69,69,69,69,67,67,69,69,70,68,69,68,68,68,69,69,67,68,68,69,68,68,69,69,69,70,68,68,68,69,69,68,68,69,69,68,68,69,68,67,67,68,69,68,69,69,69,68,69,68,69,68,69,69,62,61,62,61,61,61,61,60,59,59,60,61,62,62,62,62,61,61,61,61,60,61,61,61,61,61,61,61,61,62,60,60,60,61,61,62,61,61,63,62,61,62,60,60,61,62,62,63,61,62,61,61,61,61,59,60,61,62,62,62,61,61,61,61,62,59,60,60,62,62,62,62,62,62,61,63,62,61,60,62,62,61,61,61,62,61,62,62,62,60,61,62,61,62,69,67,66,65,67,67,67,68,68,67,67,67,67,67,67,67,67,67,67,68,67,68,67,67,67,66,66,67,67,68,68,68,68,67,67,68,68,68,67,68,67,67,67,67,67,69,67,67,66,66,67,68,67,67,67,67,67,68,68,67,67,67,67,67,67,68,67,67,67,67,67,67,66,68,67,67,67,67,68,67,67,68,67,67,66,67,67,68,67,67,69,68,67,68,67],  # Add all your Red channel data here
    'Green': [71,71,71,71,71,69,70,70,71,71,76,75,75,76,76,75,76,76,73,75,76,75,76,75,75,76,75,75,75,74,75,75,75,75,76,75,75,75,75,75,75,75,76,75,76,75,75,76,74,74,75,75,75,75,75,75,75,75,76,76,75,74,75,75,75,75,75,75,75,75,75,74,73,75,75,74,75,75,75,75,75,75,76,74,73,75,75,75,75,75,75,75,75,75,75,74,73,75,75,75,75,66,67,66,66,67,67,65,67,67,67,66,67,66,66,66,67,67,67,65,67,67,67,67,67,66,67,67,67,66,65,66,67,67,67,67,67,67,67,67,67,67,66,67,67,67,67,67,67,67,67,67,67,66,67,67,67,67,67,67,67,67,67,67,67,65,68,67,67,67,67,67,67,67,67,67,67,65,67,67,67,67,67,67,67,67,67,67,66,65,68,67,67,67,67,72,73,73,74,73,73,74,73,73,74,73,74,73,74,73,74,73,73,74,74,73,73,73,71,72,73,73,73,74,73,73,73,73,73,73,71,74,73,74,73,73,73,74,73,73,71,71,72,73,73,73,73,73,73,74,74,73,73,72,73,74,73,74,73,73,73,73,73,72,72,72,71,73,73,74,73,74,73,73,73,72,71,71,74,73,73,74,73,74,73,73,73,73,72,73],  # Add all your Green channel data here
    'Blue': [81,81,81,81,80,81,81,80,81,81,85,86,86,85,85,85,85,85,86,85,85,85,85,86,86,85,84,84,83,86,86,85,85,85,85,85,85,86,84,83,84,85,85,86,85,86,85,83,85,85,85,86,86,85,85,85,85,85,85,83,86,85,86,85,86,85,85,85,85,85,83,85,86,85,85,86,85,85,85,85,85,85,83,85,85,85,85,85,85,85,86,85,85,85,83,85,85,84,85,85,84,77,77,77,77,75,75,78,77,77,77,78,77,78,78,78,76,75,75,78,77,77,78,77,77,78,77,77,76,77,78,78,77,78,77,77,77,78,77,77,76,76,77,77,77,77,77,77,77,76,75,75,76,77,78,78,77,77,77,77,77,77,78,76,75,78,77,77,77,77,77,77,77,77,75,76,75,77,78,78,78,77,78,77,77,77,75,77,77,78,77,78,77,77,77,82,83,82,82,83,83,82,83,82,81,81,81,83,82,82,82,82,83,82,82,83,81,83,83,82,83,83,82,82,83,82,83,81,81,83,83,82,83,82,82,83,83,82,81,81,83,83,82,83,83,83,83,83,82,82,81,81,81,82,83,82,83,82,83,83,83,81,81,82,83,83,83,83,82,82,83,82,83,81,81,83,83,83,82,83,83,82,83,82,83,81,81,81,83,83],  # Add all your Blue channel data here
    'Clear': [209,208,204,203,208,209,209,209,207,208,219,219,219,220,215,216,219,220,219,221,219,220,220,219,214,215,219,219,221,219,219,220,220,220,219,220,216,214,219,220,218,220,219,218,215,214,215,219,220,218,220,218,218,219,218,215,215,216,214,218,218,220,218,220,218,220,219,215,215,214,218,219,218,219,220,218,218,220,214,214,213,218,218,218,219,219,219,219,219,218,217,214,215,217,220,219,220,220,220,219,220,195,195,196,196,199,200,198,199,200,200,198,200,198,194,194,198,200,200,198,198,200,198,199,199,194,196,200,199,199,198,198,199,198,198,200,195,194,196,199,199,198,199,199,199,199,199,199,194,199,200,200,199,200,198,198,199,199,199,198,194,194,198,199,200,198,198,199,199,198,198,199,196,196,199,198,199,200,198,198,198,198,198,199,195,199,199,200,200,198,198,198,199,200,194,213,213,214,213,213,213,214,208,209,212,213,212,212,213,214,213,213,212,213,209,212,212,213,213,214,212,212,214,213,212,214,212,212,214,213,212,213,213,213,214,213,213,209,212,214,213,213,214,213,212,212,212,212,209,213,212,214,212,214,213,214,213,213,213,213,207,213,212,214,212,212,213,213,214,213,212,213,207,212,214,212,212,213,213,213,212,213,212,209,207,212,212,214,212,213]  # Add all your Clear channel data here
}
# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Check the shape of the original data
print(f"Original data shape: {df.shape}")

# Reshape the data for ICC analysis - we need to convert the DataFrame into long format
df_melted = pd.melt(df, id_vars=['Test-Name'], value_vars=['Red', 'Green', 'Blue', 'Clear'], 
                    var_name='Channel', value_name='Score')

# Check the shape of the melted DataFrame
print(f"Melted data shape: {df_melted.shape}")

# Check if any groups have fewer than 5 observations
grouped_data_counts = df_melted.groupby(['Test-Name', 'Channel']).size()
print("\nGrouped data counts (before filtering):")
print(grouped_data_counts)

# Filter valid groups with at least 5 observations
valid_groups = df_melted.groupby(['Test-Name', 'Channel']).filter(lambda x: len(x) >= 5)

# Check the size of valid groups to ensure they are retained correctly
grouped_valid_counts = valid_groups.groupby(['Test-Name', 'Channel']).size()
print("\nGrouped data counts (after filtering):")
print(grouped_valid_counts)

# If we still have enough valid data, proceed with the ICC calculation
if valid_groups.shape[0] > 0:
    try:
        icc_results_all_channels = pg.intraclass_corr(data=valid_groups, targets='Test-Name', raters='Channel', ratings='Score')
        print("\nICC results (All Channels):")
        print(icc_results_all_channels)
    except AssertionError as e:
        print(f"ICC calculation failed due to assertion error: {e}")
else:
    print("Not enough data to compute ICC for all groups.")


# Function to calculate ICC for each channel separately
def calculate_icc_for_channel(channel):
    # Subset the data to just the specific channel
    df_channel = df[['Test-Name', channel]]
    
    # Reshape data for ICC analysis
    df_channel_melted = df_channel.melt(id_vars=['Test-Name'], value_vars=[channel], var_name='Channel', value_name='Score')
    
    # Debugging: print out shape and summary of data for the channel
    print(f"\n{channel} channel data shape: {df_channel_melted.shape}")
    print(f"\n{channel} channel data summary:")
    print(df_channel_melted.describe())
    
    # Calculate ICC for the specific channel
    try:
        if df_channel_melted.shape[0] >= 5:
            icc_result_channel = pg.intraclass_corr(data=df_channel_melted, targets='Test-Name', raters='Channel', ratings='Score')
            return icc_result_channel
        else:
            return f"Not enough data for {channel} channel to calculate ICC."
    except AssertionError as e:
        return f"ICC calculation failed for {channel} due to assertion error: {e}"


# Calculate ICC for Red channel
icc_red = calculate_icc_for_channel('Red')
print("\nICC for Red Channel:")
print(icc_red)

# Calculate ICC for Green channel
icc_green = calculate_icc_for_channel('Green')
print("\nICC for Green Channel:")
print(icc_green)

# Calculate ICC for Blue channel
icc_blue = calculate_icc_for_channel('Blue')
print("\nICC for Blue Channel:")
print(icc_blue)

# Calculate ICC for Clear channel
icc_clear = calculate_icc_for_channel('Clear')
print("\nICC for Clear Channel:")
print(icc_clear)