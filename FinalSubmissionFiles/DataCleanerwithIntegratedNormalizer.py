import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import stats
from datetime import datetime

class CSFDataNormalizer:
    def __init__(self, voltage_exponent: float = 2.2):
        """
        Initialize the normalizer with LED intensity decay exponent.
        
        According to general LED decay formulas, the relative light intensity
        can be approximated by (V_calib / V_current) ^ exponent. Here, a typical
        exponent might be near 2 (though 2.2 is also used in certain photometric
        contexts) [1].
        
        References:
        [1] Cinquegrani, F., & Marconi, M. (2017). Photometric Testing of LED 
            Lighting for Healthcare Applications. Lighting Research & Technology, 49(3), 1–15.
        """
        self.voltage_exponent = voltage_exponent
        self.base_values: Dict[int, Dict[str, float]] = {}
        self.calib_voltages: Dict[int, float] = {}
        
    def set_base_values(self, patient_id: int, r_base: float, g_base: float, 
                        b_base: float, c_base: float, calib_voltage: float):
        """Store base RGBC values and calibration voltage for a given patient."""
        self.base_values[patient_id] = {
            'R': r_base,
            'G': g_base,
            'B': b_base,
            'C': c_base
        }
        self.calib_voltages[patient_id] = calib_voltage
    
    def load_base_values_from_master_sheet(self, master_file: str):
        """
        Load base values and calibration voltages from the master CSV file.
        
        Expects columns: [InStudyPatientID, RBase, GBase, BBase, CBase, GuessCalibVolts].
        """
        try:
            master_df = pd.read_csv(master_file, encoding='utf-8-sig')
            master_df['InStudyPatientID'] = pd.to_numeric(master_df['InStudyPatientID'], errors='coerce')
            
            required_columns = ['InStudyPatientID', 'RBase', 'GBase', 'BBase', 'CBase', 'GuessCalibVolts']
            valid_rows = master_df.dropna(subset=required_columns)
            
            for _, row in valid_rows.iterrows():
                self.set_base_values(
                    int(row['InStudyPatientID']),
                    float(row['RBase']),
                    float(row['GBase']),
                    float(row['BBase']),
                    float(row['CBase']),
                    float(row['GuessCalibVolts'])
                )
            
            print(f"Loaded base values and calibration voltages for {len(valid_rows)} patients")
        except Exception as e:
            print(f"Error loading master sheet: {str(e)}")
            raise
    
    def voltage_compensation_factor(self, current_voltage: float, patient_id: int) -> float:
        """
        Calculate voltage compensation factor using LED intensity decay.
        
        If `current_voltage` is missing (NaN) or zero, default to the patient's
        calibration voltage.
        """
        if patient_id not in self.calib_voltages:
            raise ValueError(f"No calibration voltage found for patient {patient_id}")
        
        calib_voltage = self.calib_voltages[patient_id]
        
        if pd.isna(current_voltage) or current_voltage == 0:
            current_voltage = calib_voltage
        
        return (calib_voltage / current_voltage) ** self.voltage_exponent
    
    def normalize_value(self, value: float, base_value: float, 
                        voltage: float, patient_id: int) -> float:
        """
        Normalize a single channel's value using:
          - the patient-specific base value
          - the voltage compensation factor
        
        The normalized value is scaled to 100% if it matches the base reading
        under calibration voltage.
        """
        comp_factor = self.voltage_compensation_factor(voltage, patient_id)
        return (value * comp_factor / base_value) * 100
    
    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize all RGBC values in a DataFrame using the stored base values.
        
        The output columns are RNormalized, GNormalized, BNormalized, CNormalized.
        Missing or zero voltages are automatically handled using calibration voltages.
        """
        df_norm = df.copy()
        
        for channel in ['R', 'G', 'B', 'C']:
            norm_col = f'{channel}Normalized'
            df_norm[norm_col] = np.nan
            
            for patient_id in df_norm['InStudyID'].unique():
                if patient_id not in self.base_values:
                    print(f"Warning: No base values for patient {patient_id}")
                    continue
                
                patient_mask = df_norm['InStudyID'] == patient_id
                base_value = self.base_values[patient_id][channel]
                
                df_norm.loc[patient_mask, norm_col] = df_norm.loc[patient_mask].apply(
                    lambda row: self.normalize_value(
                        row[channel],
                        base_value,
                        row['VoltageAverage'],
                        patient_id
                    ),
                    axis=1
                )
        
        return df_norm

def mad_based_outlier_detection(values: np.array, window_size: int = 8, 
                                mad_threshold: float = 5.0) -> np.array:
    """
    Detect outliers using Median Absolute Deviation (MAD) within a sliding window.
    
    Outlier removal is done by comparing each data point to the median
    in a neighborhood of size `window_size`. Points with a MAD-based Z-score
    above `mad_threshold` are set to NaN.
    
    References:
    [2] Leys, C., et al. (2013). Detecting outliers: Do not use standard deviation 
        around the mean, use absolute deviation around the median. Journal of Experimental Social Psychology, 49(4), 764–766.
    """
    clean_values = values.copy()
    valid_mask = ~np.isnan(values)
    
    if sum(valid_mask) < window_size:
        return clean_values
        
    for i in range(len(values) - window_size + 1):
        window = values[i:i+window_size]
        valid_window = ~np.isnan(window)
        
        if sum(valid_window) < 3:
            continue
        
        window_median = np.nanmedian(window)
        abs_deviations = np.abs(window - window_median)
        window_mad = np.nanmedian(abs_deviations)
        
        if window_mad == 0:
            window_mad = np.nanmean(abs_deviations) or 0.1
        
        z_scores = abs_deviations / window_mad
        outliers = z_scores > mad_threshold
        
        clean_values[i:i+window_size][outliers & valid_window] = np.nan
    
    return clean_values

def filter_out_of_range_readings(df: pd.DataFrame,
                                 normalizer: CSFDataNormalizer,
                                 lower_factor: float = 0.5,
                                 upper_factor: float = 1.5
                                ) -> pd.DataFrame:
    """
    Remove rows where raw RGBC readings are outside [50%, 150%] of the patient's base values.
    """
    keep_mask = [True] * len(df)

    for idx, row in df.iterrows():
        patient_id = row['InStudyID']
        
        if patient_id not in normalizer.base_values:
            continue

        out_of_range = False
        for channel in ['R', 'G', 'B', 'C']:
            base_val = normalizer.base_values[patient_id][channel]
            raw_val = row[channel]

            if raw_val < (lower_factor * base_val) or raw_val > (upper_factor * base_val):
                out_of_range = True
                break
        
        if out_of_range:
            keep_mask[idx] = False

    df_filtered = df[keep_mask].copy()
    
    removed_count = len(df) - len(df_filtered)
    print(f"Removed {removed_count} rows out of range ([{int(lower_factor*100)}%, {int(upper_factor*100)}%]).")
    
    return df_filtered

def mad_based_winsorization(df: pd.DataFrame, columns: List[str], 
                            mad_multiplier: float = 5.0) -> pd.DataFrame:
    """
    Winsorize data based on MAD, handling each patient separately.
    """
    df_cleaned = df.copy()
    
    if 'InStudyID' not in df.columns:
        return df_cleaned
    
    for patient_id in df['InStudyID'].unique():
        patient_mask = df['InStudyID'] == patient_id
        for col in columns:
            valid_data = df_cleaned.loc[patient_mask, col].dropna()
            if len(valid_data) > 0:
                median = np.median(valid_data)
                mad = np.median(np.abs(valid_data - median))
                
                if mad == 0:
                    mad = np.mean(np.abs(valid_data - median)) or 0.1
                
                lower_bound = median - mad_multiplier * mad
                upper_bound = median + mad_multiplier * mad
                
                df_cleaned.loc[patient_mask, col] = df_cleaned.loc[patient_mask, col].clip(
                    lower=lower_bound, upper=upper_bound
                )
    
    return df_cleaned

def percentile_winsorize_data(df: pd.DataFrame, columns: List[str], 
                              limits: Tuple[float, float] = (0.05, 0.95)) -> pd.DataFrame:
    """
    Winsorize data to specified percentiles, handling NaN values.
    """
    df_cleaned = df.copy()
    for col in columns:
        valid_data = df_cleaned[col].dropna()
        if len(valid_data) > 0:
            lower_bound = np.percentile(valid_data, limits[0] * 100)
            upper_bound = np.percentile(valid_data, limits[1] * 100)
            df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
    return df_cleaned

def standardize_datetime(df: pd.DataFrame, datetime_column: str = 'DateTime') -> pd.DataFrame:
    """
    Standardize datetime formats to YYMMDD_HHMM. 
    Rows with unparseable datetime values are dropped.
    """
    df_cleaned = df.copy()
    
    def convert_datetime(dt_str):
        try:
            dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            try:
                dt = datetime.strptime(dt_str, '%y%m%d_%H%M')
            except ValueError:
                print(f"Warning: Could not parse datetime: {dt_str}")
                return None
        return dt.strftime('%y%m%d_%H%M')
    
    df_cleaned[datetime_column] = df_cleaned[datetime_column].apply(convert_datetime)
    invalid_dates = df_cleaned[datetime_column].isna()
    if invalid_dates.any():
        print(f"Removed {invalid_dates.sum()} rows with invalid dates")
        df_cleaned = df_cleaned.dropna(subset=[datetime_column])
    
    return df_cleaned

def sort_patient_data(df: pd.DataFrame, patient_column: str = 'InStudyID', 
                      batch_column: str = 'Batch') -> pd.DataFrame:
    """
    Sort data by patient ID and batch number, then display summary statistics.
    """
    df_sorted = df.sort_values([patient_column, batch_column])
    
    print("\nSorting Statistics:")
    patient_counts = df_sorted[patient_column].value_counts().sort_index()
    print(f"\nPatients found: {len(patient_counts)}")
    for patient_id, count in patient_counts.items():
        batch_range = df_sorted[df_sorted[patient_column] == patient_id][batch_column]
        print(f"Patient {patient_id}: {count} rows, Batch range: {batch_range.min()}-{batch_range.max()}")
    
    return df_sorted.reset_index(drop=True)

def split_by_device_and_save(df: pd.DataFrame, device_column: str = 'DeviceID'):
    """
    Create device-specific CSV files from the DataFrame, splitting on `device_column`.
    
    Files will be named 'processed_csf_data_device_<deviceID>.csv'.
    """
    if device_column not in df.columns:
        print(f"Device column '{device_column}' not found in DataFrame. Skipping split.")
        return
    
    for device_id, group_df in df.groupby(device_column):
        output_file_device = f"processed_csf_data_device_{device_id}.csv"
        group_df.to_csv(output_file_device, index=False)
        print(f"Saved {len(group_df)} rows to {output_file_device}")

def process_csf_data(input_file: str,
                     master_file: str,
                     output_file: str,
                     voltage_exponent: float = 3.0,
                     use_mad: bool = True,
                     split_by_device: bool = False,
                     device_column: str = 'DeviceID') -> pd.DataFrame:
    """
    End-to-end pipeline for:
      1) Sorting
      2) Filtering
      3) Normalizing
      4) Outlier detection
      5) Winsorizing
      6) Date-time standardization
      7) Optionally splitting by device type.
    """
    print("Starting CSF data processing pipeline...")
    
    # Step 1: Load data
    print("Loading data files...")
    df = pd.read_csv(input_file)
    
    # Step 2: Sort data
    print("\nSorting data...")
    df_sorted = sort_patient_data(df)  # sorts by InStudyID, Batch, etc.
    
    # Step 3: Initialize normalizer and load base values
    print("\nInitializing normalizer...")
    normalizer = CSFDataNormalizer(voltage_exponent=voltage_exponent)
    normalizer.load_base_values_from_master_sheet(master_file)
    
    # Step 4: Filter raw readings outside 50–150% of base values
    print("\nFiltering raw RGBC readings outside [50%, 150%] of base values...")
    df_filtered = filter_out_of_range_readings(df_sorted, normalizer)
    
    # Step 5: Normalize the now-filtered DataFrame
    print("\nNormalizing RGBC values...")
    df_normalized = normalizer.normalize_dataframe(df_filtered)
    
    # Step 6: Standardize datetime format
    print("\nStandardizing datetime format...")
    df_datetime = standardize_datetime(df_normalized)
    
    # Step 7: Outlier detection on the normalized columns
    print("\nPerforming outlier detection...")
    normalized_columns = ['RNormalized', 'GNormalized', 'BNormalized', 'CNormalized']
    df_cleaned = df_datetime.copy()
    for col in normalized_columns:
        df_cleaned[col] = mad_based_outlier_detection(df_cleaned[col].values)
    
    # Step 8: Winsorization (either MAD-based or percentile-based)
    print("\nApplying winsorization...")
    if use_mad:
        df_cleaned = mad_based_winsorization(df_cleaned, normalized_columns)
    else:
        df_cleaned = percentile_winsorize_data(df_cleaned, normalized_columns)
    
    # Print final statistics for each normalized channel
    print("\nFinal Statistics:")
    for col in normalized_columns:
        raw_channel = col.replace('Normalized', '')
        if raw_channel in df_filtered.columns:
            raw_stats = df_filtered[raw_channel].describe()
            normalized_stats = df_cleaned[col].describe()
            print(f"\nChannel {col}:")
            print(f"Original (raw) range: [{raw_stats['min']:.2f}, {raw_stats['max']:.2f}] (post 50–150% filter)")
            print(f"Normalized final range: [{normalized_stats['min']:.2f}, {normalized_stats['max']:.2f}]")
            print(f"NaN count in final: {df_cleaned[col].isna().sum()}")
    
    # New step: Report missing values by patient (overall and per channel) with percentages.
    print("\nMissing Value Statistics by Patient:")
    overall_total_rows = df_cleaned.shape[0]
    overall_missing = df_cleaned[normalized_columns].isna().any(axis=1).sum()
    overall_available = overall_total_rows - overall_missing

    print(f"Total rows in final dataset: {overall_total_rows}")
    print(f"Overall missing rows: {overall_missing} ({(overall_missing/overall_total_rows)*100:.2f}%)")
    print(f"Overall available rows: {overall_available} ({(overall_available/overall_total_rows)*100:.2f}%)\n")
    
    print(f"{'Patient':<10}{'Total Rows':<15}{'Missing (Overall)':<30}{'Available (Overall)':<30}")
    for patient_id, group in df_cleaned.groupby('InStudyID'):
        total_rows = group.shape[0]
        missing_overall = group[normalized_columns].isna().any(axis=1).sum()
        available_overall = total_rows - missing_overall
        print(f"{patient_id:<10}{total_rows:<15}{str(missing_overall) + ' (' + f'{(missing_overall/total_rows)*100:.2f}%' + ')':<30}{str(available_overall) + ' (' + f'{(available_overall/total_rows)*100:.2f}%' + ')':<30}")
    
    print("\nMissing Value Breakdown by Patient and Channel:")
    for patient_id, group in df_cleaned.groupby('InStudyID'):
        total_rows = group.shape[0]
        print(f"\nPatient {patient_id} (Total Rows: {total_rows}):")
        for channel in normalized_columns:
            missing = group[channel].isna().sum()
            available = total_rows - missing
            missing_pct = (missing / total_rows) * 100 if total_rows > 0 else 0
            available_pct = (available / total_rows) * 100 if total_rows > 0 else 0
            print(f"  {channel}: Missing {missing} ({missing_pct:.2f}%), Available {available} ({available_pct:.2f}%)")
    
    # Step 9: Save results
    if split_by_device:
        print("\nSplitting by device and saving device-specific CSV files...")
        split_by_device_and_save(df_cleaned, device_column=device_column)
    else:
        print(f"\nSaving processed data to {output_file}")
        df_cleaned.to_csv(output_file, index=False)
    
    return df_cleaned

if __name__ == "__main__":
    try:
        # Process data with sample arguments.
        input_file = "raw_csf_data.csv"
        master_file = "MasterSheet.csv"
        output_file = "processed_csf_data.csv"
        
        processed_df = process_csf_data(
            input_file=input_file,
            master_file=master_file,
            output_file=output_file,
            voltage_exponent=0.1,
            use_mad=True,
            split_by_device=False,       # Change to True if you want device-specific files
            device_column='DeviceNo'       # Adjust if your device column is named differently
        )
        
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
