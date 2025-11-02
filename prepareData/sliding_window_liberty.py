import os
import pandas as pd
from pathlib import Path
from helper import fixedSize_window, structure_log
import numpy as np

# --- Script Configuration for Liberty Dataset ---

# 1. Dataset Paths and Parameters
data_dir = r'/home/koganrath/Personal/coding/loghub-logs/liberty'
log_name = "liberty2"
output_dir = data_dir
log_format = '<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>'

# 2. Data Selection and Splitting Parameters
start_line = 40000000
end_line = 45000000
train_ratio = 0.8
validation_ratio = 0.1
chunk_size = 5000000

# 3. Windowing Parameters
window_size = 100
step_size = 100

# 4. Resampling Parameters
TARGET_ANOMALY_PERCENTAGE = 0.15


def perform_chronological_split():
    """
    Step 1: Performs a strict chronological split of the raw log data into
    an imbalanced train set, and the final validation and test sets.
    """
    structured_log_path = os.path.join(output_dir, f'{log_name}_structured.csv')
    if not os.path.exists(structured_log_path):
        print("--- Step 1: Structuring Raw Log File ---")
        structure_log(data_dir, output_dir, log_name, log_format, start_line=start_line, end_line=end_line)
    else:
        print(f"--- Step 1: Structured file already exists. Skipping structuring. ---")

    print("\n--- Step 2: Performing Chronological Split ---")
    
    with open(structured_log_path, 'r', encoding='latin-1') as f:
        total_lines = sum(1 for line in f) - 1
    
    train_boundary = int(total_lines * train_ratio)
    validation_boundary = int(total_lines * (train_ratio + validation_ratio))

    print("\nData split boundaries (row index):")
    print(f"  Training set ends at:      {train_boundary}")
    print(f"  Validation set ends at:    {validation_boundary}")
    print(f"  Test set starts at:        {validation_boundary}\n")

    reader = pd.read_csv(structured_log_path, chunksize=chunk_size, iterator=True)
    carry_over_df = pd.DataFrame()
    processed_rows = 0
    spliter = ' ;-; '
    base_cols = ['Content', 'Label']
    final_cols = ['Content', 'Label']

    # Note: The validation set created here is the original chronological one, which we will overwrite later.
    train_output_path = os.path.join(output_dir, 'train_chronological_imbalanced.csv')
    validation_output_path = os.path.join(output_dir, 'validation.csv')
    test_output_path = os.path.join(output_dir, 'test.csv')

    for path in [train_output_path, validation_output_path, test_output_path]:
        if os.path.exists(path): os.remove(path)
        pd.DataFrame(columns=final_cols).to_csv(path, index=False)

    for i, chunk in enumerate(reader):
        chunk["Label"] = chunk["Label"].apply(lambda x: int(x != "-"))
        chunk_with_carryover = pd.concat([carry_over_df, chunk], ignore_index=True)
        start_row_this_chunk = processed_rows
        end_row_this_chunk = processed_rows + len(chunk)

        if start_row_this_chunk < train_boundary:
            train_part_end = min(len(chunk_with_carryover), train_boundary - start_row_this_chunk)
            train_df_part = chunk_with_carryover.iloc[:train_part_end]
            session_df = fixedSize_window(train_df_part[base_cols], window_size, step_size)
            if not session_df.empty:
                session_df['Content'] = session_df['Content'].apply(lambda x: spliter.join(x))
                session_df.to_csv(train_output_path, mode='a', header=False, index=False, columns=final_cols)

        if end_row_this_chunk > train_boundary and start_row_this_chunk < validation_boundary:
            val_part_start = max(0, train_boundary - start_row_this_chunk)
            val_part_end = min(len(chunk_with_carryover), validation_boundary - start_row_this_chunk)
            val_df_part = chunk_with_carryover.iloc[val_part_start:val_part_end]
            session_df = fixedSize_window(val_df_part[base_cols], window_size, step_size)
            if not session_df.empty:
                session_df['Content'] = session_df['Content'].apply(lambda x: spliter.join(x))
                session_df.to_csv(validation_output_path, mode='a', header=False, index=False, columns=final_cols)

        if end_row_this_chunk > validation_boundary:
            test_part_start = max(0, validation_boundary - start_row_this_chunk)
            test_df_part = chunk_with_carryover.iloc[test_part_start:]
            session_df = fixedSize_window(test_df_part[base_cols], window_size, step_size)
            if not session_df.empty:
                session_df['Content'] = session_df['Content'].apply(lambda x: spliter.join(x))
                session_df.to_csv(test_output_path, mode='a', header=False, index=False, columns=final_cols)

        carry_over_df = chunk.iloc[-(window_size - 1):] if window_size > 1 else pd.DataFrame()
        processed_rows += len(chunk)
    
    print("Chronological splitting complete.")

def resample_and_balance_training_set():
    """
    Step 2: Resamples the imbalanced training set to create a balanced
    "golden source" dataset.
    """
    print("\n--- Step 3: Resampling Imbalanced Training Set ---")
    
    imbalanced_train_path = os.path.join(output_dir, 'train_chronological_imbalanced.csv')
    balanced_source_path = os.path.join(output_dir, 'train_balanced_source.csv')

    df = pd.read_csv(imbalanced_train_path)
    df_normal = df[df['Label'] == 0]
    df_anomalous = df[df['Label'] == 1]
    
    if len(df_normal) == 0:
        raise ValueError("No normal samples found in the training set. Cannot resample.")

    num_normal_to_keep = len(df_normal)
    num_anomalous_to_keep = int((TARGET_ANOMALY_PERCENTAGE * num_normal_to_keep) / (1 - TARGET_ANOMALY_PERCENTAGE))
    num_anomalous_to_keep = min(num_anomalous_to_keep, len(df_anomalous))

    df_anomalous_sampled = df_anomalous.sample(n=num_anomalous_to_keep, random_state=42)
    df_resampled = pd.concat([df_normal, df_anomalous_sampled])
    df_resampled = df_resampled.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df_resampled.to_csv(balanced_source_path, index=False)
    print(f"Created balanced 'golden source' training data at: {balanced_source_path}")
    print(f"Source data has {len(df_resampled)} sequences with ~{TARGET_ANOMALY_PERCENTAGE*100:.1f}% anomalies.")

def create_holdout_validation_set():
    """
    Step 3: Splits the balanced "golden source" data into the final
    train.csv and validation.csv files.
    """
    print("\n--- Step 4: Creating Representative Validation Set ---")
    
    balanced_source_path = os.path.join(output_dir, 'train_balanced_source.csv')
    final_train_path = os.path.join(output_dir, 'train.csv')
    final_validation_path = os.path.join(output_dir, 'validation.csv') # We will overwrite the old validation.csv

    df = pd.read_csv(balanced_source_path)
    
    # Create a 90/10 split
    split_index = int(0.9 * len(df))
    df_train = df.iloc[:split_index]
    df_validation = df.iloc[split_index:]

    df_train.to_csv(final_train_path, index=False)
    df_validation.to_csv(final_validation_path, index=False)
    
    print(f"Final training set created with {len(df_train)} samples.")
    print(f"Final validation set created with {len(df_validation)} samples.")
    print(f"Overwrote:\n  - {final_train_path}\n  - {final_validation_path}")


if __name__ == '__main__':
    perform_chronological_split()
    resample_and_balance_training_set()
    create_holdout_validation_set()
    print("\nLiberty dataset preparation complete.")