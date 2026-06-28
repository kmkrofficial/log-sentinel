import pandas as pd
from pathlib import Path
import re
import os
from tqdm import tqdm
import numpy as np
from helper import generate_logformat_regex, log_to_dataframe_generator

# --- Configuration ---

# 1. Source: The full path to the raw Thunderbird log file.
SOURCE_LOG_FILE = r"D:\coding\datasets\Thunderbird\Thunderbird.log"

# 2. Destination: The directory where the final train.csv, validation.csv, etc., will be saved.
DESTINATION_DIRECTORY = r"D:\coding\log-sentinel\datasets\Thunderbird"

# 3. Slicing: Define the specific lines from the raw log file to process.
START_LINE = 160000000
END_LINE = 170000000

# 4. Processing Parameters
LOG_FORMAT = '<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>'
WINDOW_SIZE = 100
STEP_SIZE = 100
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
OVERSAMPLING_FACTOR = 10
CHUNK_SIZE_FOR_PARSING = 1000000


def create_sequences_from_df(df, window_size, step_size, spliter=' ;-; '):
    if df.empty or len(df) < window_size:
        return pd.DataFrame([], columns=['Content', 'Label'])
        
    sequences = []
    labels = df['Label'].values
    content = df['Content'].values
    
    for i in range(0, len(df) - window_size + 1, step_size):
        label = 1 if np.any(labels[i:i + window_size]) else 0
        content_list = [str(item) for item in content[i:i + window_size]]
        sequences.append({'Content': spliter.join(content_list), 'Label': label})
    
    return pd.DataFrame(sequences)


def main():
    dest_dir = Path(DESTINATION_DIRECTORY)
    dest_dir.mkdir(parents=True, exist_ok=True)
    source_log = Path(SOURCE_LOG_FILE)

    if not source_log.exists():
        print(f"CRITICAL ERROR: Source log file not found at '{source_log}'")
        return

    if START_LINE >= END_LINE:
        print(f"CRITICAL ERROR: START_LINE ({START_LINE}) must be less than END_LINE ({END_LINE}).")
        return
    
    headers, regex = generate_logformat_regex(LOG_FORMAT)
    total_lines_in_slice = END_LINE - START_LINE
    
    print("\n--- Step 1: Parsing Sliced Log into a single pool using robust regex... ---")
    print(f"Processing a slice of {total_lines_in_slice} lines from the source file.")

    chunk_generator = log_to_dataframe_generator(
        log_file=source_log,
        regex=regex,
        headers=headers,
        start_line=START_LINE,
        end_line=END_LINE,
        chunk_size=CHUNK_SIZE_FOR_PARSING
    )

    all_sequences = []
    
    with tqdm(total=total_lines_in_slice, desc="Parsing log slice") as pbar:
        for chunk in chunk_generator:
            chunk['Label'] = (chunk['Label'] != '-').astype(int)
            sequences_chunk = create_sequences_from_df(chunk, WINDOW_SIZE, STEP_SIZE)
            all_sequences.append(sequences_chunk)
            pbar.update(len(chunk))
    
    df_pool = pd.concat(all_sequences, ignore_index=True)
    print(f"Parsing complete. Generated a pool of {len(df_pool)} sequences.")

    print("\n--- Step 2: Oversampling the anomaly class in the entire pool... ---")
    df_normal = df_pool[df_pool['Label'] == 0]
    df_anomalous = df_pool[df_pool['Label'] == 1]
    
    print(f"Initial pool counts -> Normal: {len(df_normal)}, Anomalous: {len(df_anomalous)}")

    if len(df_anomalous) > 0 and len(df_normal) > 0 and OVERSAMPLING_FACTOR > 1:
        df_anomalous_oversampled = pd.concat([df_anomalous] * OVERSAMPLING_FACTOR, ignore_index=True)
        df_final_pool = pd.concat([df_normal, df_anomalous_oversampled])
        print(f"Oversampled pool counts -> Normal: {len(df_normal)}, Anomalous: {len(df_anomalous_oversampled)}")
    else:
        df_final_pool = df_pool
        print("No oversampling performed (no anomalies, no normal, or factor <= 1).")

    print(f"Final pool size: {len(df_final_pool)} sequences.")
    
    print("\n--- Step 3: Shuffling and splitting the pool into train, validation, and test sets... ---")
    
    df_shuffled = df_final_pool.sample(frac=1, random_state=42).reset_index(drop=True)

    train_end_idx = int(len(df_shuffled) * TRAIN_RATIO)
    validation_end_idx = train_end_idx + int(len(df_shuffled) * VALIDATION_RATIO)
    
    df_train = df_shuffled.iloc[:train_end_idx]
    df_validation = df_shuffled.iloc[train_end_idx:validation_end_idx]
    df_test = df_shuffled.iloc[validation_end_idx:]

    datasets = {
        "train": df_train,
        "validation": df_validation,
        "test": df_test
    }

    print("\n--- Step 4: Saving final datasets and reporting statistics... ---")
    for name, df in datasets.items():
        output_path = dest_dir / f"{name}.csv"
        df.to_csv(output_path, index=False)
        
        total_seq = len(df)
        anomalous_seq = df['Label'].sum()
        normal_seq = total_seq - anomalous_seq
        anomaly_pct = (anomalous_seq / total_seq * 100) if total_seq > 0 else 0
        
        print(f"\n{name.capitalize()} dataset:")
        print(f"  - Total Sequences: {total_seq}")
        print(f"  - Normal Sequences: {normal_seq}")
        print(f"  - Anomalous Sequences: {anomalous_seq} ({anomaly_pct:.2f}%)")
        print(f"  - Saved to: {output_path}")

    print("\n\n--- Thunderbird dataset preparation finished successfully! ---")

if __name__ == "__main__":
    main()