import pandas as pd
from pathlib import Path
import re
import os
from tqdm import tqdm

# --- Configuration ---

# 1. Source: The full path to the raw Thunderbird log file.
SOURCE_LOG_FILE = "/home/koganrath/Personal/coding/loghub-logs/thunderbird/Thunderbird.log"

# 2. Destination: The directory where the final train.csv, validation.csv, etc., will be saved.
DESTINATION_DIRECTORY = "/home/koganrath/Personal/coding/log-sentinel/datasets/Thunderbird"

# 3. Processing Parameters
LOG_FORMAT = '<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>'
WINDOW_SIZE = 100
STEP_SIZE = 100
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
OVERSAMPLING_FACTOR = 10

# This controls how many raw log lines are read and processed in memory at once.
CHUNK_SIZE_FOR_PARSING = 1000000


def generate_logformat_regex(log_format):
    """Generates headers from the log format string."""
    headers = []
    splitters = re.split(r'(<[^<>]+>)', log_format)
    for k in range(len(splitters)):
        if k % 2 != 0:
            header = splitters[k].strip('<').strip('>')
            headers.append(header)
    return headers

def create_sequences_from_df(df, window_size, step_size, spliter=' ;-; '):
    """Applies a fixed-size sliding window to a DataFrame of logs to create sequences."""
    if df.empty or len(df) < window_size:
        return pd.DataFrame([], columns=['Content', 'Label'])
        
    sequences = []
    labels = df['Label'].values
    content = df['Content'].values
    
    for i in range(0, len(df) - window_size + 1, step_size):
        label = 1 if np.any(labels[i:i + window_size]) else 0
        
        # --- START OF FIX ---
        # Ensure all items are strings before joining.
        # This prevents 'TypeError: expected str instance, float found'
        content_list = [str(item) for item in content[i:i + window_size]]
        # --- END OF FIX ---

        sequences.append({'Content': spliter.join(content_list), 'Label': label})
    
    return pd.DataFrame(sequences)


def main():
    """Main function to orchestrate the entire data preparation pipeline."""
    
    dest_dir = Path(DESTINATION_DIRECTORY)
    dest_dir.mkdir(parents=True, exist_ok=True)
    source_log = Path(SOURCE_LOG_FILE)

    if not source_log.exists():
        print(f"CRITICAL ERROR: Source log file not found at '{source_log}'")
        return

    imbalanced_train_path = dest_dir / "_temp_train_imbalanced.csv"
    final_train_path = dest_dir / "train.csv"
    final_validation_path = dest_dir / "validation.csv"
    final_test_path = dest_dir / "test.csv"
    
    headers = generate_logformat_regex(LOG_FORMAT)
    num_columns = len(headers)

    print("\n--- Steps 1 & 2: Parsing, Splitting, and Windowing in memory-safe chunks... ---")
    
    for path in [imbalanced_train_path, final_validation_path, final_test_path, final_train_path]:
        if path.exists(): os.remove(path)

    try:
        total_lines = sum(1 for line in open(source_log, 'r', encoding='latin-1'))
    except Exception as e:
        print(f"Could not read file to get total lines: {e}")
        total_lines = 212000000 # Fallback for progress bar if count fails

    train_boundary = int(total_lines * TRAIN_RATIO)
    validation_boundary = int(total_lines * (TRAIN_RATIO + VALIDATION_RATIO))
    
    print(f"Total lines to process: ~{total_lines}")
    print(f"Train/Validation/Test boundaries: {train_boundary} / {validation_boundary} / {total_lines}")

    carry_over_df = pd.DataFrame()
    processed_rows = 0

    reader = pd.read_csv(
        source_log,
        sep=r'\s+',
        header=None,
        names=headers,
        engine='c',
        on_bad_lines='skip',
        chunksize=CHUNK_SIZE_FOR_PARSING,
        encoding='latin-1',
        usecols=range(num_columns)
    )

    with tqdm(total=total_lines, desc="Processing log file") as pbar:
        for chunk in reader:
            chunk["Label"] = (chunk["Label"] != "-").astype(int)
            
            chunk_with_carryover = pd.concat([carry_over_df, chunk], ignore_index=True)
            
            start_row_this_chunk = processed_rows
            end_row_this_chunk = processed_rows + len(chunk)

            if start_row_this_chunk < train_boundary:
                train_part_end = min(len(chunk_with_carryover), train_boundary - start_row_this_chunk)
                train_df_part = chunk_with_carryover.iloc[:train_part_end]
                session_df = create_sequences_from_df(train_df_part, WINDOW_SIZE, STEP_SIZE)
                session_df.to_csv(imbalanced_train_path, mode='a', header=not os.path.exists(imbalanced_train_path), index=False)

            if end_row_this_chunk > train_boundary and start_row_this_chunk < validation_boundary:
                val_part_start = max(0, train_boundary - start_row_this_chunk)
                val_part_end = min(len(chunk_with_carryover), validation_boundary - start_row_this_chunk)
                val_df_part = chunk_with_carryover.iloc[val_part_start:val_part_end]
                session_df = create_sequences_from_df(val_df_part, WINDOW_SIZE, STEP_SIZE)
                session_df.to_csv(final_validation_path, mode='a', header=not os.path.exists(final_validation_path), index=False)

            if end_row_this_chunk > validation_boundary:
                test_part_start = max(0, validation_boundary - start_row_this_chunk)
                test_df_part = chunk_with_carryover.iloc[test_part_start:]
                session_df = create_sequences_from_df(test_df_part, WINDOW_SIZE, STEP_SIZE)
                session_df.to_csv(final_test_path, mode='a', header=not os.path.exists(final_test_path), index=False)
            
            carry_over_df = chunk.iloc[-(WINDOW_SIZE - 1):] if WINDOW_SIZE > 1 else pd.DataFrame()
            processed_rows += len(chunk)
            pbar.update(len(chunk))

    print("Streaming process complete.")
    
    print("\n--- Step 3: Correcting class imbalance in training set... ---")
    if not os.path.exists(imbalanced_train_path):
        print("Warning: Imbalanced training file not found. Skipping resampling.")
    else:
        df_imbalanced = pd.read_csv(imbalanced_train_path)
        df_normal = df_imbalanced[df_imbalanced['Label'] == 0]
        df_anomalous = df_imbalanced[df_imbalanced['Label'] == 1]
        
        print(f"Original training counts -> Normal: {len(df_normal)}, Anomalous: {len(df_anomalous)}")

        if len(df_anomalous) > 0:
            df_anomalous_oversampled = pd.concat([df_anomalous] * OVERSAMPLING_FACTOR, ignore_index=True)
            df_new_train = pd.concat([df_normal, df_anomalous_oversampled]).sample(frac=1, random_state=42).reset_index(drop=True)
            print(f"Oversampled training counts -> Normal: {len(df_normal)}, Anomalous: {len(df_anomalous_oversampled)}")
            df_new_train.to_csv(final_train_path, index=False)
            print(f"Successfully created final training set with {len(df_new_train)} sequences.")
        else:
            print("No anomalies in training data. Using original set as final.")
            df_imbalanced.to_csv(final_train_path, index=False)

        print("\n--- Step 4: Cleaning up temporary files... ---")
        os.remove(imbalanced_train_path)
        print("Cleanup complete.")
    
    print("\n--- Thunderbird dataset preparation finished successfully! ---")


if __name__ == "__main__":
    import numpy as np
    main()