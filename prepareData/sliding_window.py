import os
import pandas as pd
from helper import fixedSize_window, structure_log

# --- CHOOSE YOUR DATASET HERE ---
# Options: "BGL", "Liberty", "Thunderbird"
DATASET_TO_PROCESS = "BGL"

def run_preparation():
    # --- Script Configuration ---
    if DATASET_TO_PROCESS == "BGL":
        data_dir = r'D:\coding\datasets\BGL'
        log_name = "BGL.log"
        log_format = '<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>'
        start_line = 0
        end_line = None # Process the whole file

    elif DATASET_TO_PROCESS == "Liberty":
        data_dir = r'D:\coding\datasets\liberty'
        log_name = "liberty2.log" # Assuming a .log extension
        log_format = '<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>'
        start_line = 40000000
        end_line = 45000000

    elif DATASET_TO_PROCESS == "Thunderbird":
        data_dir = r'D:\coding\datasets\thunderbird'
        log_name = "Thunderbird.log"
        log_format = '<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>'
        start_line = 160000000
        end_line = 170000000

    else:
        raise ValueError(f"Unknown dataset '{DATASET_TO_PROCESS}'. Please choose from 'BGL', 'Liberty', 'Thunderbird'.")

    output_dir = data_dir
    window_size = 100
    step_size = 100
    train_ratio = 0.8
    validation_ratio = 0.1
    chunk_size = 10000000

    print(f"--- Preparing Dataset: {DATASET_TO_PROCESS} ---")
    
    structured_log_path = os.path.join(output_dir, f'{log_name}_structured.csv')
    if not os.path.exists(structured_log_path):
        structure_log(data_dir, output_dir, log_name, log_format, start_line=start_line, end_line=end_line)
    else:
        print(f"Structured file already exists at {structured_log_path}. Skipping structuring.")

    print(f'Window Size: {window_size}; Step Size: {step_size}')

    print("Counting lines in structured file...")
    try:
        with open(structured_log_path, 'r', encoding='latin-1') as f:
            total_lines = sum(1 for line in f) - 1
        print(f"Total lines to process: {total_lines}")
    except FileNotFoundError:
        print(f"ERROR: Structured file not found at {structured_log_path}. Cannot proceed.")
        return

    train_pool_boundary = int(total_lines * train_ratio)
    validation_boundary = int(total_lines * (train_ratio + validation_ratio))

    print("\nData split boundaries (row index):")
    print(f"Training set ends at:      {train_pool_boundary}")
    print(f"Validation set ends at:    {validation_boundary}")
    print(f"Test set starts at:        {validation_boundary}\n")

    reader = pd.read_csv(structured_log_path, chunksize=chunk_size, iterator=True, encoding='latin-1')
    
    carry_over_df = pd.DataFrame()
    processed_rows = 0
    
    spliter = ' ;-; '
    base_cols = ['Content', 'Label']
    final_cols = ['Content', 'Label']
    
    train_output_path = os.path.join(output_dir, 'train.csv')
    validation_output_path = os.path.join(output_dir, 'validation.csv')
    test_output_path = os.path.join(output_dir, 'test.csv')

    for path in [train_output_path, validation_output_path, test_output_path]:
        if os.path.exists(path):
            os.remove(path)
        pd.DataFrame(columns=final_cols).to_csv(path, index=False)
    
    print("Starting chunked processing and windowing...")
    for i, chunk in enumerate(reader):
        print(f"Processing chunk {i+1}...")
        
        chunk["Label"] = chunk["Label"].apply(lambda x: int(x != "-"))
        
        chunk_with_carryover = pd.concat([carry_over_df, chunk], ignore_index=True)

        start_row_this_chunk = processed_rows

        if start_row_this_chunk < train_pool_boundary:
            train_part_end = min(len(chunk_with_carryover), train_pool_boundary - start_row_this_chunk)
            train_df_part = chunk_with_carryover.iloc[:train_part_end]
            session_df = fixedSize_window(train_df_part[base_cols], window_size, step_size)
            if not session_df.empty:
                session_df['Content'] = session_df['Content'].apply(lambda x: spliter.join(x))
                session_df.to_csv(train_output_path, mode='a', header=False, index=False, columns=final_cols)

        if start_row_this_chunk < validation_boundary and processed_rows + len(chunk_with_carryover) > train_pool_boundary:
            val_part_start = max(0, train_pool_boundary - start_row_this_chunk)
            val_part_end = min(len(chunk_with_carryover), validation_boundary - start_row_this_chunk)
            val_df_part = chunk_with_carryover.iloc[val_part_start:val_part_end]
            session_df = fixedSize_window(val_df_part[base_cols], window_size, step_size)
            if not session_df.empty:
                session_df['Content'] = session_df['Content'].apply(lambda x: spliter.join(x))
                session_df.to_csv(validation_output_path, mode='a', header=False, index=False, columns=final_cols)

        if processed_rows + len(chunk_with_carryover) > validation_boundary:
            test_part_start = max(0, validation_boundary - start_row_this_chunk)
            test_df_part = chunk_with_carryover.iloc[test_part_start:]
            session_df = fixedSize_window(test_df_part[base_cols], window_size, step_size)
            if not session_df.empty:
                session_df['Content'] = session_df['Content'].apply(lambda x: spliter.join(x))
                session_df.to_csv(test_output_path, mode='a', header=False, index=False, columns=final_cols)

        carry_over_df = chunk.iloc[-(window_size - 1):] if window_size > 1 else pd.DataFrame()
        processed_rows += len(chunk)
        
    print("\nChunked processing complete. Final datasets created.")

if __name__ == '__main__':
    run_preparation()