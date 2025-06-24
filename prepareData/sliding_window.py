import os
import pandas as pd
import numpy as np
from helper import fixedSize_window, structure_log

# --- Script Configuration ---
data_dir = r'E:\research-stuff\LogSentinel-3b\datasets\Thunderbird'
log_name = "Thunderbird.log"
output_dir = data_dir

start_line = 0
end_line = 100000000 

window_size = 100
step_size = 100
train_ratio = 0.8
validation_ratio = 0.1
# --- FIX: Increased chunk size for better performance on systems with more RAM ---
chunk_size = 10000000 # Read 10 million lines from the structured CSV at a time

if __name__ == '__main__':
    if 'thunderbird' in log_name.lower() or 'spirit' in log_name.lower() or 'liberty' in log_name.lower():
        log_format = '<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>'
    elif 'bgl' in log_name.lower():
        log_format = '<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>'
    else:
        raise Exception('missing valid log format')
        
    print(f'Auto log_format: {log_format}')
    
    structured_log_path = os.path.join(output_dir, f'{log_name}_structured.csv')
    if not os.path.exists(structured_log_path):
        structure_log(data_dir, output_dir, log_name, log_format, start_line=start_line, end_line=end_line)
    else:
        print(f"Structured file already exists at {structured_log_path}. Skipping structuring.")

    print(f'window_size: {window_size}; step_size: {step_size}')

    print("Counting lines in structured file...")
    with open(structured_log_path, 'r', encoding='latin-1') as f:
        total_lines = sum(1 for line in f) - 1
    print(f"Total lines to process: {total_lines}")

    train_pool_boundary = int(total_lines * train_ratio)
    train_boundary = int(train_pool_boundary * (1 - validation_ratio))
    validation_boundary = train_pool_boundary
    
    print("\nData split boundaries (row index):")
    print(f"Training ends at:      {train_boundary}")
    print(f"Validation ends at:    {validation_boundary}")
    print(f"Test starts at:        {validation_boundary}\n")

    reader = pd.read_csv(structured_log_path, chunksize=chunk_size, iterator=True)
    
    carry_over_df = pd.DataFrame()
    processed_rows = 0
    
    spliter = ' ;-; '
    base_cols = ['Content', 'Label']
    session_cols = ['Content', 'Label', 'item_Label']
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
        end_row_this_chunk = processed_rows + len(chunk)
        
        # Train part
        if start_row_this_chunk < train_boundary:
            train_part_end = min(len(chunk_with_carryover), train_boundary - start_row_this_chunk)
            train_df_part = chunk_with_carryover.iloc[:train_part_end]
            session_df = fixedSize_window(train_df_part[base_cols], window_size, step_size)
            if not session_df.empty:
                session_df['Content'] = session_df['Content'].apply(lambda x: spliter.join(x))
                session_df.to_csv(train_output_path, mode='a', header=False, index=False, columns=final_cols)

        # Validation part
        if end_row_this_chunk > train_boundary and start_row_this_chunk < validation_boundary:
            val_part_start = max(0, train_boundary - start_row_this_chunk)
            val_part_end = min(len(chunk_with_carryover), validation_boundary - start_row_this_chunk)
            val_df_part = chunk_with_carryover.iloc[val_part_start:val_part_end]
            session_df = fixedSize_window(val_df_part[base_cols], window_size, step_size)
            if not session_df.empty:
                session_df['Content'] = session_df['Content'].apply(lambda x: spliter.join(x))
                session_df.to_csv(validation_output_path, mode='a', header=False, index=False, columns=final_cols)

        # Test part
        if end_row_this_chunk > validation_boundary:
            test_part_start = max(0, validation_boundary - start_row_this_chunk)
            test_df_part = chunk_with_carryover.iloc[test_part_start:]
            session_df = fixedSize_window(test_df_part[base_cols], window_size, step_size)
            if not session_df.empty:
                session_df['Content'] = session_df['Content'].apply(lambda x: spliter.join(x))
                session_df.to_csv(test_output_path, mode='a', header=False, index=False, columns=final_cols)

        carry_over_df = chunk.iloc[-(window_size - 1):] if window_size > 1 else pd.DataFrame()
        processed_rows += len(chunk)
        
    print("\nChunked processing complete. Final datasets created.")
    
    # --- FIX: Re-read final CSVs to calculate and print statistics ---
    print("\nCalculating final dataset statistics...")
    
    datasets_to_analyze = {
        'Train': train_output_path,
        'Validation': validation_output_path,
        'Test': test_output_path
    }

    for name, path in datasets_to_analyze.items():
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"{name} dataset is empty, skipping stats.")
            continue
            
        df = pd.read_csv(path)
        if df.empty:
            print(f"{name} dataset is empty, skipping stats.")
            continue

        # Calculate session length after loading
        df['session_length'] = df["Content"].apply(lambda x: len(x.split(spliter)))

        mean_len = df['session_length'].mean()
        max_len = df['session_length'].max()
        num_anomalous = df['Label'].sum()
        num_normal = len(df['Label']) - num_anomalous
        total_sessions = len(df['Label'])
        
        print(f'\n{name} dataset info:')
        print(f"max session length: {max_len}; mean session length: {mean_len:.4f}")
        print(f"number of anomalous sessions: {num_anomalous}; number of normal sessions: {num_normal}; number of total sessions: {total_sessions}\n")

        info_path = os.path.join(output_dir, f'{name.lower()}_info.txt')
        with open(info_path, 'w') as file:
            file.write(f"max session length: {max_len}; mean session length: {mean_len:.4f}\n")
            file.write(f"number of anomalous sessions: {num_anomalous}; number of normal sessions: {num_normal}; number of total sessions: {total_sessions}\n")