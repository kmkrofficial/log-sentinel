import os
import re
import pandas as pd
from tqdm import tqdm

def process_apache_logs_in_chunks(log_file_path, regex, headers, chunk_size=1000000):
    """
    Reads a large log file line by line, parses it, and yields DataFrames in manageable chunks.
    This avoids loading the entire file into memory.
    """
    log_messages = []
    print("Starting line-by-line parsing of the log file...")
    with open(log_file_path, 'r', encoding='latin-1') as f:
        for line in f:
            try:
                match = regex.search(line.strip())
                if match:
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
            except Exception:
                continue # Skip lines that don't match

            if len(log_messages) == chunk_size:
                yield pd.DataFrame(log_messages, columns=headers)
                log_messages = [] # Reset for the next chunk
    
    if log_messages: # Yield the final, smaller chunk
        yield pd.DataFrame(log_messages, columns=headers)
    print("\nLine-by-line parsing complete.")


def fixed_size_windowing(df, window_size, step_size):
    """
    Groups a DataFrame of logs into fixed-size windows (sessions).
    """
    sessions = []
    for i in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[i:i + window_size]
        # Since logs are unlabeled, the session label is always 0 (Normal).
        session_label = 0
        
        # Aggregate content into a list for the session.
        content = window['Content'].tolist()
        sessions.append({'Content': content, 'Label': session_label})
    return pd.DataFrame(sessions)


if __name__ == '__main__':
    # --- Configuration ---
    data_dir = r'./datasets/Apache'  # Assumes your logs are in datasets/Apache
    log_name = "Apache.log"
    output_dir = data_dir

    window_size = 100  # Number of log lines per session
    step_size = 50     # Slide the window by 50 lines
    train_ratio = 0.8
    validation_ratio = 0.1

    log_file_path = os.path.join(data_dir, log_name)
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"Log file not found at: {log_file_path}")

    # Regex to parse the Apache log format: `[Timestamp] [LogLevel] Message`
    log_format_regex = re.compile(r'^\[(?P<Timestamp>.*?)\] \[(?P<LogLevel>.*?)\] (?P<Content>.*)$')
    headers = ['Timestamp', 'LogLevel', 'Content']

    # --- Step 1: Parse the raw log file in chunks and create sessions ---
    print("Processing log file into sessions. This may take a while for large files...")
    chunk_generator = process_apache_logs_in_chunks(log_file_path, log_format_regex, headers)
    
    all_sessions_df = pd.DataFrame()
    carry_over_df = pd.DataFrame()

    for i, chunk in enumerate(chunk_generator):
        print(f"Windowing chunk {i+1}...")
        # Add carry-over from previous chunk to handle window boundaries correctly
        chunk_with_carryover = pd.concat([carry_over_df, chunk], ignore_index=True)

        sessions_chunk = fixed_size_windowing(chunk_with_carryover, window_size, step_size)
        all_sessions_df = pd.concat([all_sessions_df, sessions_chunk], ignore_index=True)

        # The last (window_size - 1) rows become the carry-over for the next chunk.
        carry_over_df = chunk.iloc[-(window_size - 1):] if window_size > 1 else pd.DataFrame()
    
    print(f"\nTotal sessions created: {len(all_sessions_df)}")

    # Since all logs are unlabeled, assume they are normal.
    all_sessions_df['Label'] = 0

    # --- Step 2: Shuffle and Split the sessionized data ---
    print("Shuffling and splitting sessions into train, validation, and test sets...")
    
    # Shuffle the DataFrame once before splitting
    all_sessions_df = all_sessions_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split into train/val pool and test set
    train_pool_len = int(train_ratio * len(all_sessions_df))
    train_val_pool_df = all_sessions_df[:train_pool_len]
    session_test_df = all_sessions_df[train_pool_len:]

    # Split the pool into final train and validation sets
    validation_len = int(validation_ratio * len(train_val_pool_df))
    session_validation_df = train_val_pool_df[:validation_len]
    session_train_df = train_val_pool_df[validation_len:]
    
    # Reset indices
    session_train_df = session_train_df.reset_index(drop=True)
    session_validation_df = session_validation_df.reset_index(drop=True)
    session_test_df = session_test_df.reset_index(drop=True)

    # --- Step 3: Format, Save, and Report ---
    datasets = {
        'train': session_train_df,
        'validation': session_validation_df,
        'test': session_test_df
    }

    spliter = ' ;-; '
    for name, df_to_process in datasets.items():
        if df_to_process.empty:
            print(f"Warning: {name} dataset is empty after processing. Skipping.")
            continue
            
        # Join the list of log lines into a single string for our model
        df_to_process["Content"] = df_to_process["Content"].apply(lambda x: spliter.join(x))
        
        output_path = os.path.join(output_dir, f'{name}.csv')
        df_to_process.to_csv(output_path, index=False)
        
        # Print stats
        total_sessions = len(df_to_process)
        print(f"\n{name.capitalize()} dataset info:")
        print(f"  - Total sessions: {total_sessions}")
        print(f"  - Saved to: {output_path}")

    print("\nProcessing complete.")