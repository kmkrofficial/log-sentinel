import os
import re
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from helper import structure_log

# --- Main Configuration ---
data_dir = r'/home/koganrath/Personal/coding/loghub-logs/HDFS_v1'
log_name = "HDFS.log"
output_dir = "/home/koganrath/Personal/coding/log-sentinel/datasets/HDFS_sampler"

# --- Sub-sampling Configuration ---
# Set to True to create a smaller, chunked dataset for faster, local runs.
# Set to False to generate the full dataset for final, publishable results.
IS_SAMPLING_ENABLED = True
SAMPLE_FRACTION = 0.2  # Use 20% of the total available sessions.

if __name__ == '__main__':
    log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
    
    structured_log_file = os.path.join(output_dir, log_name + "_structured.csv")
    if not os.path.exists(structured_log_file):
        structure_log(data_dir, output_dir, log_name, log_format)
    else:
        print(f"Structured file already exists at {structured_log_file}. Skipping structuring.")

    spliter = ' ;-; '
    train_ratio = 0.8
    validation_ratio = 0.1

    df = pd.read_csv(structured_log_file, engine='c',
            na_filter=False, memory_map=True, dtype={'Date':object, "Time": object})

    print(f'Number of messages in {structured_log_file} is {len(df)}')

    data_dict_content = defaultdict(list)
    for idx, row in tqdm(df.iterrows(),total=len(df)):
        blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            data_dict_content[blk_Id].append(row["Content"])

    data_df = pd.DataFrame(list(data_dict_content.items()), columns=['BlockId', 'Content'])
    print(f"Assembled {len(data_df)} unique sessions (BlockIDs).")

    blk_label_dict = {}
    blk_label_file = os.path.join(data_dir, "anomaly_label.csv")
    blk_df = pd.read_csv(blk_label_file)
    for _, row in tqdm(blk_df.iterrows(),total=len(blk_df)):
        blk_label_dict[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0

    data_df["Label"] = data_df["BlockId"].apply(lambda x: blk_label_dict.get(x))

    data_df = data_df.dropna(subset=['Label'])
    data_df['Label'] = data_df['Label'].astype(int)

    if IS_SAMPLING_ENABLED:
        print(f"\n--- SAMPLING ENABLED: Using {SAMPLE_FRACTION * 100:.0f}% of the total sessions ---")
        data_df = data_df.sample(frac=SAMPLE_FRACTION, random_state=42).reset_index(drop=True)
        print(f"Sampled down to {len(data_df)} total sessions.")

    data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_pool_len = int(train_ratio * len(data_df))
    train_val_pool_df = data_df[:train_pool_len]
    session_test_df = data_df[train_pool_len:]

    validation_len = int(validation_ratio * len(train_val_pool_df))
    session_validation_df = train_val_pool_df[:validation_len]
    session_train_df = train_val_pool_df[validation_len:]
    
    session_train_df = session_train_df.reset_index(drop=True)
    session_validation_df = session_validation_df.reset_index(drop=True)
    session_test_df = session_test_df.reset_index(drop=True)

    datasets = {
        'train': session_train_df,
        'validation': session_validation_df,
        'test': session_test_df
    }

    stats = {}
    print("\n--- Generating final datasets and statistics ---")
    for name, df_to_process in datasets.items():
        df_to_process['session_length'] = df_to_process["Content"].apply(len)
        df_to_process["Content"] = df_to_process["Content"].apply(lambda x: spliter.join(x))
        
        output_path = os.path.join(output_dir, f'{name}.csv')
        df_to_process.to_csv(output_path, index=False)

        stats[name] = {
            'mean_len': df_to_process['session_length'].mean(),
            'max_len': df_to_process['session_length'].max(),
            'anomalous': df_to_process['Label'].sum(),
            'normal': len(df_to_process['Label']) - df_to_process['Label'].sum(),
            'total': len(df_to_process['Label'])
        }

    for name, stat_data in stats.items():
        print(f'\n{name.capitalize()} dataset info:')
        print(f"  - Total sessions: {stat_data['total']}")
        print(f"  - Anomalous sessions: {stat_data['anomalous']}")
        print(f"  - Normal sessions: {stat_data['normal']}")
        print(f"  - Max session length: {stat_data['max_len']}; Mean session length: {stat_data['mean_len']:.2f}")

        info_path = os.path.join(output_dir, f'{name}_info.txt')
        with open(info_path, 'w') as file:
            file.write(f"Total sessions: {stat_data['total']}\n")
            file.write(f"Anomalous sessions: {stat_data['anomalous']}\n")
            file.write(f"Normal sessions: {stat_data['normal']}\n")