import os
import pandas as pd
import re
from datetime import datetime
from tqdm import tqdm

def structure_log(input_dir, output_dir, log_name, log_format, start_line=0, end_line=None, chunk_size=1000000):
    log_file_path = os.path.join(input_dir, log_name)
    print('Structuring file: ' + log_file_path)
    start_time = datetime.now()
    headers, regex = generate_logformat_regex(log_format)
    
    output_path = os.path.join(output_dir, log_name + '_structured.csv')
    
    total_lines_to_process = None
    if end_line is not None:
        total_lines_to_process = end_line - start_line
    else:
        try:
            with open(log_file_path, 'r', encoding='latin-1') as f:
                total_lines_to_process = sum(1 for line in f) - start_line
        except Exception:
             # If counting fails, we proceed without a total for the progress bar
            total_lines_to_process = None

    chunk_generator = log_to_dataframe_generator(
        log_file=log_file_path,
        regex=regex,
        headers=headers,
        start_line=start_line,
        end_line=end_line,
        chunk_size=chunk_size
    )

    is_first_chunk = True
    with tqdm(total=total_lines_to_process, desc="Structuring log file") as pbar:
        for chunk in chunk_generator:
            if is_first_chunk:
                chunk.to_csv(output_path, index=False)
                is_first_chunk = False
            else:
                chunk.to_csv(output_path, mode='a', header=False, index=False)
            pbar.update(len(chunk))
    
    print(f"\nStructuring done. Output saved to {output_path}. [Time taken: {datetime.now() - start_time}]")

def fixedSize_window(raw_data, window_size, step_size):
    if raw_data.empty:
        return pd.DataFrame([], columns=list(raw_data.columns)+['item_Label'])
        
    aggregated = [
        [raw_data['Content'].iloc[i:i + window_size].values,
        max(raw_data['Label'].iloc[i:i + window_size]),
         raw_data['Label'].iloc[i:i + window_size].values.tolist()
         ]
        for i in range(0, len(raw_data) - window_size + 1, step_size)
    ]
    return pd.DataFrame(aggregated, columns=list(raw_data.columns)+['item_Label'])

def log_to_dataframe_generator(log_file, regex, headers, start_line=0, end_line=None, chunk_size=1000000):
    log_messages = []
    line_count = 0
    
    with open(log_file, 'r', encoding='latin-1') as fin:
        try:
            for _ in range(start_line):
                next(fin)
        except StopIteration:
            if log_messages:
                yield pd.DataFrame(log_messages, columns=headers)
            return

        for line in fin:
            current_line_num = start_line + line_count
            if end_line is not None and current_line_num >= end_line:
                break
            
            try:
                match = regex.search(line.strip())
                if match:
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
            except Exception:
                pass

            line_count += 1
            if len(log_messages) == chunk_size:
                yield pd.DataFrame(log_messages, columns=headers)
                log_messages = []
    
    if log_messages:
        yield pd.DataFrame(log_messages, columns=headers)

def generate_logformat_regex(logformat):
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', r'\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += r'(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return headers, regex