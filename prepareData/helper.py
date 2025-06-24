import os
import pandas as pd
import re
from datetime import datetime

def fixedSize_window(raw_data, window_size, step_size):
    """
    This function remains unchanged as per the request, but it will now operate
    on smaller, chunked DataFrames passed to it.
    """
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



def sliding_window(raw_data, para):
    """
    split logs into time sliding windows
    :param raw_data: dataframe columns=[timestamp, label, time duration, content]
    :param para:{window_size: seconds, step_size: seconds}
    :return: dataframe
    """
    log_size = raw_data.shape[0]
    label_data, time_data = raw_data.iloc[:, 1], raw_data.iloc[:, 0]
    deltaT_data = raw_data.iloc[:, 2]
    content= raw_data.iloc[:, 3]

    new_data = []
    start_end_index_pair = set()

    start_time = time_data[0]
    end_time = start_time + para["window_size"]
    start_index = 0
    end_index = 0

    # get the first start, end index, end time
    for cur_time in time_data:
        if cur_time < end_time:
            end_index += 1
        else:
            break

    start_end_index_pair.add(tuple([start_index, end_index]))

    # move the start and end index until next sliding window
    num_session = 1
    while end_index < log_size:
        start_time = start_time + para['step_size']
        end_time = start_time + para["window_size"]
        for i in range(start_index, log_size):
            if time_data[i] < start_time:
                i += 1
            else:
                break
        for j in range(end_index, log_size):
            if time_data[j] < end_time:
                j += 1
            else:
                break
        start_index = i
        end_index = j

        # when start_index == end_index, there is no value in the window
        if start_index != end_index:
            start_end_index_pair.add(tuple([start_index, end_index]))

        num_session += 1
        if num_session % 1000 == 0:
            print("process {} time window".format(num_session), end='\r')

    for (start_index, end_index) in start_end_index_pair:
        dt = deltaT_data[start_index: end_index].values
        dt[0] = 0
        new_data.append([
            time_data[start_index: end_index].values,
            max(label_data[start_index:end_index]),
            dt,
            content[start_index: end_index].values,
            label_data[start_index:end_index].values.tolist(),
        ])

    assert len(start_end_index_pair) == len(new_data)
    print('there are %d instances (sliding windows) in this dataset\n' % len(start_end_index_pair))
    return pd.DataFrame(new_data, columns=list(raw_data.columns)+['item_Label'])

def log_to_dataframe_generator(log_file, regex, headers, start_line, end_line, chunk_size=1000000):
    """ 
    --- FIX: Modified to be a generator that yields DataFrames in chunks ---
    This prevents loading the entire log file into memory.
    """
    log_messages = []
    line_count = 0
    
    with open(log_file, 'r', encoding='latin-1') as fin:
        for i, line in enumerate(fin):
            if i < start_line:
                continue
            if end_line is not None and i >= end_line:
                break
            
            try:
                match = regex.search(line.strip())
                if match:
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    line_count += 1
            except Exception as e:
                pass # Skip lines that don't match

            if line_count == chunk_size:
                yield pd.DataFrame(log_messages, columns=headers)
                log_messages = []
                line_count = 0
    
    # Yield any remaining log messages
    if log_messages:
        yield pd.DataFrame(log_messages, columns=headers)


def generate_logformat_regex(logformat):
    """ Function to generate regular expression to split log messages
    """
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', '\\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return headers, regex

def structure_log(input_dir, output_dir, log_name, log_format, start_line=0, end_line=None):
    """ 
    --- FIX: Modified to consume the generator and write to CSV in chunks ---
    """
    print('Structuring file: ' + os.path.join(input_dir, log_name))
    start_time = datetime.now()
    headers, regex = generate_logformat_regex(log_format)
    
    output_path = os.path.join(output_dir, log_name + '_structured.csv')
    
    # Use the generator to process the log file in chunks
    chunk_generator = log_to_dataframe_generator(os.path.join(input_dir, log_name), regex, headers, start_line, end_line)

    # Write the first chunk with a header, then append the rest
    first_chunk = True
    for i, df_chunk in enumerate(chunk_generator):
        print(f"Processing and writing chunk {i+1}...", end='\r')
        if first_chunk:
            df_chunk.to_csv(output_path, index=False, escapechar='\\')
            first_chunk = False
        else:
            df_chunk.to_csv(output_path, mode='a', header=False, index=False, escapechar='\\')
    
    print("\nStructuring done. [Time taken: {!s}]".format(datetime.now() - start_time))