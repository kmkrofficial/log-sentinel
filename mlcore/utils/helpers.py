import re
import pandas as pd
import torch
import time
import os

def merge_data(sequences):
    all_logs_flat = []
    start_positions = [0]
    
    for seq in sequences:
        if seq is None or len(seq) == 0:
            start_positions.append(start_positions[-1])
            continue
        all_logs_flat.extend(seq)
        start_positions.append(len(all_logs_flat))
        
    return all_logs_flat, start_positions

def format_time(seconds):
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h {minutes}m {secs}s"
    elif hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def get_eta(start_time, steps_done, steps_total):
    if steps_done <= 0:
        return "N/A"
    
    elapsed_time = time.time() - start_time
    avg_time_per_step = elapsed_time / steps_done
    remaining_steps = steps_total - steps_done
    remaining_time = remaining_steps * avg_time_per_step
    
    return format_time(remaining_time)