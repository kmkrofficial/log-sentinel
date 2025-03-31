import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import re

# --- Preprocessing Patterns and Function ---
patterns = [
    r'True',
    r'true',
    r'False',
    r'false',
    r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b',
    r'\b(Mon|Monday|Tue|Tuesday|Wed|Wednesday|Thu|Thursday|Fri|Friday|Sat|Saturday|Sun|Sunday)\b',
    r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})\s+\b',
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?', #  IP
    r'([0-9A-Fa-f]{2}:){11}[0-9A-Fa-f]{2}',   # Special MAC
    r'([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}',   # MAC
    r'[a-zA-Z0-9]*[:\.]*([/\\]+[^/\\\s\[\]]+)+[/\\]*',  # File Path
    r'\b[0-9a-fA-F]{8}\b',
    r'\b[0-9a-fA-F]{10}\b',
    r'(\w+[\w\.]*)@(\w+[\w\.]*)\-(\w+[\w\.]*)',
    r'(\w+[\w\.]*)@(\w+[\w\.]*)',
    r'[a-zA-Z\.\:\-\_]*\d[a-zA-Z0-9\.\:\-\_]*',  # word have number
]

# 合并所有模式
combined_pattern = '|'.join(patterns)

# 替换函数
def replace_patterns(text):
    # Ensure input is a string
    if not isinstance(text, str):
        text = str(text) # Attempt to convert non-strings
    text = re.sub(r'[\.]{3,}', '.. ', text)    # Replace multiple '.' with '.. '
    text = re.sub(combined_pattern, '<*>', text)
    return text
# --- End Preprocessing ---


class CustomDataset(Dataset):
    def __init__(self, file_path):
        print(f"Loading and preprocessing data from: {file_path}")
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise

        # Apply preprocessing safely
        # Use .loc to avoid SettingWithCopyWarning if df is a slice
        df['Processed_Content'] = df['Content'].apply(replace_patterns)

        # Split content into lists of strings, store as a list of lists
        # This ensures the main data structure is Python lists
        self.sequences = [content.split(' ;-; ') for content in df['Processed_Content'].values]

        # Store labels as numpy array for efficient indexing and calculations
        self.labels = df['Label'].values # Should be 0 or 1 based on your description

        # Calculate stats for oversampling
        self.num_normal = (self.labels == 0).sum()
        self.num_anomalous = (self.labels == 1).sum()
        print(f"Found {self.num_normal} normal samples and {self.num_anomalous} anomalous samples.")

        # Calculate weights (optional, not currently used in train.py loss)
        # self.normal_weight = max(self.num_anomalous / self.num_normal, 1) if self.num_normal > 0 else 1
        # self.anomalous_weight = max(self.num_normal / self.num_anomalous, 1) if self.num_anomalous > 0 else 1

        # Determine minority class for oversampling
        if self.num_normal > self.num_anomalous:
            self.minority_label = 1
            self.less_indexes = np.where(self.labels == self.minority_label)[0]
            self.num_majority = self.num_normal
            self.num_less = self.num_anomalous
            print(f"Minority class: Anomalous ({self.num_less})")
        elif self.num_anomalous > self.num_normal:
            self.minority_label = 0
            self.less_indexes = np.where(self.labels == self.minority_label)[0]
            self.num_majority = self.num_anomalous
            self.num_less = self.num_normal
            print(f"Minority class: Normal ({self.num_less})")
        else:
            print("Classes are balanced or only one class present.")
            self.minority_label = -1 # Indicate no specific minority class
            self.less_indexes = np.array([], dtype=int)
            self.num_majority = self.num_normal # or num_anomalous, they are equal
            self.num_less = 0 # No minority class count needed

        print(f"Dataset initialized. Total sequences: {len(self.labels)}")


    def __len__(self):
        return len(self.labels)

    def get_batch(self, indexes):
        """
        Returns a batch of sequences and corresponding string labels.
        Sequences are returned as a list of lists of strings.
        Labels are returned as a list of strings ('normal' or 'anomalous').
        """
        # --- FIX: Return list of lists for sequences ---
        # Use list comprehension for efficiency
        this_batch_seqs = [self.sequences[i] for i in indexes]
        # --- END FIX ---

        # Get corresponding numeric labels
        temp_numeric_labels = self.labels[indexes]

        # Convert numeric labels (0/1) to string labels ('normal'/'anomalous')
        # Using list comprehension is generally cleaner than numpy fancy indexing here
        this_batch_labels = ['anomalous' if lbl == 1 else 'normal' for lbl in temp_numeric_labels]

        return this_batch_seqs, this_batch_labels

    def get_label(self):
        """
        Returns all labels as a NumPy array of integers (0 or 1).
        Needed for evaluation script.
        """
        # self.labels should already be a numpy array of 0s and 1s
        return self.labels