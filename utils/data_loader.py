import pandas as pd
import numpy as np
import re
from torch.utils.data import Dataset

# --- Preprocessing Patterns ---
_PATTERNS = [
    r'True', r'true', r'False', r'false',
    r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b',
    r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}\s+\b',
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?',  # IP Address
    r'([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}',  # MAC Address
    r'[a-zA-Z0-9]*[:\.]*([/\\]+[^/\\\s\[\]]+)+[/\\]*',  # File Path
    r'[a-zA-Z\.\:\-\_]*\d[a-zA-Z0-9\.\:\-\_]*',  # Word with numbers
]
_COMBINED_PATTERN = re.compile('|'.join(_PATTERNS))

def replace_patterns(text):
    if not isinstance(text, str): text = str(text)
    text = re.sub(r'[\.]{3,}', '.. ', text)
    return _COMBINED_PATTERN.sub('<*>', text)

# --- Dataset Class ---

class LogDataset(Dataset):
    def __init__(self, file_path):
        print(f"Loading and preprocessing data from: {file_path}")
        df = pd.read_csv(file_path)

        df['Processed_Content'] = df['Content'].apply(replace_patterns)
        
        self.sequences = [content.split(' ;-; ') for content in df['Processed_Content'].values]
        # Ensure labels are integers, handle potential NaNs from file read
        self.labels = df['Label'].fillna(-1).astype(int).values

        self._calculate_class_stats()
        print(f"Dataset initialized. Total sequences: {len(self.labels)}")
        print(f"Class counts (0: Normal, 1: Anomalous): {self.class_counts}")


    def _calculate_class_stats(self):
        """Calculates statistics needed for oversampling and weighted loss."""
        # Use numpy for efficient counting of valid labels (0 and 1)
        valid_labels = self.labels[self.labels != -1]
        unique, counts = np.unique(valid_labels, return_counts=True)
        self.class_counts = dict(zip(unique, counts))

        num_normal = self.class_counts.get(0, 0)
        num_anomalous = self.class_counts.get(1, 0)

        if num_normal > num_anomalous:
            self.minority_label = 1
            self.num_less = num_anomalous
            self.num_majority = num_normal
        else:
            self.minority_label = 0
            self.num_less = num_normal
            self.num_majority = num_anomalous
            
        if self.num_less > 0:
            self.less_indexes = np.where(self.labels == self.minority_label)[0]
        else:
            self.less_indexes = np.array([], dtype=int)

    def __len__(self):
        return len(self.labels)

    def get_batch(self, indexes):
        """
        Returns sequences and string labels for a given list of indices.
        """
        this_batch_seqs = [self.sequences[i] for i in indexes]
        temp_numeric_labels = self.labels[indexes]
        this_batch_labels = ['anomalous' if lbl == 1 else 'normal' for lbl in temp_numeric_labels]
        return this_batch_seqs, this_batch_labels

    def get_all_labels(self):
        """Returns all labels as a NumPy array of integers."""
        return self.labels