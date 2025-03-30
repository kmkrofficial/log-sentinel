import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import re

# patterns = [
#     r'[a-zA-Z0-9]*:*([/\\]+[^/\\\s]+)+[/\\]*',  # 文件路径
#     r'[a-zA-Z\.\:\-\_]*\d[a-zA-Z0-9\.\:\-\_]*',  # 中间一定要有数字  数字和字母和 . 或 : 或 - 的组合
#     # r'[a-zA-Z0-9]+\.[a-zA-Z0-9]+',
# ]
#
# # 合并所有模式
# combined_pattern = '|'.join(patterns)
#
# # 替换函数
# def replace_patterns(text):
#     return re.sub(combined_pattern, '<*>', text)



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
    text = re.sub(r'[\.]{3,}', '.. ', text)    # Replace multiple '.' with '.. '
    text = re.sub(combined_pattern, '<*>', text)
    return text


class CustomDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        contents = df['Content'].apply(replace_patterns).values     #pre processing
        # contents = df['EventTemplate'].values
        self.sequences = np.array([content.split(' ;-; ') for content in contents], dtype=object)
        self.labels = df['Label'].values

        self.num_normal = (self.labels == 0).sum()
        self.num_anomalous = (self.labels == 1).sum()

        self.normal_weight = max(self.num_anomalous / self.num_normal,1)
        self.anomalous_weight = max(self.num_normal / self.num_anomalous,1)

        if self.num_normal >  self.num_anomalous:
            self.less_indexes = np.where(self.labels == 1)[0]
            self.num_majority = self.num_normal
            self.num_less = self.num_anomalous
        else:
            self.less_indexes = np.where(self.labels == 0)[0]
            self.num_majority = self.num_anomalous
            self.num_less = self.num_normal



    def __len__(self):
        return len(self.labels)

    def get_batch(self, indexes):
        this_batch_seqs = self.sequences[indexes]
        temp =  self.labels[indexes]
        this_batch_labels = temp.astype(object)
        this_batch_labels[temp == 0] = 'normal'
        this_batch_labels[temp == 1] = 'anomalous'
        return this_batch_seqs, this_batch_labels

    def get_label(self):
        return self.labels
