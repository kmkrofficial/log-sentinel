import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler, Dataset
import random
import numpy as np
import h5py
from pathlib import Path

class HDF5Dataset(Dataset):
    def __init__(self, h5_path):
        super().__init__()
        self.h5_path = Path(h5_path)
        self.h5_file = None

        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found at {self.h5_path}")

        with h5py.File(self.h5_path, 'r') as f:
            self.total_samples = len(f['labels'])
            self.labels = torch.from_numpy(f['labels'][:])

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        sequence = torch.from_numpy(self.h5_file['sequences'][idx])
        label = self.labels[idx]
        return sequence, label

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BalancedSampler(Sampler):
    def __init__(self, labels, min_less_portion=0.5):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.labels = labels
        
        unique_labels = np.unique(self.labels)
        if len(unique_labels) < 2:
            self.minority_label, self.majority_label = 0, 0
        else:
            self.minority_label = 1 if np.sum(labels) < len(labels) / 2 else 0
            self.majority_label = 1 - self.minority_label

        self.majority_indexes = np.where(self.labels == self.majority_label)[0]
        self.less_indexes = np.where(self.labels == self.minority_label)[0]

        self.num_majority = len(self.majority_indexes)
        self.num_less = len(self.less_indexes)
        self.min_less_portion = min_less_portion

    def __iter__(self):
        if self.num_less == 0 or self.num_majority == 0:
            all_indices = np.concatenate([self.majority_indexes, self.less_indexes]).tolist()
            random.shuffle(all_indices)
            return iter(all_indices)

        oversampled_less_count = int(self.num_majority * self.min_less_portion)
        num_to_add = max(0, oversampled_less_count - self.num_less)

        oversampled_indices = random.choices(self.less_indexes, k=num_to_add)

        indexes = np.concatenate([
            self.majority_indexes,
            self.less_indexes,
            np.array(oversampled_indices, dtype=np.int64)
        ]).tolist()

        random.shuffle(indexes)
        return iter(indexes)

    def __len__(self):
        if self.num_less == 0 or self.num_majority == 0:
            return self.num_majority + self.num_less
        oversampled_less_count = int(self.num_majority * self.min_less_portion)
        return self.num_majority + self.num_less + max(0, oversampled_less_count - self.num_less)