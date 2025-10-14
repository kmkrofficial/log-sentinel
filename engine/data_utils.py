import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler, Dataset
import random
import numpy as np

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
        self.labels = labels
        self.minority_label = 1 if np.sum(labels) < len(labels) / 2 else 0
        self.majority_label = 1 - self.minority_label
        
        self.majority_indexes = np.where(self.labels == self.majority_label)[0]
        self.less_indexes = np.where(self.labels == self.minority_label)[0]
        
        self.num_majority = len(self.majority_indexes)
        self.num_less = len(self.less_indexes)
        self.min_less_portion = min_less_portion

    def __iter__(self):
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
        oversampled_less_count = int(self.num_majority * self.min_less_portion)
        return self.num_majority + self.num_less + max(0, oversampled_less_count - self.num_less)