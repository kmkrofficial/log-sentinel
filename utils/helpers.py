import torch
import numpy as np
from torch import nn

def merge_data(data):
    """Merges a list of lists into a single list and returns start positions."""
    merged_data, start_positions, current_position = [], [], 0
    for sublist in data:
        if isinstance(sublist, (list, tuple)):
            start_positions.append(current_position)
            merged_data.extend(sublist)
            current_position += len(sublist)
    return merged_data, start_positions

def stack_and_pad_left(tensors):
    """Stacks a list of tensors, padding them on the left to the same length."""
    if not tensors:
        return torch.tensor([]), torch.tensor([])
    max_len = max(t.shape[0] for t in tensors)
    padded_tensors, padding_masks = [], []
    for tensor in tensors:
        pad_len = max_len - tensor.shape[0]
        padded_tensor = nn.functional.pad(tensor, (0, 0, pad_len, 0))
        padding_masks.append(torch.cat([
            torch.zeros(pad_len, dtype=torch.long, device=tensor.device),
            torch.ones(tensor.shape[0], dtype=torch.long, device=tensor.device)
        ]))
        padded_tensors.append(padded_tensor)
    return torch.stack(padded_tensors), torch.stack(padding_masks)

def safe_np_array(data, default_val=-1):
    """Converts a list to a numpy array, replacing None with a default value."""
    return np.array([item if item is not None else default_val for item in data])