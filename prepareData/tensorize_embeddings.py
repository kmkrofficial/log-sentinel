import torch
import os
from tqdm import tqdm
import pickle
from pathlib import Path

def tensorize_dataset(cache_file_path: Path, max_seq_len: int):
    tensor_file_path = cache_file_path.with_suffix('.tensor.pt')
    
    if tensor_file_path.exists():
        print(f"Tensor file {tensor_file_path} already exists. Skipping tensorization.")
        return

    print(f"Loading embeddings from {cache_file_path}...")
    with open(cache_file_path, 'rb') as f:
        data = pickle.load(f)
        
    all_embeddings = data['embeddings']
    all_labels = data['labels']
    
    print(f"Loaded {len(all_embeddings)} sequences. Tensorizing...")
    
    tensorized_sequences = []
    tensorized_labels = []

    if not all_embeddings:
        print("Error: No embeddings found in cache file.")
        return

    sample_embedding_dim = all_embeddings[0].shape[1]

    for i in tqdm(range(len(all_embeddings)), desc="Tensorizing sequences"):
        seq_embeddings = all_embeddings[i]
        seq_label = all_labels[i]
        
        if seq_embeddings is None or seq_embeddings.shape[0] == 0:
            continue
            
        seq_len = seq_embeddings.shape[0]

        if seq_len > max_seq_len:
            tensorized_seq = seq_embeddings[-max_seq_len:]
        elif seq_len < max_seq_len:
            padding_len = max_seq_len - seq_len
            padding = torch.zeros(padding_len, sample_embedding_dim, dtype=torch.float32)
            tensorized_seq = torch.cat([seq_embeddings, padding], dim=0)
        else:
            tensorized_seq = seq_embeddings
            
        tensorized_sequences.append(tensorized_seq)
        tensorized_labels.append(torch.tensor(seq_label, dtype=torch.long))

    if not tensorized_sequences:
        print("Error: No sequences were tensorized. Check data and max_seq_len.")
        return

    sequences_tensor = torch.stack(tensorized_sequences)
    labels_tensor = torch.stack(tensorized_labels)
    
    torch.save({
        'sequences': sequences_tensor,
        'labels': labels_tensor
    }, tensor_file_path)
    
    print(f"Tensorized data saved to {tensor_file_path}")