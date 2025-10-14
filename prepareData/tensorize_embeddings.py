import torch
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

def pad_and_stack_sequences(embeddings, max_len, pad_value=0):
    padded_tensors = []
    for emb_sequence in embeddings:
        if emb_sequence is None or len(emb_sequence) == 0:
            continue
        
        seq_len = emb_sequence.shape[0]
        if seq_len > max_len:
            # Truncate from the beginning (keeping most recent logs)
            padded_tensor = emb_sequence[seq_len-max_len:]
        else:
            # Pad on the left
            pad_width = max_len - seq_len
            padded_tensor = torch.nn.functional.pad(emb_sequence, (0, 0, pad_width, 0), 'constant', pad_value)
        
        padded_tensors.append(padded_tensor)
        
    if not padded_tensors:
        return None
    return torch.stack(padded_tensors)

def tensorize_dataset(embedding_cache_path, max_seq_len):
    if not embedding_cache_path.exists():
        print(f"Embedding cache file not found: {embedding_cache_path}")
        return

    print(f"Loading cached embeddings from: {embedding_cache_path}")
    try:
        data = torch.load(embedding_cache_path, weights_only=False)
        embeddings, labels = data.get('embeddings'), data.get('labels')
    except Exception as e:
        print(f"Warning: Could not load cache file. Error: {e}")
        return

    if not embeddings or labels is None:
        print("No embeddings or labels found in the cache file.")
        return

    print(f"Tensorizing {len(embeddings)} sequences to a fixed length of {max_seq_len}...")
    
    tensorized_sequences = pad_and_stack_sequences(embeddings, max_seq_len)
    
    if tensorized_sequences is None:
        print("No valid sequences to tensorize.")
        return

    output_path = embedding_cache_path.with_suffix('.tensor.pt')
    
    print(f"Saving tensorized dataset to: {output_path}")
    torch.save({
        'sequences': tensorized_sequences.to(torch.float16), # Save as float16 to save space
        'labels': labels
    }, output_path)
    
    print("Tensorization complete.")

if __name__ == '__main__':
    # This part allows for manual execution if needed
    # Example usage:
    # python prepareData/tensorize_embeddings.py path/to/your/embedding_cache.pt 128
    import sys
    if len(sys.argv) != 3:
        print("Usage: python tensorize_embeddings.py <path_to_embedding_cache> <max_seq_len>")
    else:
        cache_path = Path(sys.argv[1])
        max_len = int(sys.argv[2])
        tensorize_dataset(cache_path, max_len)

