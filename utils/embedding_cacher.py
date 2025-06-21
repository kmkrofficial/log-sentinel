import torch
import os
import numpy as np
from pathlib import Path
from config import EMBEDDING_CACHE_DIR # Import new central cache dir

class EmbeddingCacher:
    def __init__(self, encoder_name, dataset_path):
        """
        Initializes the cacher with the components that make an embedding unique.
        """
        self.dataset_path = Path(dataset_path)
        
        self.encoder_name_sanitized = encoder_name.replace('/', '__')
        
        # --- FIX: Use the new centralized cache directory ---
        self.cache_dir = EMBEDDING_CACHE_DIR
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create a unique filename based on dataset and encoder
        dataset_name = self.dataset_path.parent.name
        file_stem = self.dataset_path.stem
        self.cache_file_path = self.cache_dir / f"{dataset_name}_{file_stem}_{self.encoder_name_sanitized}.pt"

    def load_embeddings(self):
        """
        Loads embeddings and labels from the cache if it exists.
        Returns a tuple of (embeddings, labels) or (None, None) if not found.
        """
        if self.cache_file_path.exists():
            print(f"Loading cached embeddings from: {self.cache_file_path}")
            try:
                data = torch.load(self.cache_file_path, weights_only=False)
                return data.get('embeddings'), data.get('labels')
            except Exception as e:
                print(f"Warning: Could not load cache file. It might be corrupt. Error: {e}")
                return None, None
        return None, None

    def save_embeddings(self, embeddings, labels):
        """
        Saves the generated embeddings and labels to the cache file.
        """
        print(f"Saving embeddings to cache: {self.cache_file_path}")
        try:
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels)

            torch.save({'embeddings': embeddings, 'labels': labels}, self.cache_file_path)
            print("Cache saved successfully.")
        except Exception as e:
            print(f"Error: Could not save cache file. Error: {e}")