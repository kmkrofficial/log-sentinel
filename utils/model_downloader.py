import os
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

def ensure_model_downloaded(model_name: str, target_dir: Path, progress_callback=None):
    if target_dir.exists() and any(target_dir.iterdir()):
        if progress_callback:
            progress_callback(f"Model '{model_name}' already cached at {target_dir}")
        return str(target_dir)

    if progress_callback:
        progress_callback(f"Downloading '{model_name}' to {target_dir}...")
    
    snapshot_download(
        repo_id=model_name,
        local_dir=target_dir,
        local_dir_use_symlinks=False
    )
    
    if progress_callback:
        progress_callback(f"Download complete for '{model_name}'.")
    
    return str(target_dir)

def get_model_path(model_name: str, models_dir: Path, progress_callback=None):
    target_dir = models_dir / model_name.split('/')[-1]
    return ensure_model_downloaded(model_name, target_dir, progress_callback)