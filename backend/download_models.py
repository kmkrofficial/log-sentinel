import os
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi
from transformers import logging as hf_logging
import sys

hf_logging.set_verbosity_error()

sys.path.append(str(Path(__file__).resolve().parent))
try:
    from config import MODELS_DIR, DEFAULT_ENCODER_MODEL, DEFAULT_LLAMA_MODEL
except ImportError:
    print("Error: Could not import config.py.")
    print("Please make sure this script is in the root directory of your project.")
    MODELS_DIR = Path(__file__).resolve().parent / 'models'
    DEFAULT_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_LLAMA_MODEL = "meta-llama/Llama-3.2-1B"


MODELS_TO_DOWNLOAD = [
    DEFAULT_ENCODER_MODEL,
    DEFAULT_LLAMA_MODEL
]

def check_hf_auth():
    try:
        api = HfApi()
        api.whoami()
        print("Hugging Face authentication successful.")
        return True
    except Exception as e:
        print(f"Hugging Face authentication failed: {e}")
        print("\n--- PLEASE READ ---")
        print("Llama-3.2-1B is a gated model. You MUST be logged in to download it.")
        print("Please run 'huggingface-cli login' in your terminal, then run this script again.")
        print("--- PLEASE READ ---\n")
        return False

def ensure_model_downloaded(model_name: str, target_dir: Path):
    target_path = target_dir / model_name.split('/')[-1]
    
    if target_path.exists() and any(target_path.iterdir()):
        print(f"Model '{model_name}' already cached at {target_path}")
        return str(target_path)

    print(f"Downloading '{model_name}' to {target_path}...")
    
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=target_path,
            local_dir_use_symlinks=False
        )
    except Exception as e:
        print(f"\nFailed to download {model_name}. Error: {e}\n")
        return None
    
    print(f"Download complete for '{model_name}'.")
    return str(target_path)

def main():
    print("--- LogSentinel Model Downloader ---")
    
    if not check_hf_auth():
        return

    MODELS_DIR.mkdir(exist_ok=True)
    
    print(f"Ensuring all models are downloaded to {MODELS_DIR}...")
    
    for model in MODELS_TO_DOWNLOAD:
        ensure_model_downloaded(model, MODELS_DIR)
        
    print("--- Model download check complete. ---")

if __name__ == "__main__":
    main()