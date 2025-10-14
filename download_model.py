import os
from pathlib import Path
from huggingface_hub import snapshot_download

# Define the root directory for models, relative to this script's location
MODELS_DIR = Path(__file__).resolve().parent / 'models'

# List of Hugging Face model repository IDs to download
MODELS_TO_DOWNLOAD = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "meta-llama/Llama-3.2-1B"
]

def download_model(model_id: str, target_dir: Path):
    """
    Downloads a model from the Hugging Face Hub to a specified directory.

    Args:
        model_id (str): The repository ID of the model (e.g., "user/model-name").
        target_dir (Path): The directory where the model should be saved.
    """
    # Create a specific subdirectory for the model to keep files organized
    # Hugging Face repo IDs can contain '/', which should be part of the path
    model_path = target_dir / model_id
    
    # Check if the model directory already exists and is not empty
    if model_path.exists() and any(model_path.iterdir()):
        print(f"✅ Model '{model_id}' already exists in '{model_path}'. Skipping.")
        return

    print(f"⏳ Downloading model '{model_id}' to '{model_path}'...")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(model_path),
            resume_download=True,
            # For Windows compatibility and to avoid symlink issues
            local_dir_use_symlinks=False 
        )
        print(f"✅ Successfully downloaded '{model_id}'.")
    except Exception as e:
        print(f"❌ Failed to download '{model_id}'. Error: {e}")

if __name__ == "__main__":
    print("--- Starting Model Download Script ---")
    
    # Ensure the main models directory exists
    MODELS_DIR.mkdir(exist_ok=True)
    
    for model in MODELS_TO_DOWNLOAD:
        download_model(model, MODELS_DIR)
        
    print("\n--- Model download process finished. ---")
    print("You can now run the training or inference scripts.")