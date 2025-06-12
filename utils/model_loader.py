import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from transformers.utils import is_flash_attn_2_available
from config import MODELS_DIR

def get_local_models():
    """Scans the models directory and returns a list of available local models."""
    if not MODELS_DIR.is_dir():
        return []
    # Return subdirectories that are not the default BERT model
    return [
        p.name for p in MODELS_DIR.iterdir() 
        if p.is_dir() and p.name != 'bert-base-uncased'
    ]

def load_model_and_tokenizer(model_name_or_path, is_train_mode=True):
    """
    Loads a language model and its tokenizer, either from a local path or
    by downloading from the Hugging Face Hub. Applies 8-bit quantization.
    """
    local_path = MODELS_DIR / model_name_or_path
    path_to_load = local_path if local_path.exists() else model_name_or_path

    print(f"Loading tokenizer from: {path_to_load}")
    tokenizer = AutoTokenizer.from_pretrained(path_to_load, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer pad_token set to eos_token.")

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"

    print(f"Loading model '{path_to_load}' with 8-bit quantization...")
    print(f"Using compute_dtype='{compute_dtype}' and attn_implementation='{attn_implementation}'")
    
    model = AutoModelForCausalLM.from_pretrained(
        path_to_load,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        device_map='auto',
        torch_dtype=compute_dtype,
        attn_implementation=attn_implementation,
        # Trust remote code if the model is not local and requires it
        trust_remote_code=not local_path.exists() 
    )
    
    # Cache the model locally if it was downloaded
    if not local_path.exists() and is_train_mode:
        try:
            print(f"Caching downloaded model to {local_path}...")
            model.save_pretrained(local_path)
            tokenizer.save_pretrained(local_path)
            print("Model and tokenizer cached successfully.")
        except Exception as e:
            print(f"Warning: Could not cache the model. Error: {e}")

    return model, tokenizer