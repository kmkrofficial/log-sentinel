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
    if not MODELS_DIR.is_dir():
        return []
    return [
        p.name for p in MODELS_DIR.iterdir() 
        if p.is_dir() and p.name != 'bert-base-uncased'
    ]

def load_model_and_tokenizer(model_name_or_path, is_train_mode=True):
    local_path = MODELS_DIR / model_name_or_path
    path_to_load = local_path if local_path.exists() else model_name_or_path

    print(f"Loading tokenizer from: {path_to_load}")
    tokenizer = AutoTokenizer.from_pretrained(path_to_load, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer pad_token set to eos_token.")

    # --- FIX: Force float16 for stable Windows compatibility ---
    # bfloat16 has poor support in bitsandbytes on Windows. 
    # float16 is the reliable choice for half-precision.
    compute_dtype = torch.float16
    print(f"Using compute data type: {compute_dtype} for Windows compatibility.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype
    )
    
    attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
    print(f"Using attention implementation: {attn_implementation}")
    if attn_implementation == "flash_attention_2":
        print("Flash Attention 2 will be used for faster and more memory-efficient inference.")
    else:
        print("Flash Attention 2 not available. Falling back to default 'sdpa' implementation.")

    print(f"Loading model '{path_to_load}' with 4-bit quantization...")
    print(f"Model computations will be performed in '{compute_dtype}'.")
    
    model = AutoModelForCausalLM.from_pretrained(
        path_to_load,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        device_map='auto',
        torch_dtype=compute_dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=not local_path.exists() 
    )
    
    if not local_path.exists() and is_train_mode:
        try:
            print(f"Caching downloaded model to {local_path}...")
            model.save_pretrained(local_path)
            tokenizer.save_pretrained(local_path)
            print("Model and tokenizer cached successfully.")
        except Exception as e:
            print(f"Warning: Could not cache the model. Error: {e}")

    return model, tokenizer