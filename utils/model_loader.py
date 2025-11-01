import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import os
from .model_downloader import get_model_path
from config import MODELS_DIR

def load_model_and_tokenizer(model_name, is_train_mode, progress_callback=None):
    try:
        token = os.environ.get("HF_TOKEN")
        if token:
            login(token=token)
    except Exception as e:
        if progress_callback:
            progress_callback(f"Could not log in to Hugging Face: {e}")

    model_path = get_model_path(model_name, MODELS_DIR, progress_callback)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    if is_train_mode:
        model.gradient_checkpointing_enable()

    return model, tokenizer