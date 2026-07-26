import torch
from mlcore.utils.runtime_compat import configure_bitsandbytes_cuda

configure_bitsandbytes_cuda(torch.version.cuda)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

def load_model_and_tokenizer(model_path, is_train_mode, progress_callback=None):

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
        attn_implementation="flash_attention_3"
    )

    model.config.use_cache = False
    
    if is_train_mode:
        model.gradient_checkpointing_enable()

    return model, tokenizer