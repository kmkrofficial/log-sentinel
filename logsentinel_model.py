import torch
import os
from torch import nn
from peft import PeftModel, LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoModelForCausalLM
from utils.model_loader import load_model_and_tokenizer
from utils.helpers import stack_and_pad_left
import traceback
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*You passed `quantization_config`.*")

class LogSentinelModel(nn.Module):
    def __init__(self, llama_path, encoder_hidden_size, ft_path=None, is_train_mode=True, device=None):
        super().__init__()
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.llama_model, self.llama_tokenizer = load_model_and_tokenizer(llama_path, is_train_mode)

        projector_device = self.llama_model.device
        compute_dtype = self.llama_model.dtype
        llama_hidden_size = self.llama_model.config.hidden_size

        self.projector = nn.Sequential(
            nn.Linear(encoder_hidden_size, llama_hidden_size),
            nn.GELU(),
            nn.Linear(llama_hidden_size, llama_hidden_size)
        ).to(projector_device).to(compute_dtype)

        self.classifier = nn.Linear(llama_hidden_size, 2).to(projector_device).to(compute_dtype)

        self.instruc_tokens = self.llama_tokenizer(
            ['Below is a sequence of system log messages:'],
            return_tensors="pt", padding=True
        ).to(projector_device)
        
        self._setup_peft(ft_path, is_train_mode)

    def _setup_peft(self, ft_path, is_train_mode):
        try:
            if ft_path and os.path.exists(os.path.join(ft_path, 'Llama_ft', 'adapter_config.json')):
                self._log(f"Found existing adapter at {ft_path}. Loading...")
                self.llama_model = PeftModel.from_pretrained(self.llama_model, os.path.join(ft_path, 'Llama_ft'), is_trainable=is_train_mode)
                self._log("PEFT adapter loaded successfully.")
                
                projector_path = os.path.join(ft_path, 'projector.pt')
                classifier_path = os.path.join(ft_path, 'classifier.pt')
                if os.path.exists(projector_path):
                    self.projector.load_state_dict(torch.load(projector_path, map_location=self.device))
                    self._log("Projector weights loaded.")
                if os.path.exists(classifier_path):
                    self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
                    self._log("Classifier weights loaded.")

            elif is_train_mode:
                self._log("No adapter found. Creating new PEFT configuration for training.")
                if getattr(self.llama_model, "is_loaded_in_8bit", False) or getattr(self.llama_model, "is_loaded_in_4bit", False):
                    # --- DEFINITIVE FIX: Disable gradient checkpointing to accelerate training time ---
                    # This increases VRAM usage during training but makes the forward pass much faster.
                    # It has NO impact on inference speed or memory.
                    self.llama_model = prepare_model_for_kbit_training(self.llama_model, use_gradient_checkpointing=False)
                    self._log("Gradient checkpointing disabled for faster training.")

                lora_config = LoraConfig(r=64, lora_alpha=64, lora_dropout=0.1, target_modules=["q_proj", "v_proj"], bias="none", task_type=TaskType.CAUSAL_LM)
                self.llama_model = get_peft_model(self.llama_model, lora_config)
                self.llama_model.print_trainable_parameters()
            else:
                self._log("Warning: Evaluation mode selected but no fine-tuned adapter path was provided. Using base model only.")
        except Exception as e:
            self._log(f"FATAL: An error occurred during PEFT setup in LogSentinelModel.")
            self._log(traceback.format_exc())
            raise e

    def _log(self, message):
        print(message)

    def save_ft_model(self, path):
        os.makedirs(path, exist_ok=True)
        self.llama_model.save_pretrained(os.path.join(path, 'Llama_ft'))
        torch.save(self.projector.state_dict(), os.path.join(path, 'projector.pt'))
        torch.save(self.classifier.state_dict(), os.path.join(path, 'classifier.pt'))
        self._log(f"Fine-tuned adapter and components saved to {path}")

    def _set_trainable(self, **kwargs):
        for name, param in self.named_parameters():
            param.requires_grad = False
            if 'projector' in name and kwargs.get('projector'): param.requires_grad = True
            elif 'classifier' in name and kwargs.get('classifier'): param.requires_grad = True
            elif 'llama_model' in name and 'lora_' in name and kwargs.get('llama_lora'): param.requires_grad = True

    def set_train_projector_and_classifier(self): self._set_trainable(projector=True, classifier=True)
    def set_finetuning_all(self): self._set_trainable(projector=True, classifier=True, lora_lora=True)
    
    def get_logits(self, precomputed_embeddings):
        if not precomputed_embeddings: return None, None

        projector_dtype = next(self.projector.parameters()).dtype
        projected_embeddings = [self.projector(emb.to(self.device).to(projector_dtype)) for emb in precomputed_embeddings]
        
        seq_embeddings = [emb.to(self.llama_model.dtype) for emb in projected_embeddings]
        
        embed_layer = self.llama_model.get_input_embeddings(); instruc_embeds = embed_layer(self.instruc_tokens['input_ids'])
        valid_embeddings, original_indices = [], []
        for i, seq_embed in enumerate(seq_embeddings):
            if seq_embed is not None and seq_embed.shape[0] > 0:
                full_embed = torch.cat([instruc_embeds[0], seq_embed], dim=0); valid_embeddings.append(full_embed); original_indices.append(i)
        
        if not valid_embeddings: return None, None
        
        inputs_embeds, attention_mask = stack_and_pad_left(valid_embeddings)
        outputs = self.llama_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True)
        
        if hasattr(outputs, 'last_hidden_state'): last_hidden_state = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states'): last_hidden_state = outputs.hidden_states[-1]
        else: raise AttributeError("Model output does not contain 'last_hidden_state' or 'hidden_states'.")
        
        sequence_lengths = attention_mask.sum(dim=1) - 1; batch_indices = torch.arange(len(valid_embeddings), device=last_hidden_state.device)
        cls_input_hidden_state = last_hidden_state[batch_indices, sequence_lengths]
        classifier_dtype = next(self.classifier.parameters()).dtype; logits = self.classifier(cls_input_hidden_state.to(classifier_dtype))
        return logits, original_indices

    def train_helper(self, labels, precomputed_embeddings):
        self.train()
        logits, original_indices = self.get_logits(precomputed_embeddings=precomputed_embeddings)
        if logits is None: return torch.tensor([]), torch.tensor([])
        valid_str_labels = [labels[i] for i in original_indices]
        integer_labels = torch.tensor([1 if lbl == 'anomalous' else 0 for lbl in valid_str_labels], dtype=torch.long, device=logits.device)
        return logits, integer_labels