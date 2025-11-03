import torch
import os
from torch import nn
from peft import PeftModel, LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM
from utils.model_loader import load_model_and_tokenizer
import traceback
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*You passed `quantization_config`.*")

class LogSentinelModel(nn.Module):
    def __init__(self, llama_model_path, encoder_hidden_size, hyperparameters, ft_path=None, is_train_mode=True, device=None, log_callback=None):
        super().__init__()
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hp = hyperparameters
        self.log_callback = log_callback or print

        self.llama_model, self.llama_tokenizer = load_model_and_tokenizer(llama_model_path, is_train_mode, self.log_callback)

        projector_device = self.llama_model.device
        compute_dtype = self.llama_model.dtype
        llama_hidden_size = self.llama_model.config.hidden_size

        self.projector = nn.Sequential(
            nn.Linear(encoder_hidden_size, llama_hidden_size),
            nn.GELU(),
            nn.Linear(llama_hidden_size, llama_hidden_size)
        ).to(projector_device).to(compute_dtype)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(llama_hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 2)
        ).to(projector_device).to(compute_dtype)

        self.instruc_tokens = self.llama_tokenizer(
            ['Below is a sequence of system log messages:'],
            return_tensors="pt", padding=True
        ).to(projector_device)

        self._setup_peft(ft_path, is_train_mode)

        if ft_path:
            self.load_ft_model(ft_path)

    def load_ft_model(self, path):
        projector_path = os.path.join(path, 'projector.pt')
        classifier_path = os.path.join(path, 'classifier.pt')
        if os.path.exists(projector_path):
            self._log(f"Loading projector weights from {projector_path}")
            self.projector.load_state_dict(torch.load(projector_path, map_location=self.llama_model.device))
        if os.path.exists(classifier_path):
            self._log(f"Loading classifier weights from {classifier_path}")
            self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.llama_model.device))

    def _setup_peft(self, ft_path, is_train_mode):
        try:
            if ft_path and os.path.exists(os.path.join(ft_path, 'Llama_ft', 'adapter_config.json')):
                self._log(f"Found existing adapter at {ft_path}. Loading...")
                self.llama_model = PeftModel.from_pretrained(self.llama_model, os.path.join(ft_path, 'Llama_ft'), is_trainable=is_train_mode)
            elif is_train_mode:
                self._log("No adapter found. Creating new PEFT configuration for training.")
                self.llama_model = prepare_model_for_kbit_training(self.llama_model, use_gradient_checkpointing=False)

                lora_rank = self.hp.get('lora_r', 64)
                self._log(f"Using LoRA rank (r): {lora_rank}")
                lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_rank, lora_dropout=0.1, target_modules=["q_proj", "v_proj"], bias="none", task_type=TaskType.CAUSAL_LM)
                self.llama_model = get_peft_model(self.llama_model, lora_config)
                self.llama_model.print_trainable_parameters()
        except Exception as e:
            self._log(f"FATAL: An error occurred during PEFT setup: {e}\n{traceback.format_exc()}")
            raise e

    def _log(self, message):
        self.log_callback(message)

    def save_ft_model(self, path):
        os.makedirs(path, exist_ok=True)
        self.llama_model.save_pretrained(os.path.join(path, 'Llama_ft'))
        torch.save(self.projector.state_dict(), os.path.join(path, 'projector.pt'))
        torch.save(self.classifier.state_dict(), os.path.join(path, 'classifier.pt'))
        self._log(f"Fine-tuned components saved to {path}")

    def set_trainable(self, **kwargs):
        for name, param in self.named_parameters():
            is_lora = 'lora_' in name
            is_projector = 'projector' in name
            is_classifier = 'classifier' in name

            if (is_lora and kwargs.get('llama_lora')) or \
               (is_projector and kwargs.get('projector')) or \
               (is_classifier and kwargs.get('classifier')):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def set_train_projector_and_classifier(self): self.set_trainable(projector=True, classifier=True)
    def set_finetuning_all(self): self.set_trainable(projector=True, classifier=True, llama_lora=True)

    def get_logits(self, sequence_tensor_batch):
        batch_size = sequence_tensor_batch.shape[0]

        projector_dtype = next(self.projector.parameters()).dtype
        projected_batch = self.projector(sequence_tensor_batch.to(projector_dtype))

        embed_layer = self.llama_model.get_input_embeddings()
        instruc_embeds = embed_layer(self.instruc_tokens['input_ids']).expand(batch_size, -1, -1)

        inputs_embeds = torch.cat([instruc_embeds, projected_batch], dim=1)

        attention_mask = torch.ones(inputs_embeds.shape[:2], device=self.device, dtype=torch.long)
        
        # Get the core transformer model, bypassing the CausalLM head
        base_model = self.llama_model.model.model
        
        outputs = base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )

        last_hidden_state = outputs.last_hidden_state

        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(batch_size, device=last_hidden_state.device)
        cls_input_hidden_state = last_hidden_state[batch_indices, sequence_lengths]

        classifier_dtype = next(self.classifier.parameters()).dtype
        logits = self.classifier(cls_input_hidden_state.to(classifier_dtype))

        return logits, batch_indices