# --- START OF FILE model.py ---
# (Keep imports and other functions as they were)
import os.path
import peft
import torch
from transformers import BertTokenizerFast, BertModel, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from torch import nn
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model, TaskType

# --- Helper Functions (merge_data, stack_and_pad_right, stack_and_pad_left) remain the same ---
def merge_data(data):
    merged_data = []
    start_positions = []
    current_position = 0
    for sublist in data:
        start_positions.append(current_position)
        merged_data.extend(sublist)
        current_position += len(sublist)
    return merged_data, start_positions

def stack_and_pad_right(tensors):
    max_len = max(tensor.shape[0] for tensor in tensors)
    padded_tensors = []
    padding_masks = []
    for tensor in tensors:
        pad_len = max_len - tensor.shape[0]
        padded_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_len))
        padded_tensors.append(padded_tensor)
        padding_mask = torch.cat([torch.ones(tensor.shape[0], dtype=torch.long),
                                  torch.zeros(pad_len, dtype=torch.long)])
        padding_masks.append(padding_mask)
    stacked_tensor = torch.stack(padded_tensors)
    padding_masks = torch.stack(padding_masks)
    return stacked_tensor, padding_masks

def stack_and_pad_left(tensors):
    max_len = max(tensor.shape[0] for tensor in tensors)
    padded_tensors = []
    padding_masks = []
    for tensor in tensors:
        pad_len = max_len - tensor.shape[0]
        padded_tensor = torch.nn.functional.pad(tensor, (0, 0, pad_len, 0))
        padded_tensors.append(padded_tensor)
        padding_mask = torch.cat([torch.zeros(pad_len, dtype=torch.long),
                                 torch.ones(tensor.shape[0], dtype=torch.long)])
        padding_masks.append(padding_mask)
    stacked_tensor = torch.stack(padded_tensors)
    padding_masks = torch.stack(padding_masks)
    return stacked_tensor, padding_masks


# --- Quantization Config ---
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

class LogLLM(nn.Module):
    def __init__(self, Bert_path, Llama_path, ft_path=None, is_train_mode=True, device = torch.device("cuda:0"), max_content_len = 128, max_seq_len = 128):
        super().__init__()
        self.max_content_len = max_content_len
        self.max_seq_len = max_seq_len
        self.device = device # Primary device (e.g., cuda:0)

        # --- Llama Setup ---
        self.Llama_tokenizer = AutoTokenizer.from_pretrained(Llama_path, padding_side="right")
        if self.Llama_tokenizer.pad_token is None:
             print("Warning: Llama tokenizer does not have a pad token. Setting to EOS token.")
             self.Llama_tokenizer.pad_token = self.Llama_tokenizer.eos_token
        print(f"Llama tokenizer pad token ID: {self.Llama_tokenizer.pad_token_id}")

        print("Loading Llama model with 8-bit quantization...")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Using compute dtype: {compute_dtype}")
        self.Llama_model = AutoModelForCausalLM.from_pretrained(
            Llama_path,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            device_map='auto',
            torch_dtype=compute_dtype
        )
        print(f"Llama model loaded on device(s): {self.Llama_model.device}")
        # --- End Llama Setup ---

        # --- BERT Setup ---
        print("Loading BERT model...")
        self.Bert_tokenizer = BertTokenizerFast.from_pretrained(Bert_path, do_lower_case=True)
        self.Bert_model = BertModel.from_pretrained(
            Bert_path,
            low_cpu_mem_usage=True,
        ).to(device)
        print(f"BERT model loaded on device: {self.Bert_model.device}")
        # --- End BERT Setup ---

        # --- Projector Setup (MLP Implementation) ---
        projector_device = self.Llama_model.device
        if str(projector_device) == 'cpu' or str(projector_device) == 'meta':
             projector_device = device

        print(f"Initializing MLP Projector on device: {projector_device}")
        bert_hidden_size = self.Bert_model.config.hidden_size
        llama_hidden_size = self.Llama_model.config.hidden_size
        projection_intermediate_dim = llama_hidden_size

        # Define projector using compute_dtype from the start
        self.projector = nn.Sequential(
            nn.Linear(bert_hidden_size, projection_intermediate_dim),
            nn.GELU(),
            nn.Linear(projection_intermediate_dim, llama_hidden_size)
        ).to(projector_device).to(compute_dtype) # Apply dtype during creation/move
        print(f"MLP Projector created with intermediate dim {projection_intermediate_dim}, dtype {compute_dtype}")
        # --- End Projector Setup ---

        # --- Instruction Tokens ---
        self.instruc_tokens = self.Llama_tokenizer(
            ['Below is a sequence of system log messages:', '. Is this sequence normal or anomalous? \\n'],
            return_tensors="pt", padding=True).to(projector_device)
        # --- End Instruction Tokens ---

        # --- PEFT Loading / Creation ---
        # (This section remains the same as previous version)
        if ft_path is not None:
            print(f'Loading peft model from {ft_path}.')
            Llama_ft_path = os.path.join(ft_path, 'Llama_ft')
            Bert_ft_path = os.path.join(ft_path, 'Bert_ft') # BERT LoRA path (if used)
            projector_path = os.path.join(ft_path, 'projector.pt')

            try:
                self.Llama_model = PeftModel.from_pretrained(
                    self.Llama_model, Llama_ft_path, is_trainable=is_train_mode)
                print("Llama PEFT adapters loaded.")
            except Exception as e:
                print(f"Could not load Llama PEFT adapters from {Llama_ft_path}: {e}. Using base Llama.")

            if os.path.exists(Bert_ft_path) and isinstance(self.Bert_model, BertModel):
                 try:
                     self.Bert_model = PeftModel.from_pretrained(
                        self.Bert_model, Bert_ft_path, is_trainable=is_train_mode)
                     print("BERT PEFT adapters loaded.")
                 except Exception as e:
                     print(f"Could not load BERT PEFT adapters from {Bert_ft_path}: {e}. Using base BERT.")
            elif os.path.exists(Bert_ft_path):
                 print(f"BERT PEFT path exists ({Bert_ft_path}), but BERT model is already PEFT? Skipping load.")
            else:
                 print(f"BERT PEFT model not found at {Bert_ft_path}, using base BERT.")

            if os.path.exists(projector_path):
                try:
                    state_dict = torch.load(projector_path, map_location=projector_device)
                    self.projector.load_state_dict(state_dict)
                    print("Projector state dict loaded.")
                except Exception as e:
                    print(f"Could not load projector state dict from {projector_path}: {e}. Using initialized projector.")
            else:
                print(f"Projector state dict not found at {projector_path}, using initialized projector.")
        else:
            print(f'Creating new peft model for Llama.')
            Llama_peft_config = LoraConfig(
                r=8, lora_alpha=16, lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],
                bias="none", task_type=TaskType.CAUSAL_LM
            )
            self.Llama_model = get_peft_model(self.Llama_model, Llama_peft_config)
            print("Llama model wrapped with PEFT LoRA.")
            self.Llama_model.print_trainable_parameters()
        # --- End PEFT Loading / Creation ---

    # --- save_ft_model, set_train_* methods remain the same ---
    def save_ft_model(self, path):
        if not os.path.exists(path): os.makedirs(path)
        Llama_ft_path = os.path.join(path,'Llama_ft')
        Bert_ft_path = os.path.join(path,'Bert_ft')
        projector_path = os.path.join(path,'projector.pt')
        print(f"Saving Llama PEFT model to {Llama_ft_path}")
        self.Llama_model.save_pretrained(Llama_ft_path)
        if isinstance(self.Bert_model, PeftModel):
             print(f"Saving Bert PEFT model to {Bert_ft_path}")
             self.Bert_model.save_pretrained(Bert_ft_path)
        else: print("BERT model is not a PEFT model, skipping save.")
        print(f"Saving projector state dict to {projector_path}")
        torch.save(self.projector.state_dict(), projector_path)

    def set_train_only_projector(self):
        print("Setting trainable: Projector ONLY")
        for param in self.projector.parameters(): param.requires_grad = True
        for param in self.Bert_model.parameters(): param.requires_grad = False
        for param in self.Llama_model.parameters(): param.requires_grad = False

    def set_train_only_Llama(self):
        print("Setting trainable: Llama LoRA adapters ONLY")
        for param in self.projector.parameters(): param.requires_grad = False
        for param in self.Bert_model.parameters(): param.requires_grad = False
        for name, param in self.Llama_model.named_parameters():
            if 'lora_' in name: param.requires_grad = True
            else: param.requires_grad = False

    def set_train_projectorAndBert(self):
        print("Setting trainable: Projector and BERT LoRA adapters")
        for param in self.projector.parameters(): param.requires_grad = True
        for name, param in self.Bert_model.named_parameters():
            if isinstance(self.Bert_model, PeftModel) and 'lora_' in name: param.requires_grad = True
            else: param.requires_grad = False
        for param in self.Llama_model.parameters(): param.requires_grad = False

    def set_finetuning_all(self):
        print("Setting trainable: Projector, BERT LoRA, and Llama LoRA adapters")
        for param in self.projector.parameters(): param.requires_grad = True
        for name, param in self.Bert_model.named_parameters():
            if isinstance(self.Bert_model, PeftModel) and 'lora_' in name: param.requires_grad = True
            else: param.requires_grad = False
        for name, param in self.Llama_model.named_parameters():
            if 'lora_' in name: param.requires_grad = True
            else: param.requires_grad = False


    # --- train_helper method ---
    def train_helper(self, sequences_, labels):
        '''
        :param sequences: list of list: [seq, seq, ...,seq]  , seq:[item, ..., item]
        :param labels:  list of labels, label is one of ['anomalous', 'normal']
        :return: Llama_output[label_mask], target_tokens_ids[target_tokens_atts]
        '''
        sequences = [sequence[:self.max_seq_len] for sequence in sequences_]
        batch_size = len(sequences)
        data, seq_positions = merge_data(sequences)
        if not data:
             output_device = self.Llama_model.device
             return torch.tensor([], device=output_device), torch.tensor([], device=output_device)
        seq_positions = seq_positions[1:]

        # BERT encoding (on self.device)
        inputs = self.Bert_tokenizer(data, return_tensors="pt", max_length=self.max_content_len, padding=True,
                                     truncation=True).to(self.device)
        bert_outputs = self.Bert_model(**inputs).pooler_output
        # --- DTYPE FIX START ---
        # REMOVE the explicit .float() cast here:
        # bert_outputs = bert_outputs.float()

        # Projector (on projector_device)
        projector_input_device = next(self.projector.parameters()).device
        # Determine the dtype expected by the projector's first layer
        projector_dtype = next(self.projector.parameters()).dtype
        # CAST bert_outputs to the projector's device AND dtype BEFORE passing it in
        projected_outputs = self.projector(bert_outputs.to(device=projector_input_device, dtype=projector_dtype))
        # --- DTYPE FIX END ---


        # Determine Llama embedding layer's device and dtype
        try:
            llama_embed_device = self.Llama_model.get_input_embeddings().weight.device
            llama_embed_dtype = self.Llama_model.get_input_embeddings().weight.dtype
        except AttributeError:
             print("Warning: Could not determine Llama embedding layer device/dtype automatically.")
             llama_embed_device = projector_input_device
             llama_embed_dtype = projector_dtype # Use projector's dtype as fallback

        # Cast projector output to Llama's expected embedding dtype and device
        projected_outputs = projected_outputs.to(dtype=llama_embed_dtype, device=llama_embed_device)

        # --- Rest of the function remains the same ---
        if projected_outputs.shape[0] > 0:
            if seq_positions: seq_embeddings = torch.tensor_split(projected_outputs, seq_positions)
            else: seq_embeddings = [projected_outputs]
        else: seq_embeddings = []

        prefix = "The sequence is "
        labels_str = [str(lbl) for lbl in labels]
        full_labels = [f"{prefix}{lbl}." for lbl in labels_str]
        answer_tokens = self.Llama_tokenizer(full_labels, padding=True, return_tensors="pt").to(llama_embed_device)

        target_tokens_ids = answer_tokens['input_ids'][:, 1:].contiguous()
        target_tokens_atts = answer_tokens['attention_mask'][:, 1:].contiguous().bool()
        answer_tokens_ids = answer_tokens['input_ids'][:, 1:].contiguous()
        answer_tokens_atts = answer_tokens['attention_mask'][:, 1:].contiguous().bool()

        embed_layer = self.Llama_model.get_input_embeddings()
        instruc_tokens_dev = self.instruc_tokens['input_ids'].to(llama_embed_device)
        instruc_embeddings = embed_layer(instruc_tokens_dev)
        answer_embeddings = embed_layer(answer_tokens_ids)

        instruc_tokens_mask_dev = self.instruc_tokens['attention_mask'].to(llama_embed_device)
        ins1 = instruc_embeddings[0][instruc_tokens_mask_dev[0].bool()]
        ins2_mask = instruc_tokens_mask_dev[1].bool()
        is_bos = self.Llama_tokenizer.bos_token_id is not None and \
                 instruc_tokens_dev[1, 0] == self.Llama_tokenizer.bos_token_id
        if is_bos and ins2_mask.sum() > 0: ins2 = instruc_embeddings[1][ins2_mask][1:]
        elif ins2_mask.sum() > 0: ins2 = instruc_embeddings[1][ins2_mask]
        else: ins2 = torch.tensor([], device=llama_embed_device, dtype=llama_embed_dtype)

        embeddings = []
        num_target_tokens_per_seq = []
        for i in range(target_tokens_atts.shape[0]):
            seq_embedding = seq_embeddings[i] if i < len(seq_embeddings) else None
            answer_embedding = answer_embeddings[i]
            current_answer_tokens_att = answer_tokens_atts[i]
            current_target_tokens_att = target_tokens_atts[i]
            num_targets = current_target_tokens_att.sum().item()
            num_target_tokens_per_seq.append(num_targets)
            if seq_embedding is not None and seq_embedding.shape[0] > 0:
                 actual_answer_embedding = answer_embedding[current_answer_tokens_att]
                 full_prompt_embedding = torch.cat([ins1, seq_embedding, ins2, actual_answer_embedding], dim=0)
                 embeddings.append(full_prompt_embedding)
            else:
                 print(f"Warning: Encountered empty sequence embedding for batch index {i}.")
                 actual_answer_embedding = answer_embedding[current_answer_tokens_att]
                 full_prompt_embedding = torch.cat([ins1, ins2, actual_answer_embedding], dim=0)
                 embeddings.append(full_prompt_embedding)

        if not embeddings:
             output_device = self.Llama_model.device
             return torch.tensor([], device=output_device), torch.tensor([], device=output_device)

        inputs_embeds, attention_mask = stack_and_pad_left(embeddings)
        model_device = self.Llama_model.device
        inputs_embeds = inputs_embeds.to(model_device)
        attention_mask = attention_mask.to(model_device)

        label_mask = torch.zeros_like(attention_mask, dtype=torch.bool, device=model_device)
        for i in range(label_mask.shape[0]):
            num_targets = num_target_tokens_per_seq[i]
            if num_targets > 0:
                sequence_length = attention_mask[i].sum().item()
                if sequence_length >= num_targets:
                     start_logit_index = sequence_length - num_targets
                     end_logit_index = sequence_length
                     label_mask[i, start_logit_index:end_logit_index] = True

        Llama_output = self.Llama_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits

        masked_logits = Llama_output[label_mask]
        masked_targets = target_tokens_ids[target_tokens_atts]
        masked_targets = masked_targets.to(masked_logits.device)

        if masked_logits.shape[0] != masked_targets.shape[0]:
             print(f"FATAL Error: Mismatch between masked logits ({masked_logits.shape[0]}) and targets ({masked_targets.shape[0]}).")
             return torch.tensor([], device=masked_logits.device), torch.tensor([], device=masked_logits.device)

        return masked_logits, masked_targets


    # --- forward (inference) method ---
    def forward(self, sequences_):
        ''' Inference function. '''
        self.eval()
        sequences = [sequence[:self.max_seq_len] for sequence in sequences_]
        batch_size = len(sequences)
        data, seq_positions = merge_data(sequences)
        if not data:
            output_device = self.Llama_model.device
            return torch.tensor([[]] * batch_size, dtype=torch.long, device=output_device)
        seq_positions = seq_positions[1:]

        inputs = self.Bert_tokenizer(data, return_tensors="pt", max_length=self.max_content_len, padding=True,
                                     truncation=True).to(self.device)

        with torch.no_grad():
            bert_outputs = self.Bert_model(**inputs).pooler_output
            # --- DTYPE FIX FOR INFERENCE ---
            # REMOVE explicit .float() cast if it was here

            projector_input_device = next(self.projector.parameters()).device
            projector_dtype = next(self.projector.parameters()).dtype
            # CAST bert_outputs to projector's device and dtype
            projected_outputs = self.projector(bert_outputs.to(device=projector_input_device, dtype=projector_dtype))
            # --- END DTYPE FIX ---

            try:
                llama_embed_device = self.Llama_model.get_input_embeddings().weight.device
                llama_embed_dtype = self.Llama_model.get_input_embeddings().weight.dtype
            except AttributeError:
                print("Warning: Could not determine Llama embedding layer device/dtype automatically.")
                llama_embed_device = projector_input_device
                llama_embed_dtype = projector_dtype

            projected_outputs = projected_outputs.to(dtype=llama_embed_dtype, device=llama_embed_device)

            if projected_outputs.shape[0] > 0:
                if seq_positions: seq_embeddings = torch.tensor_split(projected_outputs, seq_positions)
                else: seq_embeddings = [projected_outputs]
            else: seq_embeddings = []

            prefix = "The sequence is"
            answer_prefix_tokens = self.Llama_tokenizer(prefix, return_tensors="pt", add_special_tokens=False)['input_ids'].to(llama_embed_device)

            embed_layer = self.Llama_model.get_input_embeddings()
            instruc_tokens_dev = self.instruc_tokens['input_ids'].to(llama_embed_device)
            instruc_embeddings = embed_layer(instruc_tokens_dev)
            answer_prefix_tokens_embeddings = embed_layer(answer_prefix_tokens)
            if answer_prefix_tokens_embeddings.dim() == 3 and answer_prefix_tokens_embeddings.shape[0] == 1:
                 answer_prefix_tokens_embeddings = answer_prefix_tokens_embeddings.squeeze(0)

            instruc_tokens_mask_dev = self.instruc_tokens['attention_mask'].to(llama_embed_device)
            ins1 = instruc_embeddings[0][instruc_tokens_mask_dev[0].bool()]
            ins2_mask = instruc_tokens_mask_dev[1].bool()
            is_bos = self.Llama_tokenizer.bos_token_id is not None and \
                     instruc_tokens_dev[1, 0] == self.Llama_tokenizer.bos_token_id
            if is_bos and ins2_mask.sum() > 0: ins2 = instruc_embeddings[1][ins2_mask][1:]
            elif ins2_mask.sum() > 0: ins2 = instruc_embeddings[1][ins2_mask]
            else: ins2 = torch.tensor([], device=llama_embed_device, dtype=llama_embed_dtype)

            prompt_embeddings_list = []
            for i in range(batch_size):
                 seq_embedding = seq_embeddings[i] if i < len(seq_embeddings) else None
                 if seq_embedding is not None and seq_embedding.shape[0] > 0:
                     prompt_embedding = torch.cat([ins1, seq_embedding, ins2, answer_prefix_tokens_embeddings], dim=0)
                     prompt_embeddings_list.append(prompt_embedding)
                 else:
                     print(f"Warning: Empty sequence embedding for batch index {i} during inference.")
                     prompt_embedding = torch.cat([ins1, ins2, answer_prefix_tokens_embeddings], dim=0)
                     prompt_embeddings_list.append(prompt_embedding)

            if not prompt_embeddings_list:
                 output_device = self.Llama_model.device
                 return torch.tensor([[]] * batch_size, dtype=torch.long, device=output_device)

            inputs_embeds, attention_mask = stack_and_pad_left(prompt_embeddings_list)
            model_device = self.Llama_model.device
            inputs_embeds = inputs_embeds.to(model_device)
            attention_mask = attention_mask.to(model_device)

            pad_token_id = self.Llama_tokenizer.pad_token_id
            eos_token_id = self.Llama_tokenizer.eos_token_id

            outputs = self.Llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=10,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=False
            )

            prompt_length = inputs_embeds.shape[1]
            generated_ids = outputs[:, prompt_length:]

            return generated_ids