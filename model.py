# --- START OF FILE model.py ---
import os.path
import peft
import torch
from transformers import BertTokenizerFast, BertModel, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from torch import nn
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model, TaskType
# Import for Flash Attention check
from transformers.utils import is_flash_attn_2_available

# --- Helper Functions ---

# Renamed merge_data to merge_data_debug and added prints
def merge_data_debug(data):
    merged_data = []
    start_positions = []
    current_position = 0
    # print(f"  [merge_data_debug] Input data length: {len(data)}") # Optional: uncomment for verbose debug
    for i, sublist in enumerate(data):
        # Defensive check for sublist type
        if not isinstance(sublist, (list, tuple)): # Add other sequence types if needed
             print(f"  [merge_data_debug] WARNING: Item {i} is not a list/tuple: type={type(sublist)}, value={sublist}")
             start_positions.append(current_position) # Still record the start position
             continue # Skip extend and position update

        start_positions.append(current_position)
        merged_data.extend(sublist)
        current_position += len(sublist)
        # print(f"  [merge_data_debug] After item {i}: len={len(sublist)}, current_pos={current_position}, start_positions={start_positions}") # Optional: uncomment for verbose debug

    # print(f"  [merge_data_debug] Final start_positions: {start_positions}") # Optional: uncomment for verbose debug
    # The function should return start_positions which has length = len(data)
    return merged_data, start_positions

def stack_and_pad_right(tensors):
    if not tensors:
        return torch.tensor([]), torch.tensor([])
    # Check if tensors is a list of tensors before proceeding
    if not isinstance(tensors, (list, tuple)) or not all(isinstance(t, torch.Tensor) for t in tensors):
         print(f"Error in stack_and_pad_right: Input is not a list/tuple of tensors. Type: {type(tensors)}")
         return torch.tensor([]), torch.tensor([])
    if not all(t.numel() > 0 for t in tensors): # Check for empty tensors within the list
         print(f"Warning in stack_and_pad_right: Input contains empty tensors.")
         # Filter out empty tensors? Or return error? For now, filter.
         tensors = [t for t in tensors if t.numel() > 0]
         if not tensors: return torch.tensor([]), torch.tensor([]) # Return empty if all were empty

    try:
        max_len = max(tensor.shape[0] for tensor in tensors)
    except ValueError: # Handles case where tensors might be empty after filtering
        print("Error in stack_and_pad_right: Cannot compute max_len, possibly due to empty tensors.")
        return torch.tensor([]), torch.tensor([])

    padded_tensors = []
    padding_masks = []
    for tensor in tensors:
        pad_len = max_len - tensor.shape[0]
        try:
            padded_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_len)) # Pad last dim (seq_len)
            padded_tensors.append(padded_tensor)
            padding_mask = torch.cat([torch.ones(tensor.shape[0], dtype=torch.long, device=tensor.device),
                                      torch.zeros(pad_len, dtype=torch.long, device=tensor.device)])
            padding_masks.append(padding_mask)
        except Exception as e:
            print(f"Error padding tensor in stack_and_pad_right: {e}, tensor shape: {tensor.shape}, max_len: {max_len}, pad_len: {pad_len}")
            # Skip this tensor or return error
            continue # Skip problematic tensor

    if not padded_tensors: # If all tensors failed padding
         return torch.tensor([]), torch.tensor([])

    try:
        stacked_tensor = torch.stack(padded_tensors)
        padding_masks = torch.stack(padding_masks)
    except Exception as e:
        print(f"Error during final stack in stack_and_pad_right: {e}")
        print(f"Number of tensors to stack: {len(padded_tensors)}")
        if padded_tensors: print(f"Shape of first tensor: {padded_tensors[0].shape}")
        return torch.tensor([]), torch.tensor([])

    return stacked_tensor, padding_masks

def stack_and_pad_left(tensors):
    if not tensors:
        return torch.tensor([]), torch.tensor([])
    # Add similar checks as stack_and_pad_right
    if not isinstance(tensors, (list, tuple)) or not all(isinstance(t, torch.Tensor) for t in tensors):
         print(f"Error in stack_and_pad_left: Input is not a list/tuple of tensors. Type: {type(tensors)}")
         return torch.tensor([]), torch.tensor([])
    if not all(t.numel() > 0 for t in tensors):
         print(f"Warning in stack_and_pad_left: Input contains empty tensors.")
         tensors = [t for t in tensors if t.numel() > 0]
         if not tensors: return torch.tensor([]), torch.tensor([])

    try:
        max_len = max(tensor.shape[0] for tensor in tensors)
    except ValueError:
        print("Error in stack_and_pad_left: Cannot compute max_len.")
        return torch.tensor([]), torch.tensor([])

    padded_tensors = []
    padding_masks = []
    for tensor in tensors:
        pad_len = max_len - tensor.shape[0]
        try:
            # Pad second to last dim (seq_len) -> (0, 0 for hidden_dim, pad_len, 0 for seq_dim)
            padded_tensor = torch.nn.functional.pad(tensor, (0, 0, pad_len, 0))
            padded_tensors.append(padded_tensor)
            padding_mask = torch.cat([torch.zeros(pad_len, dtype=torch.long, device=tensor.device),
                                     torch.ones(tensor.shape[0], dtype=torch.long, device=tensor.device)])
            padding_masks.append(padding_mask)
        except Exception as e:
            print(f"Error padding tensor in stack_and_pad_left: {e}, tensor shape: {tensor.shape}, max_len: {max_len}, pad_len: {pad_len}")
            continue

    if not padded_tensors:
         return torch.tensor([]), torch.tensor([])

    try:
        stacked_tensor = torch.stack(padded_tensors)
        padding_masks = torch.stack(padding_masks)
    except Exception as e:
        print(f"Error during final stack in stack_and_pad_left: {e}")
        print(f"Number of tensors to stack: {len(padded_tensors)}")
        if padded_tensors: print(f"Shape of first tensor: {padded_tensors[0].shape}")
        return torch.tensor([]), torch.tensor([])
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
        self.device = device

        # --- Llama Setup ---
        self.Llama_tokenizer = AutoTokenizer.from_pretrained(Llama_path, padding_side="right")
        if self.Llama_tokenizer.pad_token is None:
             print("Warning: Llama tokenizer does not have a pad token. Setting to EOS token.")
             self.Llama_tokenizer.pad_token = self.Llama_tokenizer.eos_token
        print(f"Llama tokenizer pad token ID: {self.Llama_tokenizer.pad_token_id}")

        print("Loading Llama model with 8-bit quantization...")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Using compute dtype: {compute_dtype}")

        if is_flash_attn_2_available():
            print("Flash Attention 2 available, enabling...")
            attn_impl = "flash_attention_2"
        else:
            print("Flash Attention 2 not available, using default SDPA.")
            attn_impl = "sdpa"

        self.Llama_model = AutoModelForCausalLM.from_pretrained(
            Llama_path,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            device_map='auto',
            torch_dtype=compute_dtype,
            attn_implementation=attn_impl
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
        try:
             projector_device = self.Llama_model.lm_head.weight.device
        except AttributeError:
             projector_device = self.Llama_model.device
             if str(projector_device) == 'cpu' or str(projector_device) == 'meta':
                  projector_device = device

        print(f"Initializing MLP Projector and Classifier on device: {projector_device}")
        bert_hidden_size = self.Bert_model.config.hidden_size
        llama_hidden_size = self.Llama_model.config.hidden_size
        projection_intermediate_dim = llama_hidden_size

        self.projector = nn.Sequential(
            nn.Linear(bert_hidden_size, projection_intermediate_dim),
            nn.GELU(),
            nn.Linear(projection_intermediate_dim, llama_hidden_size)
        ).to(projector_device).to(compute_dtype)
        print(f"MLP Projector created with intermediate dim {projection_intermediate_dim}, dtype {compute_dtype}")
        # --- End Projector Setup ---

        # --- CLASSIFICATION HEAD ---
        self.classifier = nn.Linear(llama_hidden_size, 2).to(projector_device).to(compute_dtype)
        print(f"Classification head created with output size 2, dtype {compute_dtype}")
        # --- END CLASSIFICATION HEAD ---

        # --- Instruction Tokens (Simpler prompt for classification) ---
        self.instruc_tokens = self.Llama_tokenizer(
            ['Below is a sequence of system log messages:'],
            return_tensors="pt", padding=True).to(projector_device)
        # --- End Instruction Tokens ---

        # --- PEFT Loading / Creation ---
        if ft_path is not None:
            print(f'Loading fine-tuned components from {ft_path}.')
            Llama_ft_path = os.path.join(ft_path, 'Llama_ft')
            Bert_ft_path = os.path.join(ft_path, 'Bert_ft')
            projector_path = os.path.join(ft_path, 'projector.pt')
            classifier_path = os.path.join(ft_path, 'classifier.pt')

            try:
                self.Llama_model = PeftModel.from_pretrained(
                    self.Llama_model, Llama_ft_path, is_trainable=is_train_mode)
                print("Llama PEFT adapters loaded.")
            except Exception as e: print(f"Could not load Llama PEFT adapters: {e}. Using base Llama.")

            if os.path.exists(Bert_ft_path) and isinstance(self.Bert_model, BertModel):
                 try:
                     self.Bert_model = PeftModel.from_pretrained(
                        self.Bert_model, Bert_ft_path, is_trainable=is_train_mode)
                     print("BERT PEFT adapters loaded.")
                 except Exception as e: print(f"Could not load BERT PEFT adapters: {e}. Using base BERT.")
            else: print(f"BERT PEFT model not found or base BERT already PEFT, using base BERT.")

            if os.path.exists(projector_path):
                try:
                    state_dict = torch.load(projector_path, map_location=projector_device)
                    self.projector.load_state_dict(state_dict)
                    print("Projector state dict loaded.")
                except Exception as e: print(f"Could not load projector state dict: {e}. Using initialized projector.")
            else: print(f"Projector state dict not found, using initialized projector.")

            if os.path.exists(classifier_path):
                try:
                    state_dict = torch.load(classifier_path, map_location=projector_device)
                    self.classifier.load_state_dict(state_dict)
                    print("Classifier state dict loaded.")
                except Exception as e: print(f"Could not load classifier state dict: {e}. Using initialized classifier.")
            else: print(f"Classifier state dict not found, using initialized classifier.")

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

    def save_ft_model(self, path):
        if not os.path.exists(path): os.makedirs(path)
        Llama_ft_path = os.path.join(path,'Llama_ft')
        Bert_ft_path = os.path.join(path,'Bert_ft')
        projector_path = os.path.join(path,'projector.pt')
        classifier_path = os.path.join(path, 'classifier.pt')

        print(f"Saving Llama PEFT model to {Llama_ft_path}")
        self.Llama_model.save_pretrained(Llama_ft_path)
        if isinstance(self.Bert_model, PeftModel):
             print(f"Saving Bert PEFT model to {Bert_ft_path}")
             self.Bert_model.save_pretrained(Bert_ft_path)
        else: print("BERT model is not a PEFT model, skipping save.")
        print(f"Saving projector state dict to {projector_path}")
        torch.save(self.projector.state_dict(), projector_path)
        print(f"Saving classifier state dict to {classifier_path}")
        torch.save(self.classifier.state_dict(), classifier_path)


    # --- Updated Training Control Methods ---
    def _set_trainable(self, bert_lora=False, projector=False, llama_lora=False, classifier=False):
        """Helper to set requires_grad for different components."""
        for name, param in self.Bert_model.named_parameters():
            param.requires_grad = bert_lora and 'lora_' in name
        for param in self.projector.parameters():
            param.requires_grad = projector
        for name, param in self.Llama_model.named_parameters():
            is_lora = 'lora_' in name
            param.requires_grad = llama_lora and is_lora
        for param in self.classifier.parameters():
            param.requires_grad = classifier

    def set_train_only_projector(self):
        print("Setting trainable: Projector ONLY")
        self._set_trainable(projector=True)

    def set_train_only_Llama(self):
        print("Setting trainable: Llama LoRA adapters ONLY")
        self._set_trainable(llama_lora=True)

    def set_train_projectorAndBert(self):
        print("Setting trainable: Projector and BERT LoRA adapters")
        self._set_trainable(bert_lora=True, projector=True)

    def set_train_only_classifier(self):
        print("Setting trainable: Classifier ONLY")
        self._set_trainable(classifier=True)

    def set_train_projector_and_classifier(self):
        print("Setting trainable: Projector and Classifier")
        self._set_trainable(projector=True, classifier=True)

    def set_finetuning_all(self):
        print("Setting trainable: Projector, Classifier, BERT LoRA, and Llama LoRA adapters")
        self._set_trainable(bert_lora=True, projector=True, llama_lora=True, classifier=True)
    # --- End Training Control Methods ---


    def get_cls_embeddings(self, sequences_):
        """
        Helper function to get BERT embeddings projected into Llama space.
        """
        input_seq_count = len(sequences_) # Expected output list length
        sequences = [sequence[:self.max_seq_len] for sequence in sequences_]

        if any(len(s) == 0 for s in sequences):
            print(f"WARNING in get_cls_embeddings: Input contains zero-length sequences. Lengths: {[len(s) for s in sequences]}")

        # Use the debug version of merge_data
        merged_log_data, start_positions = merge_data_debug(sequences)

        if not merged_log_data:
            print("WARNING in get_cls_embeddings: No data after merge_data.")
            return None, None

        # Check length of start_positions
        if len(start_positions) != input_seq_count:
             print(f"CRITICAL ERROR in get_cls_embeddings: merge_data returned {len(start_positions)} start positions, expected {input_seq_count}.")
             return None, None

        # BERT encoding using the merged data
        inputs = self.Bert_tokenizer(merged_log_data, return_tensors="pt", max_length=self.max_content_len, padding=True,
                                     truncation=True).to(self.device)
        try:
            bert_outputs = self.Bert_model(**inputs).pooler_output
        except Exception as e:
            print(f"Error during BERT forward pass: {e}")
            return None, None

        # Projector
        projector_input_device = next(self.projector.parameters()).device
        projector_dtype = next(self.projector.parameters()).dtype
        try:
            projected_outputs = self.projector(bert_outputs.to(device=projector_input_device, dtype=projector_dtype))
        except Exception as e:
            print(f"Error during Projector forward pass: {e}")
            return None, None

        # Determine Llama embedding layer's device and dtype
        try:
            llama_embed_device = self.Llama_model.get_input_embeddings().weight.device
            llama_embed_dtype = self.Llama_model.get_input_embeddings().weight.dtype
        except AttributeError:
             llama_embed_device = projector_input_device
             llama_embed_dtype = projector_dtype

        projected_outputs = projected_outputs.to(dtype=llama_embed_dtype, device=llama_embed_device)

        # Split back into sequences
        seq_embeddings_list = []
        if projected_outputs.shape[0] > 0:
            # Calculate the indices needed for splitting (all start positions EXCEPT the first one)
            split_indices = start_positions[1:]

            # --- Debug tensor_split inputs ---
            # print(f"Debug tensor_split: projected_outputs.shape={projected_outputs.shape}, split_indices={split_indices}, expected_splits={input_seq_count}")
            # --- End Debug ---

            if split_indices: # If there are split points (more than one sequence)
                 # Check if number of indices matches expectation
                 if len(split_indices) != input_seq_count - 1:
                      # *** This is where the previous error occurred ***
                      print(f"CRITICAL WARNING: Index count mismatch! Expected {input_seq_count - 1} indices in split_indices, but got {len(split_indices)}. Indices: {split_indices}")
                      print(f"  Originating from start_positions (len {len(start_positions)}): {start_positions}")
                      # Returning None as this indicates a bug in merge_data_debug
                      return None, None

                 try:
                     # Explicitly split along dimension 0
                     seq_embeddings_list = list(torch.tensor_split(projected_outputs, split_indices, dim=0))
                 except Exception as e:
                     print(f"Error during tensor_split: {e}")
                     print(f"  projected_outputs shape: {projected_outputs.shape}")
                     print(f"  split_indices: {split_indices}")
                     return None, None
            elif input_seq_count == 1: # Only one sequence in the batch
                 seq_embeddings_list = [projected_outputs]
            else: # input_seq_count > 1 but split_indices is empty - error in merge_data
                 print(f"ERROR: split_indices is empty but input_seq_count is {input_seq_count}.")
                 return None, None
        # else: seq_embeddings_list remains empty

        # Final Length Check
        if len(seq_embeddings_list) != input_seq_count:
             print(f"CRITICAL WARNING in get_cls_embeddings: Mismatch AFTER split! Input sequences count was {input_seq_count}, but got {len(seq_embeddings_list)} embedding sequences.")
             print(f"  Debug Info: projected_outputs.shape={projected_outputs.shape}, split_indices={split_indices}")
             return None, None # Return None to signal error

        return seq_embeddings_list, llama_embed_device


    def train_helper(self, sequences_, labels):
        '''
        Runs data through BERT -> Projector -> Llama -> Classifier.
        Returns classification logits and integer labels.
        '''
        batch_size = len(sequences_) # Original micro-batch size
        seq_embeddings_list, llama_embed_device = self.get_cls_embeddings(sequences_)

        if seq_embeddings_list is None:
             output_device = self.classifier.weight.device
             return torch.tensor([], device=output_device), torch.tensor([], device=output_device, dtype=torch.long)

        embed_layer = self.Llama_model.get_input_embeddings()
        instruc_tokens_dev = self.instruc_tokens['input_ids'].to(llama_embed_device)
        instruc_embeddings = embed_layer(instruc_tokens_dev)
        instruc_tokens_mask_dev = self.instruc_tokens['attention_mask'].to(llama_embed_device)
        ins1 = instruc_embeddings[0][instruc_tokens_mask_dev[0].bool()]

        embeddings = []
        valid_batch_indices = []
        # Iterate directly over the list returned by get_cls_embeddings
        for i, seq_embedding in enumerate(seq_embeddings_list):
            # Removed the problematic check: if i >= batch_size:
            if seq_embedding is not None and seq_embedding.shape[0] > 0:
                 full_prompt_embedding = torch.cat([ins1, seq_embedding], dim=0)
                 embeddings.append(full_prompt_embedding)
                 valid_batch_indices.append(i) # Index 'i' corresponds to position in seq_embeddings_list & original batch
            else:
                 print(f"Warning: Skipping sequence {i} due to empty embedding in train_helper.")

        if not embeddings:
             output_device = self.classifier.weight.device
             return torch.tensor([], device=output_device), torch.tensor([], device=output_device, dtype=torch.long)

        num_valid_seqs = len(embeddings)

        inputs_embeds, attention_mask = stack_and_pad_left(embeddings)
        if inputs_embeds.numel() == 0:
             print("Warning: inputs_embeds is empty after padding in train_helper.")
             output_device = self.classifier.weight.device
             return torch.tensor([], device=output_device), torch.tensor([], device=output_device, dtype=torch.long)

        model_device = self.Llama_model.device
        inputs_embeds = inputs_embeds.to(model_device)
        attention_mask = attention_mask.to(model_device) # Shape [num_valid_seqs, padded_len]

        outputs = self.Llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        try:
            last_hidden_state = outputs.last_hidden_state # Shape [num_valid_seqs, padded_len, hidden_size]
        except AttributeError:
            try:
                last_hidden_state = outputs.hidden_states[-1]
            except (AttributeError, TypeError, IndexError) as e:
                 print("ERROR: Could not find last_hidden_state in Llama model output.")
                 if hasattr(outputs, 'keys'): print(f"Output keys: {outputs.keys()}")
                 raise e

        actual_batch_size_llama = last_hidden_state.size(0)
        if actual_batch_size_llama != num_valid_seqs:
             print(f"WARNING: Batch size mismatch! Expected {num_valid_seqs} valid sequences, but Llama output has batch size {actual_batch_size_llama}.")
             num_valid_seqs = actual_batch_size_llama

        if attention_mask.size(0) != num_valid_seqs:
             print(f"ERROR: Attention mask batch size ({attention_mask.size(0)}) != Llama output batch size ({num_valid_seqs}).")
             output_device = self.classifier.weight.device
             return torch.tensor([], device=output_device), torch.tensor([], device=output_device, dtype=torch.long)

        sequence_lengths = attention_mask.sum(dim=1) - 1
        padded_len = last_hidden_state.shape[1]
        sequence_lengths = torch.clamp(sequence_lengths, max=padded_len - 1)
        if torch.any(sequence_lengths < 0):
             print(f"ERROR: Negative sequence lengths calculated. Attention mask sums: {attention_mask.sum(dim=1)}")
             output_device = self.classifier.weight.device
             return torch.tensor([], device=output_device), torch.tensor([], device=output_device, dtype=torch.long)

        batch_indices = torch.arange(num_valid_seqs, device=model_device)

        cls_input_hidden_state = last_hidden_state[batch_indices, sequence_lengths]

        classifier_device = self.classifier.weight.device
        cls_input_hidden_state = cls_input_hidden_state.to(classifier_device)

        logits = self.classifier(cls_input_hidden_state) # Shape [num_valid_seqs, 2]

        # Ensure valid_batch_indices are within bounds of original labels list
        if not all(idx < len(labels) for idx in valid_batch_indices):
             print(f"ERROR: Invalid index found in valid_batch_indices! Indices: {valid_batch_indices}, Original Labels Length: {len(labels)}")
             output_device = self.classifier.weight.device
             return torch.tensor([], device=output_device), torch.tensor([], device=output_device, dtype=torch.long)

        valid_str_labels = [labels[i] for i in valid_batch_indices]

        if len(valid_str_labels) != logits.shape[0]:
             print(f"ERROR: Mismatch between number of valid labels ({len(valid_str_labels)}) and number of logits ({logits.shape[0]}).")
             if len(valid_str_labels) > logits.shape[0]:
                 valid_str_labels = valid_str_labels[:logits.shape[0]]
                 print("Warning: Truncated labels to match logit count.")
             else:
                 output_device = self.classifier.weight.device
                 return torch.tensor([], device=output_device), torch.tensor([], device=output_device, dtype=torch.long)

        integer_labels = torch.tensor([1 if lbl == 'anomalous' else 0 for lbl in valid_str_labels],
                                      dtype=torch.long, device=logits.device)

        return logits, integer_labels


    def forward(self, sequences_):
        '''
        Inference function for classification. Returns classification logits.
        '''
        self.eval()
        batch_size = len(sequences_)

        with torch.no_grad():
            seq_embeddings_list, llama_embed_device = self.get_cls_embeddings(sequences_)

            if seq_embeddings_list is None:
                 output_device = self.classifier.weight.device
                 return torch.empty((batch_size, 2), device=output_device).fill_(-float('inf'))

            embed_layer = self.Llama_model.get_input_embeddings()
            instruc_tokens_dev = self.instruc_tokens['input_ids'].to(llama_embed_device)
            instruc_embeddings = embed_layer(instruc_tokens_dev)
            instruc_tokens_mask_dev = self.instruc_tokens['attention_mask'].to(llama_embed_device)
            ins1 = instruc_embeddings[0][instruc_tokens_mask_dev[0].bool()]

            embeddings = []
            original_indices = []
            for i, seq_embedding in enumerate(seq_embeddings_list):
                 if seq_embedding is not None and seq_embedding.shape[0] > 0:
                      full_prompt_embedding = torch.cat([ins1, seq_embedding], dim=0)
                      embeddings.append(full_prompt_embedding)
                      original_indices.append(i)

            if not embeddings:
                 output_device = self.classifier.weight.device
                 return torch.empty((batch_size, 2), device=output_device).fill_(-float('inf'))

            num_valid_seqs = len(embeddings)

            inputs_embeds, attention_mask = stack_and_pad_left(embeddings)
            if inputs_embeds.numel() == 0:
                 print("Warning: inputs_embeds is empty after padding in forward.")
                 output_device = self.classifier.weight.device
                 return torch.empty((batch_size, 2), device=output_device).fill_(-float('inf'))

            model_device = self.Llama_model.device
            inputs_embeds = inputs_embeds.to(model_device)
            attention_mask = attention_mask.to(model_device)

            outputs = self.Llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

            try:
                last_hidden_state = outputs.last_hidden_state
            except AttributeError:
                try:
                    last_hidden_state = outputs.hidden_states[-1]
                except (AttributeError, TypeError, IndexError) as e:
                     print("ERROR: Could not find last_hidden_state in Llama model output during inference.")
                     output_device = self.classifier.weight.device
                     return torch.full((batch_size, 2), -float('inf'), device=output_device)

            actual_batch_size_llama = last_hidden_state.size(0)
            if actual_batch_size_llama != num_valid_seqs:
                 print(f"WARNING: Inference batch size mismatch! Expected {num_valid_seqs}, Llama output {actual_batch_size_llama}.")
                 num_valid_seqs = actual_batch_size_llama

            if attention_mask.size(0) != num_valid_seqs:
                 print(f"ERROR: Inference Attention mask batch size ({attention_mask.size(0)}) != Llama output batch size ({num_valid_seqs}).")
                 output_device = self.classifier.weight.device
                 return torch.full((batch_size, 2), -float('inf'), device=output_device)

            sequence_lengths = attention_mask.sum(dim=1) - 1
            padded_len = last_hidden_state.shape[1]
            sequence_lengths = torch.clamp(sequence_lengths, max=padded_len - 1)
            if torch.any(sequence_lengths < 0):
                 print(f"ERROR: Negative sequence lengths in inference. Attention mask sums: {attention_mask.sum(dim=1)}")
                 output_device = self.classifier.weight.device
                 return torch.full((batch_size, 2), -float('inf'), device=output_device)

            batch_indices = torch.arange(num_valid_seqs, device=model_device)
            cls_input_hidden_state = last_hidden_state[batch_indices, sequence_lengths]

            classifier_device = self.classifier.weight.device
            cls_input_hidden_state = cls_input_hidden_state.to(classifier_device)

            logits = self.classifier(cls_input_hidden_state) # [num_valid_seqs, 2]

            full_batch_logits = torch.full((batch_size, 2), -float('inf'), device=logits.device, dtype=logits.dtype)
            if len(original_indices) == num_valid_seqs:
                 full_batch_logits[original_indices] = logits
            else:
                 print(f"Warning: Mismatch between original_indices ({len(original_indices)}) and num_valid_seqs ({num_valid_seqs}) during inference mapping.")
                 min_len = min(len(original_indices), num_valid_seqs)
                 if min_len > 0:
                      full_batch_logits[original_indices[:min_len]] = logits[:min_len]

            return full_batch_logits