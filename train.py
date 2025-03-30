import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
import random
from model import LogLLM # Assuming model.py is in the same directory or accessible
from customDataset import CustomDataset # Assuming customDataset.py is accessible
from torch import optim
import gc # Garbage collector
from transformers import get_linear_schedule_with_warmup # Import scheduler


# --- Hyperparameters and Configuration ---
# Aggressive settings + MLP Projector focus
n_epochs_1 = 2      # Llama LoRA only
n_epochs_2_1 = 5    # Projector only (Keep increased focus)
n_epochs_2_2 = 3    # Projector + BERT LoRA
n_epochs_3 = 25     # All trainable (Keep significantly Increased)
dataset_name = 'BGL'
batch_size = 8
micro_batch_size = 2
gradient_accumulation_steps = batch_size // micro_batch_size

# Learning Rates (Adjusted for MLP Projector focus)
lr_1 = 5e-4         # Initial Llama LoRA tuning
lr_2_1 = 1e-4         # Projector tuning (Keep higher LR for projector phase)
lr_2_2 = 3e-5         # Projector + BERT LoRA tuning
lr_3 = 1e-5         # Full model tuning (Keep very low)

max_content_len = 100
max_seq_len = 128

# Paths (Using user's paths)
data_path = r'E:\research-stuff\LogLLM-3b\dataset\train.csv'
Bert_path = r"E:\research-stuff\LogLLM-3b\models\bert-base-uncased"
Llama_path = r"E:\research-stuff\LogLLM-3b\models\Llama-3.2-3B"

ROOT_DIR = Path(__file__).resolve().parent
ft_path = os.path.join(ROOT_DIR, r"ft_model_{}".format(dataset_name))

# Skewed dataset handling (Force 50/50)
min_less_portion = 0.5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("--- Training Configuration (MLP Projector Attempt) ---")
print(f'n_epochs_1 (Llama LoRA): {n_epochs_1}')
print(f'n_epochs_2_1 (Projector): {n_epochs_2_1}')
print(f'n_epochs_2_2 (Proj + BERT LoRA): {n_epochs_2_2}')
print(f'n_epochs_3 (All Trainable): {n_epochs_3}')
print(f'dataset_name: {dataset_name}')
print(f'batch_size: {batch_size}')
print(f'micro_batch_size: {micro_batch_size}')
print(f'gradient_accumulation_steps: {gradient_accumulation_steps}')
print(f'lr_1: {lr_1}')
print(f'lr_2_1: {lr_2_1}')
print(f'lr_2_2: {lr_2_2}')
print(f'lr_3: {lr_3}')
print(f'max_content_len: {max_content_len}')
print(f'max_seq_len: {max_seq_len}')
print(f'min_less_portion for oversampling: {min_less_portion}')
print(f'Using device: {device}')
print(f'Llama model path: {Llama_path}')
print(f'BERT model path: {Bert_path}')
print(f'Training data path: {data_path}')
print(f'Fine-tuning save path: {ft_path}')
print("-----------------------------")

def print_number_of_trainable_model_parameters(model):
    # (Function remains the same)
    trainable_model_params = 0
    all_model_params = 0
    trainable_param_set = set()
    for name, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
            trainable_param_set.add(param)
    print(f"Total model parameters: {all_model_params:,}")
    print(f"Trainable model parameters: {trainable_model_params:,}")
    print(f"Percentage trainable: {100 * trainable_model_params / all_model_params:.4f}%")
    return trainable_param_set


def trainModel(model, dataset, micro_batch_size, gradient_accumulation_steps, n_epochs, lr, num_samples=None):
    """Trains the model for a given phase."""
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='mean')

    print("\n--- Preparing Optimizer and Scheduler ---")
    trainable_model_params = print_number_of_trainable_model_parameters(model)
    if not trainable_model_params:
         print("Warning: No trainable parameters found for this training phase. Skipping.")
         return

    optimizer = torch.optim.AdamW(trainable_model_params, lr=lr, betas=(0.9, 0.98), eps=1e-9)
    print(f"Optimizer: AdamW with lr={lr}")

    # --- Token analysis ---
    # (Remains the same)
    try:
        target_phrase_normal = "The sequence is normal."
        target_phrase_anomalous = "The sequence is anomalous."
        normal_target_tokens = model.Llama_tokenizer(target_phrase_normal, add_special_tokens=True)['input_ids'][1:]
        anomalous_target_tokens = model.Llama_tokenizer(target_phrase_anomalous, add_special_tokens=True)['input_ids'][1:]
        special_normal_tokens = set(normal_target_tokens) - set(anomalous_target_tokens)
        special_anomalous_tokens = set(anomalous_target_tokens) - set(normal_target_tokens)
        print(f"Target 'normal' tokens (IDs): {normal_target_tokens}")
        print(f"Target 'anomalous' tokens (IDs): {anomalous_target_tokens}")
        print(f"Unique 'normal' target tokens: {special_normal_tokens}")
        print(f"Unique 'anomalous' target tokens: {special_anomalous_tokens}")
        acc_calc_tokens = special_normal_tokens.union(special_anomalous_tokens)
        if not acc_calc_tokens: print("Warning: No unique target tokens found.")
    except Exception as e:
        print(f"Error getting unique tokens: {e}.")
        acc_calc_tokens = set()
    # --- End Token analysis ---

    # --- Data Oversampling ---
    # (Remains the same, using min_less_portion = 0.5)
    indexes = list(range(len(dataset)))
    original_size = len(indexes)
    try:
        num_less = dataset.num_less
        num_majority = dataset.num_majority
        less_indexes = dataset.less_indexes
        if original_size == 0: less_portion = 0
        else: less_portion = num_less / original_size
        print(f"Dataset size: {original_size}, Minority class samples: {num_less} ({less_portion:.2%})")

        if num_less > 0 and less_portion < min_less_portion:
            print(f"Minority class proportion ({less_portion:.2%}) is less than target ({min_less_portion:.2%}). Oversampling...")
            if num_majority == 0 and min_less_portion < 1.0:
                 target_less_num = num_less
            elif (1 - min_less_portion) == 0:
                 target_less_num = int(1e9)
            else:
                 target_less_num = int((min_less_portion * num_majority) / (1 - min_less_portion))
            add_num = max(0, target_less_num - num_less)
            if add_num > 0:
                print(f"Adding {add_num} samples from the minority class (Target proportion: {min_less_portion:.1%}).")
                oversampled_indices = np.random.choice(less_indexes, add_num, replace=True).tolist()
                indexes.extend(oversampled_indices)
                print(f"Dataset size after oversampling: {len(indexes)}")
            else: print("Calculated number of samples to add is zero or negative.")
        else: print("Minority class proportion meets the target or no minority samples exist.")
    except AttributeError as e:
        print(f"Warning: Dataset object missing attributes for oversampling: {e}. Skipping.")
    # --- End Data Oversampling ---

    # --- Training Loop Setup ---
    if num_samples is not None:
        num_samples = min(num_samples, len(indexes))
        print(f"Training on a subset of {num_samples} samples per epoch.")
        effective_len = num_samples
    else:
        effective_len = len(indexes)

    # --- Configure Scheduler ---
    total_batches_per_epoch = (effective_len + micro_batch_size - 1) // micro_batch_size
    total_optimization_steps = (total_batches_per_epoch * n_epochs + gradient_accumulation_steps -1) // gradient_accumulation_steps
    num_warmup_steps = int(total_optimization_steps * 0.05) # 5% warmup
    print(f"Total optimization steps: {total_optimization_steps}")
    print(f"Warmup steps: {num_warmup_steps}")
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_optimization_steps
    )
    print(f"Scheduler: Linear warmup ({num_warmup_steps} steps) and decay")
    print("--------------------------------------")

    global_step = 0
    for epoch in range(int(n_epochs)):
        print(f"\n--- Epoch {epoch + 1}/{int(n_epochs)} ---")
        model.train()
        epoch_total_loss, epoch_processed_tokens = 0.0, 0
        epoch_total_acc, epoch_total_acc_count = 0, 0

        random.shuffle(indexes)
        epoch_indexes = indexes[:effective_len]
        optimizer.zero_grad()
        pbar = tqdm(range(0, len(epoch_indexes), micro_batch_size), desc=f'Epoch {epoch + 1}', leave=False)

        interval_loss, interval_tokens, interval_acc, interval_acc_count = 0.0, 0, 0, 0
        log_interval = 200 # Log every 200 optimization steps

        for i_th, start_idx in enumerate(pbar):
            end_idx = min(start_idx + micro_batch_size, len(epoch_indexes))
            this_batch_indexes = epoch_indexes[start_idx:end_idx]
            if not this_batch_indexes: continue

            this_batch_seqs, this_batch_labels = dataset.get_batch(this_batch_indexes)

            # --- Batch Composition Logging ---
            log_freq = gradient_accumulation_steps * 50 # Increase logging frequency
            if global_step == 0 or (global_step > 0 and global_step % log_freq == 0):
                try:
                    if isinstance(this_batch_labels[0], str): num_anomalous = sum(1 for lbl in this_batch_labels if lbl == 'anomalous')
                    else: num_anomalous = sum(1 for lbl in this_batch_labels if lbl == 1)
                    num_normal = len(this_batch_labels) - num_anomalous
                    print(f"\n  [Debug Step {global_step}] Micro-batch: Normal={num_normal}, Anomalous={num_anomalous}")
                except Exception as log_e: print(f"\n [Debug Step {global_step}] Error logging batch composition: {log_e}")
            # --- End Batch Composition Logging ---

            try:
                outputs, targets = model.train_helper(this_batch_seqs, this_batch_labels)
                if outputs.shape[0] == 0 or targets.shape[0] == 0: continue
                loss = criterion(outputs, targets)
                loss_val = loss.item()
                loss = loss / gradient_accumulation_steps
            except Exception as e:
                 print(f"\nError FWD/Loss step {i_th} (Global {global_step}): {e}")
                 continue

            try:
                 loss.backward()
            except Exception as e:
                 print(f"\nError BWD step {i_th} (Global {global_step}): {e}")
                 optimizer.zero_grad()
                 continue

            # Accumulate stats
            epoch_total_loss += loss_val * targets.size(0)
            epoch_processed_tokens += targets.size(0)
            interval_loss += loss_val * targets.size(0)
            interval_tokens += targets.size(0)

            # --- Accuracy Calculation ---
            if acc_calc_tokens and targets.numel() > 0:
                with torch.no_grad():
                    acc_mask = torch.zeros_like(targets, dtype=torch.bool, device=targets.device)
                    for token_id in acc_calc_tokens: acc_mask |= (targets == token_id)
                    num_relevant = acc_mask.sum().item()
                    if num_relevant > 0:
                        preds = outputs.argmax(dim=-1)[acc_mask].to(targets.device)
                        correct = (preds == targets[acc_mask]).sum().item()
                        epoch_total_acc += correct
                        epoch_total_acc_count += num_relevant
                        interval_acc += correct
                        interval_acc_count += num_relevant
            # --- End Accuracy ---

            # --- Optimizer Step ---
            if (i_th + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient Clipping
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{current_lr:.2e}")

                # --- Periodic Logging ---
                if global_step > 0 and global_step % log_interval == 0:
                    avg_loss_interval = interval_loss / interval_tokens if interval_tokens > 0 else 0.0
                    avg_acc_interval = interval_acc / interval_acc_count if interval_acc_count > 0 else 0.0
                    print(f"\n  [Step {global_step}] Avg Loss (last {log_interval} steps): {avg_loss_interval:.4f}, Acc (unique tokens): {avg_acc_interval:.4f}, LR: {current_lr:.2e}")
                    interval_loss, interval_tokens, interval_acc, interval_acc_count = 0.0, 0, 0, 0

        # --- End of Epoch ---
        epoch_avg_loss = epoch_total_loss / epoch_processed_tokens if epoch_processed_tokens > 0 else 0.0
        epoch_avg_acc = epoch_total_acc / epoch_total_acc_count if epoch_total_acc_count > 0 else 0.0
        current_lr = optimizer.param_groups[0]['lr'] # Get LR at end of epoch
        print(f"\n[Epoch {epoch + 1} Summary] Average Loss: {epoch_avg_loss:.4f}, Accuracy (on unique tokens): {epoch_avg_acc:.4f}, End LR: {current_lr:.2e}")

        torch.cuda.empty_cache(); gc.collect()

    print("\n--- Training Phase Complete ---")


if __name__ == '__main__':
    print("--- Initializing Dataset and Model ---")
    if not os.path.exists(data_path): raise FileNotFoundError(f"Training data not found: {data_path}")
    if not os.path.exists(Bert_path): raise FileNotFoundError(f"BERT model not found: {Bert_path}")
    if not os.path.exists(Llama_path): raise FileNotFoundError(f"Llama model not found: {Llama_path}")

    dataset = CustomDataset(data_path)
    print(f"Dataset loaded: {len(dataset)} sequences.")

    load_existing_ft = False
    effective_ft_path = None
    llama_adapter_config = os.path.join(ft_path, 'Llama_ft', 'adapter_config.json')
    projector_file = os.path.join(ft_path, 'projector.pt')
    if load_existing_ft and os.path.exists(llama_adapter_config) and os.path.exists(projector_file):
        print(f"Found existing fine-tuned model files at {ft_path}. Loading...")
        effective_ft_path = ft_path
    else:
        if load_existing_ft: print(f"Existing fine-tuned model not found at {ft_path}. Initializing new model.")
        else: print("load_existing_ft is False. Initializing new model.")

    model = LogLLM(Bert_path, Llama_path, ft_path=effective_ft_path, is_train_mode=True, device=device,
                   max_content_len=max_content_len, max_seq_len=max_seq_len)
    print("Model initialized.")

    # --- Training Phases ---
    if n_epochs_1 > 0:
        print("\n" + "*" * 10 + f" Phase 1: Training Llama LoRA ({n_epochs_1} epochs, LR={lr_1}) " + "*" * 10)
        model.set_train_only_Llama()
        trainModel(model, dataset, micro_batch_size, gradient_accumulation_steps, n_epochs_1, lr_1)
        torch.cuda.empty_cache(); gc.collect()
    else: print("\nSkipping Phase 1")

    if n_epochs_2_1 > 0:
        print("\n" + "*" * 10 + f" Phase 2.1: Training Projector ({n_epochs_2_1} epochs, LR={lr_2_1}) " + "*" * 10)
        model.set_train_only_projector()
        trainModel(model, dataset, micro_batch_size, gradient_accumulation_steps, n_epochs_2_1, lr_2_1)
        torch.cuda.empty_cache(); gc.collect()
    else: print("\nSkipping Phase 2.1")

    if n_epochs_2_2 > 0:
        print("\n" + "*" * 10 + f" Phase 2.2: Training Projector & BERT LoRA ({n_epochs_2_2} epochs, LR={lr_2_2}) " + "*" * 10)
        model.set_train_projectorAndBert()
        trainModel(model, dataset, micro_batch_size, gradient_accumulation_steps, n_epochs_2_2, lr_2_2)
        torch.cuda.empty_cache(); gc.collect()
    else: print("\nSkipping Phase 2.2")

    if n_epochs_3 > 0:
        print("\n" + "*" * 10 + f" Phase 3: Training All Trainable ({n_epochs_3} epochs, LR={lr_3}) " + "*" * 10)
        model.set_finetuning_all()
        trainModel(model, dataset, micro_batch_size, gradient_accumulation_steps, n_epochs_3, lr_3)
        torch.cuda.empty_cache(); gc.collect()
    else: print("\nSkipping Phase 3")

    # --- Save Final Model ---
    print("\n--- Saving Final Fine-tuned Model ---")
    model.save_ft_model(ft_path)
    print(f"Final model saved to: {ft_path}")
    print("--- Training Script Finished ---")