import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
import random
# Import updated model (which now includes classifier)
from model import LogLLM
from customDataset import CustomDataset
from torch import optim
import gc
from transformers import get_linear_schedule_with_warmup

# --- Hyperparameters and Configuration ---
# Reduced Epochs for faster observation run with Classification Head
n_epochs_phase1 = 2    # Projector only
n_epochs_phase2 = 3    # Classifier only
n_epochs_phase3 = 3    # Projector + Classifier
n_epochs_phase4 = 4    # All trainable (Proj, Cls, Llama LoRA, Bert LoRA if enabled) -> Total 12 Epochs

dataset_name = 'BGL'
batch_size = 8
micro_batch_size = 4 # Increased based on VRAM availability
gradient_accumulation_steps = batch_size // micro_batch_size # 8 // 4 = 2

# Learning Rates for Classification
lr_phase1 = 1e-4     # Projector LR
lr_phase2 = 5e-4     # Classifier LR (can be higher initially)
lr_phase3 = 7e-5     # Projector + Classifier LR
lr_phase4 = 1e-5     # Final fine-tuning LR (low)

max_content_len = 100
max_seq_len = 128

# Paths
data_path = r'E:\research-stuff\LogLLM-3b\dataset\train.csv'
Bert_path = r"E:\research-stuff\LogLLM-3b\models\bert-base-uncased"
Llama_path = r"E:\research-stuff\LogLLM-3b\models\Llama-3.2-3B"

ROOT_DIR = Path(__file__).resolve().parent
# Use new path for classification model checkpoints
ft_path = os.path.join(ROOT_DIR, r"ft_model_cls_{}".format(dataset_name))

# Oversampling
min_less_portion = 0.5 # Keep 50/50 target

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("--- Training Configuration (Classification Head - Reduced Epochs) ---")
print(f'Phase 1 Epochs (Projector): {n_epochs_phase1}, LR: {lr_phase1}')
print(f'Phase 2 Epochs (Classifier): {n_epochs_phase2}, LR: {lr_phase2}')
print(f'Phase 3 Epochs (Proj+Cls): {n_epochs_phase3}, LR: {lr_phase3}')
print(f'Phase 4 Epochs (All): {n_epochs_phase4}, LR: {lr_phase4}')
print(f'Total Epochs: {n_epochs_phase1 + n_epochs_phase2 + n_epochs_phase3 + n_epochs_phase4}')
print(f'dataset_name: {dataset_name}')
print(f'batch_size: {batch_size}')
print(f'micro_batch_size: {micro_batch_size}')
print(f'gradient_accumulation_steps: {gradient_accumulation_steps}')
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


def trainModel(model, dataset, micro_batch_size, gradient_accumulation_steps, n_epochs, lr, phase_name, num_samples=None):
    """Trains the model for a given classification phase."""
    print(f"\n--- Starting Training Phase: {phase_name} ---")
    model.train()
    # Use standard CrossEntropyLoss for classification (expects logits)
    criterion = nn.CrossEntropyLoss()

    print("\n--- Preparing Optimizer and Scheduler ---")
    # Filter parameters that require gradients for the optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
         print("Warning: No trainable parameters found for this training phase. Skipping.")
         return
    print_number_of_trainable_model_parameters(model) # Print stats

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.98), eps=1e-9)
    print(f"Optimizer: AdamW with lr={lr}")

    # --- Data Oversampling ---
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
            if num_majority == 0 and min_less_portion < 1.0: target_less_num = num_less
            elif (1 - min_less_portion) == 0: target_less_num = int(1e9)
            else: target_less_num = int((min_less_portion * num_majority) / (1 - min_less_portion))
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
    total_optimization_steps = (total_batches_per_epoch * int(n_epochs) + gradient_accumulation_steps -1) // gradient_accumulation_steps
    num_warmup_steps = int(total_optimization_steps * 0.05) # 5% warmup
    print(f"Total optimization steps for this phase: {total_optimization_steps}")
    print(f"Warmup steps for this phase: {num_warmup_steps}")
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_optimization_steps
    )
    print(f"Scheduler: Linear warmup ({num_warmup_steps} steps) and decay")
    print("--------------------------------------")

    global_step = 0
    for epoch in range(int(n_epochs)):
        print(f"\n--- Epoch {epoch + 1}/{int(n_epochs)} ({phase_name}) ---")
        model.train()
        epoch_total_loss = 0.0
        epoch_correct_preds = 0
        epoch_total_samples = 0

        random.shuffle(indexes)
        epoch_indexes = indexes[:effective_len]
        optimizer.zero_grad()
        pbar = tqdm(range(0, len(epoch_indexes), micro_batch_size), desc=f'Epoch {epoch + 1}', leave=False)

        interval_loss, interval_samples, interval_correct = 0.0, 0, 0
        log_interval = 50 # Log more frequently with potentially faster steps

        for i_th, start_idx in enumerate(pbar):
            end_idx = min(start_idx + micro_batch_size, len(epoch_indexes))
            this_batch_indexes = epoch_indexes[start_idx:end_idx]
            if not this_batch_indexes: continue

            this_batch_seqs, this_batch_str_labels = dataset.get_batch(this_batch_indexes)

            # --- Batch Composition Logging ---
            log_freq = gradient_accumulation_steps * 25
            if global_step == 0 or (global_step > 0 and global_step % log_freq == 0):
                try:
                    if isinstance(this_batch_str_labels[0], str): num_anomalous = sum(1 for lbl in this_batch_str_labels if lbl == 'anomalous')
                    else: num_anomalous = sum(1 for lbl in this_batch_str_labels if lbl == 1)
                    num_normal = len(this_batch_str_labels) - num_anomalous
                    # print(f"\n  [Debug Step {global_step}] Micro-batch: Normal={num_normal}, Anomalous={num_anomalous}") # Optional: reduce verbosity
                except Exception as log_e: print(f"\n [Debug Step {global_step}] Error logging batch composition: {log_e}")
            # --- End Batch Composition Logging ---

            try:
                logits, integer_labels = model.train_helper(this_batch_seqs, this_batch_str_labels)
                if logits.shape[0] == 0: continue
                loss = criterion(logits, integer_labels)
                loss_val = loss.item()
                loss = loss / gradient_accumulation_steps
            except Exception as e:
                 print(f"\nError FWD/Loss step {i_th} (Global {global_step}): {e}")
                 continue

            try:
                 # Scale gradients for mixed precision if using amp
                 loss.backward()
            except Exception as e:
                 print(f"\nError BWD step {i_th} (Global {global_step}): {e}")
                 optimizer.zero_grad()
                 continue

            # Accumulate stats
            batch_actual_size = logits.shape[0]
            epoch_total_loss += loss_val * batch_actual_size
            epoch_total_samples += batch_actual_size
            interval_loss += loss_val * batch_actual_size
            interval_samples += batch_actual_size

            # --- Accuracy Calculation ---
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                correct_preds = (preds == integer_labels).sum().item()
                epoch_correct_preds += correct_preds
                interval_correct += correct_preds
            # --- End Accuracy ---

            # --- Optimizer Step ---
            if (i_th + 1) % gradient_accumulation_steps == 0:
                # Clip gradients only for trainable parameters
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=f"{loss_val:.4f}", acc=f"{(correct_preds/batch_actual_size):.3f}", lr=f"{current_lr:.2e}")

                # --- Periodic Logging ---
                if global_step > 0 and global_step % log_interval == 0:
                    avg_loss_interval = interval_loss / interval_samples if interval_samples > 0 else 0.0
                    avg_acc_interval = interval_correct / interval_samples if interval_samples > 0 else 0.0
                    print(f"\n  [Step {global_step}] Avg Loss (last {log_interval} steps): {avg_loss_interval:.4f}, Acc: {avg_acc_interval:.4f}, LR: {current_lr:.2e}")
                    interval_loss, interval_samples, interval_correct = 0.0, 0, 0

        # --- End of Epoch ---
        epoch_avg_loss = epoch_total_loss / epoch_total_samples if epoch_total_samples > 0 else 0.0
        epoch_avg_acc = epoch_correct_preds / epoch_total_samples if epoch_total_samples > 0 else 0.0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n[Epoch {epoch + 1} Summary] Average Loss: {epoch_avg_loss:.4f}, Accuracy: {epoch_avg_acc:.4f}, End LR: {current_lr:.2e}")

        torch.cuda.empty_cache(); gc.collect()

    print(f"\n--- Finished Training Phase: {phase_name} ---")


if __name__ == '__main__':
    print("--- Initializing Dataset and Model ---")
    if not os.path.exists(data_path): raise FileNotFoundError(f"Training data not found: {data_path}")
    if not os.path.exists(Bert_path): raise FileNotFoundError(f"BERT model not found: {Bert_path}")
    if not os.path.exists(Llama_path): raise FileNotFoundError(f"Llama model not found: {Llama_path}")

    dataset = CustomDataset(data_path)
    print(f"Dataset loaded: {len(dataset)} sequences.")

    load_existing_ft = False # Start fresh for classification approach
    effective_ft_path = None
    print("Initializing new model for classification task.")

    model = LogLLM(Bert_path, Llama_path, ft_path=effective_ft_path, is_train_mode=True, device=device,
                   max_content_len=max_content_len, max_seq_len=max_seq_len)
    print("Model initialized.")

    # --- Training Phases for Classification (Reduced Epochs) ---

    # Phase 1: Train Projector Only
    if n_epochs_phase1 > 0:
        model.set_train_only_projector()
        trainModel(model, dataset, micro_batch_size, gradient_accumulation_steps, n_epochs_phase1, lr_phase1, "Projector Only")
        torch.cuda.empty_cache(); gc.collect()
    else: print("\nSkipping Phase 1: Projector Training")

    # Phase 2: Train Classifier Only
    if n_epochs_phase2 > 0:
        model.set_train_only_classifier()
        trainModel(model, dataset, micro_batch_size, gradient_accumulation_steps, n_epochs_phase2, lr_phase2, "Classifier Only")
        torch.cuda.empty_cache(); gc.collect()
    else: print("\nSkipping Phase 2: Classifier Training")

    # Phase 3: Train Projector and Classifier Together
    if n_epochs_phase3 > 0:
        model.set_train_projector_and_classifier()
        trainModel(model, dataset, micro_batch_size, gradient_accumulation_steps, n_epochs_phase3, lr_phase3, "Projector + Classifier")
        torch.cuda.empty_cache(); gc.collect()
    else: print("\nSkipping Phase 3: Projector + Classifier Training")

    # Phase 4: Fine-tune All Trainable Components
    if n_epochs_phase4 > 0:
        model.set_finetuning_all()
        trainModel(model, dataset, micro_batch_size, gradient_accumulation_steps, n_epochs_phase4, lr_phase4, "All Trainable")
        torch.cuda.empty_cache(); gc.collect()
    else: print("\nSkipping Phase 4: Full Fine-tuning")

    # --- Save Final Model ---
    print("\n--- Saving Final Classification Model ---")
    # Ensure ft_path points to the classification model directory
    if not os.path.exists(ft_path):
        os.makedirs(ft_path)
        print(f"Created directory: {ft_path}")
    model.save_ft_model(ft_path)
    print(f"Final model saved to: {ft_path}")
    print("--- Training Script Finished ---")