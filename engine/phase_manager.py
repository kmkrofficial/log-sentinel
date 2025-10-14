import torch
import time
import math
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader
import bitsandbytes as bnb
import os

from engine.data_utils import FocalLoss, BalancedSampler

def train_phase(controller, phase_name, n_epochs, lr, train_dataset, validation_dataset, progress_state):
    if not n_epochs > 0:
        return True, None

    controller._log(f"\n--- Starting Training Phase: {phase_name} (max {n_epochs} epochs) ---")
    
    patience = controller.hp.get("early_stopping_patience", 2)
    min_delta = controller.hp.get("early_stopping_min_delta", 0.0)
    patience_counter = 0
    best_score = -1.0
    temp_best_model_dir = controller.report_dir / f"phase_{phase_name}_model"

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    trainable_params = [p for p in controller.model.parameters() if p.requires_grad]
    if not trainable_params:
        controller._log(f"Phase '{phase_name}' has no trainable parameters, skipping.")
        return True, None
    
    optimizer = bnb.optim.PagedAdamW8bit(trainable_params, lr=lr)
    
    sampler = BalancedSampler(train_dataset.tensors[1].numpy(), controller.hp.get('min_less_portion', 0.5))
    train_loader = DataLoader(
        train_dataset,
        batch_size=controller.hp['micro_batch_size'],
        sampler=sampler,
        num_workers=min(os.cpu_count(), 16), # USE MULTIPLE CPU CORES
        pin_memory=True,
        persistent_workers=True # AVOID PER-EPOCH OVERHEAD
    )
    
    num_optimizer_steps = math.ceil(len(sampler) / controller.hp['batch_size']) * n_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=num_optimizer_steps, pct_start=0.1)
    
    grad_accum_steps = controller.hp['batch_size'] // controller.hp['micro_batch_size']
    
    progress_state['phase_steps'] = 0
    progress_state['phase_total_steps'] = len(train_loader) * n_epochs
    progress_state['phase_start_time'] = time.time()
    
    for epoch in range(int(n_epochs)):
        controller.model.train()
        controller._log(f"--- Epoch {epoch + 1}/{int(n_epochs)} ({phase_name}) ---")
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for batch_idx, (sequences, labels) in enumerate(pbar):
            progress_state['global_step'] += 1
            progress_state['phase_steps'] += 1

            sequences = sequences.to(controller.device)
            labels = labels.to(controller.device)
            
            logits, _ = controller.model.get_logits(sequence_tensor_batch=sequences)
            loss = criterion(logits, labels) / grad_accum_steps
            
            loss.backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            pbar.set_postfix(loss=loss.item() * grad_accum_steps)
            controller.batch_losses.append(loss.item() * grad_accum_steps)

            # Update progress for UI
            time_elapsed_total = time.time() - controller.run_start_time
            time_elapsed_phase = time.time() - progress_state['phase_start_time']
            
            progress_overall = progress_state['global_step'] / progress_state['total_steps'] if progress_state['total_steps'] > 0 else 0
            
            etc_overall = (time_elapsed_total / progress_overall) * (1 - progress_overall) if progress_overall > 0 else 0
            
            progress_phase = progress_state['phase_steps'] / progress_state['phase_total_steps'] if progress_state['phase_total_steps'] > 0 else 0
            etc_phase = (time_elapsed_phase / progress_phase) * (1 - progress_phase) if progress_phase > 0 else 0
            
            status = {
                "epoch": f"Epoch {epoch + 1}/{int(n_epochs)} ({phase_name})",
                "progress": progress_overall,
                "loss": loss.item() * grad_accum_steps,
                "time_elapsed": time_elapsed_total,
                "etc_overall": etc_overall,
                "etc_phase": etc_phase
            }
            if controller.callback(status) == 'STOP':
                controller._log("Stop request received.")
                pbar.close()
                return False, temp_best_model_dir

        if validation_dataset:
            val_metrics = evaluate_and_visualize(controller, validation_dataset, f"epoch_{epoch+1}_val")
            current_score = val_metrics[f"epoch_{epoch+1}_val"]['overall']['f1_score']
            controller._log(f"Epoch {epoch+1} Validation F1-Score: {current_score:.4f} (Best: {best_score:.4f})")
            
            if current_score - best_score > min_delta:
                best_score = current_score
                patience_counter = 0
                controller.model.save_ft_model(temp_best_model_dir)
                controller._log(f"New best score! Saving model state.")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    controller._log(f"EARLY STOPPING: Validation score has not improved for {patience} epochs.")
                    break
    
    return True, temp_best_model_dir

def evaluate_and_visualize(controller, dataset, dataset_name_prefix):
    controller._log(f"\n--- Starting Evaluation on {dataset_name_prefix.capitalize()} Set ---")
    controller.model.eval()
    
    all_preds, all_probas = [], []
    gt_labels = dataset.tensors[1].numpy()
    
    eval_loader = DataLoader(dataset, batch_size=controller.hp['batch_size'])

    with torch.no_grad():
        for sequences, _ in tqdm(eval_loader, desc=f"Evaluating {dataset_name_prefix.capitalize()} Set"):
            sequences = sequences.to(controller.device)
            logits, _ = controller.model.get_logits(sequence_tensor_batch=sequences)
            if logits is not None:
                probas = torch.softmax(logits, dim=-1)
                all_probas.extend(probas[:, 1].cpu().numpy())
                all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
    
    p, r, f1, _ = precision_recall_fscore_support(gt_labels, all_preds, average='binary', pos_label=1, zero_division=0)
    metrics = {"overall": {"accuracy": accuracy_score(gt_labels, all_preds), "precision": p, "recall": r, "f1_score": f1}}
    
    controller.visualizer.plot_confusion_matrix(confusion_matrix(gt_labels, all_preds), ['Normal', 'Anomalous'], filename_prefix=dataset_name_prefix)
    controller.visualizer.plot_roc_curve(gt_labels, np.array(all_probas), filename_prefix=dataset_name_prefix)
    controller.visualizer.plot_overall_metrics(metrics['overall'], filename_prefix=dataset_name_prefix)

    return {dataset_name_prefix: metrics}