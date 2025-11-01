import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import os
import gc
from typing import TYPE_CHECKING

from engine.data_utils import BalancedSampler, FocalLoss
from utils.helpers import format_time, get_eta

if TYPE_CHECKING:
    from engine.training_controller import TrainingController

def train_phase(controller: 'TrainingController', phase_name, num_epochs, learning_rate, train_dataset, validation_dataset, progress_state):
    phase_start_time = time.time()
    
    if num_epochs == 0:
        controller._log(f"Skipping {phase_name} phase (0 epochs).")
        return True, None, 0

    controller._log(f"\n>>>> STARTING {phase_name.upper()} PHASE <<<<")
    controller._log(f"Epochs: {num_epochs}, LR: {learning_rate}, Num Workers: {controller.num_workers}")
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, controller.model.parameters()), lr=learning_rate)
    criterion = FocalLoss().to(controller.device)
    
    sampler = BalancedSampler(train_dataset.tensors[1].cpu().numpy(), controller.hp['min_less_portion'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=controller.hp['micro_batch_size'],
        sampler=sampler,
        num_workers=controller.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    best_metric = -1
    patience_counter = 0
    best_model_path = None
    metric_key = controller.hp['early_stopping_metric']
    min_delta = controller.hp['early_stopping_min_delta']
    
    phase_total_steps = len(train_loader) * num_epochs
    phase_steps = 0
    
    if progress_state.get('phase_start_time') is None:
        progress_state['phase_start_time'] = time.time()

    for epoch in range(num_epochs):
        controller._log(f"\n--- Epoch {epoch + 1}/{num_epochs} ({phase_name}) ---")
        controller.model.train()
        
        epoch_loss = 0
        epoch_start_time = time.time()
        
        pbar_desc = f"Epoch {epoch + 1}/{num_epochs} ({phase_name})"
        is_gui_mode = controller.is_gui_mode
        pbar = tqdm(train_loader, desc=pbar_desc, disable=is_gui_mode)
        
        for i, (sequences, labels) in enumerate(pbar):
            sequences, labels = sequences.to(controller.device, non_blocking=True), labels.to(controller.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            logits, _ = controller.model.get_logits(sequences)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            epoch_loss += batch_loss
            controller.batch_losses.append(batch_loss)
            
            progress_state['global_step'] += 1
            phase_steps += 1
            
            if i % 20 == 0 or i == len(train_loader) - 1:
                metrics = {
                    'loss': batch_loss,
                    'lr': optimizer.param_groups[0]['lr']
                }
                
                eta_str = get_eta(progress_state['phase_start_time'], progress_state['global_step'], progress_state['total_steps'])
                
                if not pbar.disable:
                    pbar.set_postfix(metrics)
                
                if is_gui_mode:
                    controller.callback({
                        "progress": progress_state['global_step'] / progress_state['total_steps'] if progress_state['total_steps'] > 0 else 0,
                        "status": f"Epoch {epoch+1} ({phase_name}) - Batch {i+1}/{len(train_loader)} - ETA: {eta_str}",
                        "metrics": metrics
                    })

        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        controller._log(f"Epoch {epoch + 1} Complete. Avg Loss: {avg_epoch_loss:.4f}, Time: {format_time(epoch_time)}")

        if validation_dataset:
            controller._log("Running validation...")
            val_metrics, _ = evaluate(controller, validation_dataset, "validation", epoch)
            
            current_metric = val_metrics[f'val_{metric_key}']
            
            if current_metric > best_metric + min_delta:
                best_metric = current_metric
                patience_counter = 0
                controller._log(f"New best model! {metric_key.capitalize()}: {best_metric:.4f}. Saving model...")
                
                temp_model_dir = os.path.join(controller.execution_dir, f"temp_model_{phase_name}")
                controller.model.save_ft_model(temp_model_dir)
                best_model_path = temp_model_dir
            else:
                patience_counter += 1
                controller._log(f"No improvement. {metric_key.capitalize()}: {current_metric:.4f}. Patience: {patience_counter}/{controller.hp['early_stopping_patience']}")

            if patience_counter >= controller.hp['early_stopping_patience']:
                controller._log("Early stopping triggered.")
                break
    
    if not best_model_path and not validation_dataset:
        controller._log("No validation dataset. Saving final model for phase.")
        temp_model_dir = os.path.join(controller.execution_dir, f"temp_model_{phase_name}")
        controller.model.save_ft_model(temp_model_dir)
        best_model_path = temp_model_dir
    
    phase_duration = time.time() - phase_start_time
    return True, best_model_path, phase_duration


def evaluate(controller: 'TrainingController', dataset, dataset_name, epoch=-1):
    start_time = time.time()
    controller.model.eval()
    
    loader = DataLoader(
        dataset,
        batch_size=controller.hp['micro_batch_size'] * 2,
        num_workers=controller.num_workers,
        pin_memory=True
    )
    
    all_preds = []
    all_labels = []
    
    is_gui_mode = controller.is_gui_mode
    pbar_desc = f"Evaluating {dataset_name}"
    
    with torch.no_grad():
        for sequences, labels in tqdm(loader, desc=pbar_desc, disable=is_gui_mode):
            sequences = sequences.to(controller.device, non_blocking=True)
            
            logits, _ = controller.model.get_logits(sequences)
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    
    metrics = {
        f'{dataset_name}_accuracy': accuracy,
        f'{dataset_name}_precision': precision,
        f'{dataset_name}_recall': recall,
        f'{dataset_name}_f1_score': f1
    }
    
    epoch_str = f" (Epoch {epoch+1})" if epoch != -1 else " (Final)"
    controller._log(f"Validation Metrics{epoch_str}: " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
    
    if controller.callback and epoch != -1:
        controller.callback({"validation_metrics": metrics})
    
    duration = time.time() - start_time
    return metrics, duration

def evaluate_and_visualize(controller: 'TrainingController', dataset, dataset_name):
    controller._log(f"Running final evaluation on {dataset_name} dataset...")
    
    metrics, eval_time = evaluate(controller, dataset, dataset_name, epoch=999)
    
    metrics[f'{dataset_name}_inference_time_sec'] = eval_time
    metrics[f'{dataset_name}_samples_per_sec'] = len(dataset) / eval_time
    
    controller._log(f"Final {dataset_name.upper()} Metrics: ")
    for k, v in metrics.items():
        controller._log(f"  {k}: {v:.4f}")
        
    if controller.visualizer:
        try:
            loader = DataLoader(dataset, batch_size=controller.hp['micro_batch_size'], shuffle=False, num_workers=controller.num_workers, pin_memory=True)
            
            all_labels = []
            all_probs = []
            
            controller.model.eval()
            with torch.no_grad():
                for sequences, labels in tqdm(loader, desc=f"Generating plots for {dataset_name}", disable=True):
                    sequences = sequences.to(controller.device, non_blocking=True)
                    logits, _ = controller.model.get_logits(sequences)
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    all_probs.extend(probs)
                    all_labels.extend(labels.cpu().numpy())

            all_preds = (np.array(all_probs) > 0.5).astype(int)

            controller.visualizer.plot_confusion_matrix(all_labels, all_preds, f"{dataset_name}_confusion_matrix")
            controller.visualizer.plot_roc_curve(all_labels, all_probs, f"{dataset_name}_roc_curve")
            controller.visualizer.plot_precision_recall_curve(all_labels, all_probs, f"{dataset_name}_pr_curve")
            controller.visualizer.plot_distributions(all_probs, all_labels, f"{dataset_name}_distributions")
        except Exception as e:
            controller._log(f"Failed to generate plots for {dataset_name}: {e}")

    return {dataset_name: metrics}, eval_time