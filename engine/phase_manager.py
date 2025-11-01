import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import os
import gc

from engine.data_utils import BalancedSampler, FocalLoss
from utils.helpers import format_time, get_eta

def train_phase(controller, phase_name, num_epochs, learning_rate, train_dataset, validation_dataset, progress_state):
    if num_epochs == 0:
        controller._log(f"Skipping {phase_name} phase (0 epochs).")
        return True, None

    controller._log(f"\n>>>> STARTING {phase_name.upper()} PHASE <<<<")
    controller._log(f"Epochs: {num_epochs}, LR: {learning_rate}, Num Workers: {controller.num_workers}")
    
    optimizer = torch.optim.AdamW(controller.model.parameters(), lr=learning_rate)
    criterion = FocalLoss()
    
    sampler = BalancedSampler(train_dataset.tensors[1].numpy(), controller.hp['min_less_portion'])
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
    
    progress_state['phase_total_steps'] = len(train_loader) * num_epochs
    progress_state['phase_steps'] = 0

    for epoch in range(num_epochs):
        controller._log(f"\n--- Epoch {epoch + 1}/{num_epochs} ({phase_name}) ---")
        controller.model.train()
        
        epoch_loss = 0
        epoch_start_time = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", disable=controller.callback != TrainingController.__init__.__defaults__[0])
        
        for i, (sequences, labels) in enumerate(pbar):
            progress_state['phase_start_time'] = progress_state.get('phase_start_time', time.time())
            
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
            progress_state['phase_steps'] += 1
            
            if i % 20 == 0:
                metrics = {
                    'loss': batch_loss,
                    'lr': optimizer.param_groups[0]['lr']
                }
                eta_str = get_eta(progress_state['phase_start_time'], progress_state['phase_steps'], progress_state['phase_total_steps'])
                if not pbar.disable:
                    pbar.set_postfix(metrics)
                    controller.callback({
                        "progress": progress_state['global_step'] / progress_state['total_steps'],
                        "status": f"Epoch {epoch+1} ({phase_name}) - Batch {i+1}/{len(train_loader)} - ETA: {eta_str}",
                        "metrics": metrics
                    })

        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        controller._log(f"Epoch {epoch + 1} Complete. Avg Loss: {avg_epoch_loss:.4f}, Time: {format_time(epoch_time)}")

        if validation_dataset:
            controller._log("Running validation...")
            val_metrics = evaluate(controller, validation_dataset, "validation", epoch)
            
            current_metric = val_metrics[f'val_{metric_key}']
            
            if current_metric > best_metric + min_delta:
                best_metric = current_metric
                patience_counter = 0
                controller._log(f"New best model! {metric_key.capitalize()}: {best_metric:.4f}. Saving model...")
                
                temp_model_dir = os.path.join(controller.report_dir, f"temp_model_{phase_name}")
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
        temp_model_dir = os.path.join(controller.report_dir, f"temp_model_{phase_name}")
        controller.model.save_ft_model(temp_model_dir)
        best_model_path = temp_model_dir

    return True, best_model_path


def evaluate(controller, dataset, dataset_name, epoch=-1):
    controller.model.eval()
    
    loader = DataLoader(
        dataset,
        batch_size=controller.hp['micro_batch_size'] * 2,
        num_workers=controller.num_workers,
        pin_memory=True
    )
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in tqdm(loader, desc=f"Evaluating {dataset_name}", disable=controller.callback != TrainingController.__init__.__defaults__[0]):
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
    
    controller._log(f"Validation Metrics (Epoch {epoch+1}): " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
    
    if controller.callback:
        controller.callback({"validation_metrics": metrics})
        
    return metrics

def evaluate_and_visualize(controller, dataset, dataset_name):
    controller._log(f"Running final evaluation on {dataset_name} dataset...")
    
    start_time = time.time()
    metrics = evaluate(controller, dataset, dataset_name, epoch=999)
    eval_time = time.time() - start_time
    
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

    return {dataset_name: metrics}