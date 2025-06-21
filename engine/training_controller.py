import os
import gc
import random
import numpy as np
import torch
import time
import math
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import bitsandbytes as bnb
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModel, AutoConfig
import tempfile
import shutil
import pandas as pd
import traceback

from config import REPORTS_DIR, DEFAULT_BERT_PATH, TEMP_MODELS_DIR
from utils.database_manager import DatabaseManager
from utils.data_loader import LogDataset
from utils.resource_monitor import ResourceMonitor
from utils.log_visualizer import LogVisualizer
from logsentinel_model import LogSentinelModel
from utils.embedding_cacher import EmbeddingCacher
from utils.helpers import merge_data

class PrecomputedDataset:
    def __init__(self, embeddings, labels, class_counts=None, minority_label=None, less_indexes=None, majority_indexes=None):
        self.embeddings = embeddings
        self.labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
        self.class_counts = class_counts
        self.minority_label = minority_label
        self.less_indexes = less_indexes
        self.majority_indexes = majority_indexes

    def __len__(self):
        return len(self.labels)
    def get_batch(self, indexes):
        this_batch_embeds = [self.embeddings[i] for i in indexes]; temp_numeric_labels = self.labels[indexes]
        this_batch_labels = ['anomalous' if lbl == 1 else 'normal' for lbl in temp_numeric_labels]
        return this_batch_embeds, this_batch_labels
    def get_all_labels(self):
        return self.labels

class TrainingController:
    def __init__(self, model_name, dataset_name, hyperparameters, db_manager, callback=None, nickname=None, use_cached_embeddings=True, is_test_run=False, test_run_percentage=0.3):
        self.model_name, self.dataset_name = model_name, dataset_name
        self.nickname = nickname or f"{model_name.split('/')[-1]}-{dataset_name}"
        self.hp = hyperparameters; self.db = db_manager; self.callback = callback or (lambda *args: 'CONTINUE')
        self.run_id, self.model = None, None; self.visualizer = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.global_step_count, self.total_training_steps, self.run_start_time = 0, 0, 0
        self.batch_losses = []; self.use_cached_embeddings = use_cached_embeddings
        self.is_test_run = is_test_run
        self.test_run_percentage = test_run_percentage
        self.test_run_cache_files_to_clean = []

        if self.is_test_run:
            self.use_cached_embeddings = False
            self._log(f"--- QUICK TEST RUN MODE ACTIVATED ({self.test_run_percentage*100:.0f}% data): Embedding cache is disabled. ---")

    def _log(self, message):
        print(message); self.callback({"log": message})

    def _initialize_run(self):
        run_nickname = f"[TEST] {self.nickname}" if self.is_test_run else self.nickname
        
        if self.is_test_run:
            self.hp['is_test_run'] = True
            self.hp['test_run_percentage'] = self.test_run_percentage

        self.run_id = self.db.create_new_run('Training', self.model_name, self.dataset_name, self.hp, run_nickname)
        if self.run_id:
            self._log(f"Created new training run with ID: {self.run_id} (Nickname: {run_nickname})")
            self.report_dir = REPORTS_DIR / str(self.run_id); self.report_dir.mkdir(exist_ok=True)
            self.visualizer = LogVisualizer(plot_dir=self.report_dir)
        return self.run_id is not None

    def _cleanup(self, model_to_clean=None):
        target = model_to_clean if model_to_clean else self.model
        if target: self._log(f"Cleaning up model: {type(target).__name__}..."); del target
        gc.collect();
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        self._log("Cleanup complete.")

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _prepare_data(self, dataset_type, encoder_model_for_cache):
        dataset_path = os.path.join("datasets", self.dataset_name, f'{dataset_type}.csv')
        if not os.path.exists(dataset_path):
            if dataset_type == 'validation':
                self._log(f"Warning: '{dataset_type}.csv' not found. Skipping validation and early stopping.")
                return None
            else:
                raise FileNotFoundError(f"Required dataset file not found: {dataset_path}")

        df = pd.read_csv(dataset_path)
        if self.is_test_run:
            subset_size = int(self.test_run_percentage * len(df))
            self._log(f"QUICK TEST RUN: Using first {subset_size} of {len(df)} rows for '{dataset_type}.csv'.")
            df = df.head(subset_size)
        
        full_dataset = LogDataset(dataframe=df)

        cacher = EmbeddingCacher(encoder_name=encoder_model_for_cache, dataset_path=dataset_path)
        if self.is_test_run:
            test_cache_path = cacher.cache_file_path.with_suffix(f'.{self.test_run_percentage:.2f}_test_subset.pt')
            cacher.cache_file_path = test_cache_path
            self.test_run_cache_files_to_clean.append(test_cache_path)

        force_regenerate = not self.use_cached_embeddings
        if force_regenerate: self._log(f"Cache is disabled for this run. Forcing regeneration of {dataset_type} embeddings.")
        
        embeddings, labels = cacher.load_embeddings()
        
        if embeddings is not None and labels is not None and not force_regenerate:
            self._log(f"Loaded {len(embeddings)} cached embeddings for {dataset_type} set.")
        else:
            self._log(f"Generating new embeddings for {dataset_type} set...")
            encoder_path = DEFAULT_BERT_PATH.parent / DEFAULT_BERT_PATH.name
            encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_path)
            encoder_model = AutoModel.from_pretrained(encoder_path).to(self.device).eval()

            all_logs_flat, start_positions = merge_data(full_dataset.sequences)
            all_line_embeddings = []
            with torch.no_grad():
                total_lines = len(all_logs_flat); physical_batch_size = 256
                for i in tqdm(range(0, total_lines, physical_batch_size), desc=f"Generating {dataset_type} Embeddings"):
                    self.callback({"epoch": f"Embedding ({dataset_type})", "progress": (i + 1) / total_lines})
                    batch_logs = all_logs_flat[i:i+physical_batch_size]
                    inputs = encoder_tokenizer(batch_logs, return_tensors="pt", padding=True, truncation=True, max_length=self.hp['max_content_len']).to(self.device)
                    model_output = encoder_model(**inputs)
                    if 'sentence-transformer' in encoder_model.config._name_or_path:
                        line_embeddings = self._mean_pooling(model_output, inputs['attention_mask'])
                    else:
                        line_embeddings = model_output.last_hidden_state[:, 0, :]
                    all_line_embeddings.append(line_embeddings.cpu())
            all_line_embeddings_tensor = torch.cat(all_line_embeddings, dim=0)
            embeddings = list(torch.tensor_split(all_line_embeddings_tensor, start_positions[1:]))
            labels = full_dataset.get_all_labels()
            
            if dataset_type in ['train', 'validation'] and self.use_cached_embeddings:
                cacher.save_embeddings(embeddings, labels)

            self._cleanup(model_to_clean=encoder_model)
            del encoder_tokenizer
        
        return PrecomputedDataset(embeddings, labels, full_dataset.class_counts, full_dataset.minority_label, full_dataset.less_indexes, full_dataset.majority_indexes)

    def _evaluate_and_visualize(self, dataset, dataset_name_prefix):
        if not dataset:
            self._log(f"Skipping evaluation for '{dataset_name_prefix}' as dataset is not available.")
            return {}

        self._log(f"\n--- Starting Evaluation on {dataset_name_prefix.capitalize()} Set ---")
        self.model.eval()
        
        all_preds, all_probas = [], []
        gt_labels = dataset.get_all_labels()

        with torch.no_grad():
            for i in tqdm(range(0, len(dataset), self.hp['batch_size']), desc=f"Evaluating {dataset_name_prefix.capitalize()} Set"):
                end_idx = min(i + self.hp['batch_size'], len(dataset))
                embeds, _ = dataset.get_batch(list(range(i, end_idx)))
                logits, _ = self.model.get_logits(precomputed_embeddings=embeds)
                if logits is not None:
                    probas = torch.softmax(logits, dim=-1)
                    all_probas.extend(probas[:, 1].cpu().numpy())
                    all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        
        if len(all_preds) != len(gt_labels):
            gt_labels = gt_labels[:len(all_preds)]

        p, r, f1, _ = precision_recall_fscore_support(gt_labels, all_preds, average='binary', pos_label=1, zero_division=0)
        metrics = {"overall": {"accuracy": accuracy_score(gt_labels, all_preds), "precision": p, "recall": r, "f1_score": f1}}
        
        self.visualizer.plot_confusion_matrix(confusion_matrix(gt_labels, all_preds), ['Normal', 'Anomalous'], filename_prefix=dataset_name_prefix)
        self.visualizer.plot_roc_curve(gt_labels, np.array(all_probas), filename_prefix=dataset_name_prefix)
        self.visualizer.plot_overall_metrics(metrics['overall'], filename_prefix=dataset_name_prefix)

        return {dataset_name_prefix: metrics}
        
    def _get_validation_f1_score(self, validation_dataset):
        self.model.eval()
        all_preds, gt_labels = [], validation_dataset.get_all_labels()
        with torch.no_grad():
            for i in range(0, len(validation_dataset), self.hp['batch_size']):
                end_idx = min(i + self.hp['batch_size'], len(validation_dataset))
                embeds, _ = validation_dataset.get_batch(list(range(i, end_idx)))
                logits, _ = self.model.get_logits(precomputed_embeddings=embeds)
                if logits is not None:
                    all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        
        if len(all_preds) != len(gt_labels): gt_labels = gt_labels[:len(all_preds)]
        
        _, _, f1, _ = precision_recall_fscore_support(gt_labels, all_preds, average='binary', pos_label=1, zero_division=0)
        return f1


    def _train_phase(self, phase_name, n_epochs, lr, train_dataset, validation_dataset):
        if not n_epochs > 0: return True
        self._log(f"\n--- Starting Training Phase: {phase_name} (max {n_epochs} epochs) ---")
        
        patience = self.hp.get("early_stopping_patience", 2)
        min_delta = self.hp.get("early_stopping_min_delta", 0.0)
        patience_counter = 0
        best_score = -1.0
        temp_best_model_dir = self.report_dir / f"phase_{phase_name}_model"

        phase_start_time = time.time()
        steps_in_phase_count = 0
        num_majority = len(train_dataset.majority_indexes); num_less = len(train_dataset.less_indexes)
        oversampled_less_count = int(num_majority * self.hp.get('min_less_portion', 0.5))
        effective_epoch_size = num_majority + max(num_less, oversampled_less_count)
        micro_batches_per_epoch = math.ceil(effective_epoch_size / self.hp['micro_batch_size'])
        total_steps_this_phase = micro_batches_per_epoch * n_epochs

        num_normal = train_dataset.class_counts.get(0, 0); num_anomalous = train_dataset.class_counts.get(1, 0)
        total_samples = num_normal + num_anomalous
        if num_normal > 0 and num_anomalous > 0:
            weight_normal = total_samples / (2.0 * num_normal); weight_anomalous = total_samples / (2.0 * num_anomalous)
            class_weights = torch.tensor([weight_normal, weight_anomalous], dtype=torch.float32).to(self.device)
            self._log(f"Using weighted loss. Weights - Normal: {weight_normal:.2f}, Anomalous: {weight_anomalous:.2f}")
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params: self._log(f"Phase '{phase_name}' has no trainable parameters, skipping."); return True
        optimizer = bnb.optim.PagedAdamW8bit(trainable_params, lr=lr)
        
        num_optimizer_steps = math.ceil(effective_epoch_size / self.hp['batch_size']) * n_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(num_optimizer_steps * 0.1), num_training_steps=num_optimizer_steps)
        grad_accum_steps = self.hp['batch_size'] // self.hp['micro_batch_size']
        
        for epoch in range(int(n_epochs)):
            self.model.train()
            self._log(f"--- Epoch {epoch + 1}/{int(n_epochs)} ({phase_name}) ---")
            
            if num_less > 0 and num_majority > 0:
                num_to_add = max(0, oversampled_less_count - num_less)
                oversampled_less_indices = random.choices(train_dataset.less_indexes, k=num_to_add)
                indexes = np.concatenate([
                    np.array(train_dataset.majority_indexes, dtype=np.int64), 
                    np.array(train_dataset.less_indexes, dtype=np.int64), 
                    np.array(oversampled_less_indices, dtype=np.int64)
                ]).tolist()
            else:
                indexes = list(range(len(train_dataset)))
            random.shuffle(indexes)
            
            pbar = tqdm(total=len(indexes), desc=f"Epoch {epoch+1} Training")
            data_iterator = range(0, len(indexes), self.hp['micro_batch_size'])
            for batch_idx, start_idx in enumerate(data_iterator):
                self.global_step_count += 1
                steps_in_phase_count += 1
                
                try:
                    end_idx = min(start_idx + self.hp['micro_batch_size'], len(indexes))
                    batch_indexes = indexes[start_idx:end_idx]
                    
                    embeds, labels = train_dataset.get_batch(batch_indexes)

                    with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=(self.device.type == 'cuda')):
                        logits, int_labels = self.model.train_helper(labels=labels, precomputed_embeddings=embeds)
                        if logits.shape[0] == 0:
                            pbar.update(len(batch_indexes))
                            continue
                        loss = criterion(logits, int_labels) / grad_accum_steps
                    
                    loss.backward()
                    
                    if (batch_idx + 1) % grad_accum_steps == 0 or (start_idx + self.hp['micro_batch_size']) >= len(indexes):
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                
                except Exception as e:
                    self._log(f"FATAL: An error occurred during training step {self.global_step_count}.")
                    self._log(traceback.format_exc())
                    pbar.close()
                    raise e

                pbar.update(len(batch_indexes))
                current_loss = loss.item() * grad_accum_steps
                self.batch_losses.append(current_loss)
                
                progress_overall = self.global_step_count / self.total_training_steps if self.total_training_steps > 0 else 0
                elapsed_time_this_phase = time.time() - phase_start_time
                time_per_step = elapsed_time_this_phase / steps_in_phase_count if steps_in_phase_count > 0 else 0
                etc_phase = (total_steps_this_phase - steps_in_phase_count) * time_per_step if time_per_step > 0 else 0
                etc_overall = (self.total_training_steps - self.global_step_count) * time_per_step if time_per_step > 0 else 0

                status = {"epoch": f"Epoch {epoch + 1}/{int(n_epochs)} ({phase_name})", "progress": progress_overall, "loss": current_loss, "etc_phase": etc_phase, "etc_overall": etc_overall}
                if self.callback(status) == 'STOP': self._log("Stop request received."); pbar.close(); return False
            pbar.close()

            if validation_dataset:
                current_score = self._get_validation_f1_score(validation_dataset)
                metric_name = "F1-Score"
                self._log(f"Epoch {epoch+1} Validation {metric_name}: {current_score:.4f} (Best: {best_score:.4f})")
                
                if current_score - best_score > min_delta:
                    best_score = current_score
                    patience_counter = 0
                    self.model.save_ft_model(temp_best_model_dir)
                    self._log(f"New best score! Saving model state to '{temp_best_model_dir}' and resetting patience.")
                else:
                    patience_counter += 1
                    self._log(f"No significant improvement. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    self._log(f"EARLY STOPPING: Validation score has not improved significantly for {patience} epochs. Stopping phase '{phase_name}'.")
                    break
        
        if best_score == -1 and n_epochs > 0:
            self._log(f"No improvement found in phase '{phase_name}'. Saving final model state.")
            self.model.save_ft_model(temp_best_model_dir)

        return True

    def run(self):
        monitor = ResourceMonitor(); monitor.start(); final_status = 'FAILED'
        try:
            if not self._initialize_run(): raise RuntimeError("Failed to create a new run record.")
            self.run_start_time = time.time()
            
            encoder_path = DEFAULT_BERT_PATH.parent / DEFAULT_BERT_PATH.name
            encoder_config = AutoConfig.from_pretrained(encoder_path)
            encoder_hidden_size = encoder_config.hidden_size
            self.hp['encoder_hidden_size'] = encoder_hidden_size

            self._log("Preparing training data...")
            train_dataset = self._prepare_data('train', encoder_path.name)
            self._log("Preparing validation data...")
            validation_dataset = self._prepare_data('validation', encoder_path.name)
            self._log("Preparing testing data...")
            test_dataset = self._prepare_data('test', encoder_path.name)

            if not train_dataset: raise RuntimeError("Training dataset could not be loaded.")

            num_majority = len(train_dataset.majority_indexes); num_less = len(train_dataset.less_indexes)
            oversampled_less_count = int(num_majority * self.hp.get('min_less_portion', 0.5))
            effective_epoch_size = num_majority + max(num_less, oversampled_less_count)
            micro_batches_per_epoch = math.ceil(effective_epoch_size / self.hp['micro_batch_size'])
            self.total_training_steps = sum([self.hp.get(f'n_epochs_phase{i+1}', 0) * micro_batches_per_epoch for i in range(4)])
            self._log(f"Max total training micro-batch steps calculated: {self.total_training_steps}")
            
            ft_path = None
            training_phases = [
                ("Projector", 'set_train_only_projector', self.hp['n_epochs_phase1'], self.hp['lr_phase1']),
                ("Classifier", 'set_train_only_classifier', self.hp['n_epochs_phase2'], self.hp['lr_phase2']),
                ("Projector+Classifier", 'set_train_projector_and_classifier', self.hp['n_epochs_phase3'], self.hp['lr_phase3']),
                ("Fine-tuning All", 'set_finetuning_all', self.hp['n_epochs_phase4'], self.hp['lr_phase4'])
            ]
            all_phases_completed = True
            for name, setup_func_name, epochs, lr in training_phases:
                self._log(f"\n>>>> CONFIGURING MODEL FOR PHASE: {name} <<<<")
                self.model = LogSentinelModel(self.model_name, encoder_hidden_size, ft_path, True, self.device)
                getattr(self.model, setup_func_name)()
                
                if not self._train_phase(name, epochs, lr, train_dataset, validation_dataset):
                    all_phases_completed = False; final_status = 'ABORTED'; break
                
                ft_path = self.report_dir / f"phase_{name}_model"
                self._cleanup(self.model)

            if all_phases_completed:
                self._log("\n>>>> CONFIGURING MODEL FOR FINAL EVALUATION <<<<")
                self.model = LogSentinelModel(self.model_name, encoder_hidden_size, ft_path, False, self.device)
                
                final_metrics = {}
                validation_metrics = self._evaluate_and_visualize(validation_dataset, "validation")
                test_metrics = self._evaluate_and_visualize(test_dataset, "test")
                final_metrics.update(validation_metrics)
                final_metrics.update(test_metrics)
                
                final_metrics['overall'] = { "total_run_time_sec": time.time() - self.run_start_time }
                
                self.visualizer.plot_training_loss(self.batch_losses)
                self.db.save_performance_metrics(self.run_id, final_metrics)

                if ft_path and os.path.exists(ft_path):
                    final_model_path = self.report_dir / 'final_model'
                    if final_model_path.exists(): shutil.rmtree(final_model_path)
                    shutil.copytree(ft_path, final_model_path)
                    self._log(f"Consolidated best model saved to: {final_model_path}")
                
                final_status = 'COMPLETED'
        except Exception as e:
            tb_str = traceback.format_exc()
            error_msg = f"CRITICAL ERROR in run {self.run_id}: {e}\n{tb_str}"
            self._log(error_msg)
            self.callback({"error": f"{e}\n{tb_str}"})
            final_status = 'FAILED'
        finally:
            resource_metrics = monitor.stop()
            if self.run_id:
                self.db.save_resource_metrics(self.run_id, resource_metrics)
                if self.visualizer: self.visualizer.plot_resource_usage(resource_metrics)
                self.db.update_run_status(self.run_id, final_status, str(self.report_dir) if final_status == 'COMPLETED' else None)
            if self.model: self._cleanup(self.model)

            if self.is_test_run:
                self._log("Cleaning up temporary subset cache files...")
                for f_path in self.test_run_cache_files_to_clean:
                    if f_path.exists():
                        try: os.remove(f_path); self._log(f"  - Deleted: {f_path}")
                        except OSError as e: self._log(f"  - Error deleting {f_path}: {e}")
                
            self.callback({"status": final_status, "done": True})