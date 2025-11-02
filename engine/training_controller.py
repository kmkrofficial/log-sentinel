import os
import gc
import torch
import time
import shutil
import pandas as pd
import traceback
import platform
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.utils.data import TensorDataset, DataLoader, random_split
from datetime import datetime

from config import (
    EXECUTIONS_DIR, MODELS_DIR, DEFAULT_ENCODER_MODEL,
    DEFAULT_LLAMA_MODEL, get_hyperparameters
)
from utils.database_manager import DatabaseManager
from utils.data_loader import LogDataset
from utils.resource_monitor import ResourceMonitor
from utils.log_visualizer import LogVisualizer
from logsentinel_model import LogSentinelModel
from utils.helpers import merge_data, get_eta, format_time
from engine.phase_manager import train_phase, evaluate_and_visualize
from engine.data_utils import BalancedSampler

torch.backends.cuda.matmul.allow_tf32 = True

class TrainingController:
    def __init__(self, dataset_name, db_manager, callback=None, is_test_run=False, test_run_percentage=0.3):
        self.dataset_name = dataset_name
        self.hp = get_hyperparameters(dataset_name)

        self.llama_model_path = str(MODELS_DIR / DEFAULT_LLAMA_MODEL.split('/')[-1])
        self.encoder_path_str = str(MODELS_DIR / DEFAULT_ENCODER_MODEL.split('/')[-1])

        self.db = db_manager
        self.callback = callback or (lambda *args: 'CONTINUE')
        self.run_id = None
        self.model = None
        self.visualizer = None
        self.execution_dir = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.run_start_time = 0
        self.is_test_run = is_test_run
        self.test_run_percentage = test_run_percentage
        self.batch_losses = []
        self.num_workers = self.hp.get('dataloader_num_workers', 0)
        self.is_gui_mode = callback is not None

        if self.is_test_run:
            self._log(f"--- QUICK TEST RUN MODE ACTIVATED ({self.test_run_percentage*100:.0f}% data) ---")

    def _log(self, message):
        print(message)
        if self.callback: self.callback({"log": message})

    def _generate_nickname(self):
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        base_nickname = f"{self.dataset_name}_{now}"
        if self.is_test_run:
            base_nickname += f"_{int(self.test_run_percentage * 100)}pct_TEST"
        return base_nickname

    def _initialize_run(self):
        nickname = self._generate_nickname()

        self.run_id = self.db.create_new_run('Training', DEFAULT_LLAMA_MODEL, self.dataset_name, self.hp, nickname)
        if self.run_id:
            self._log(f"Created new training run with ID: {self.run_id} (Nickname: {nickname})")
            self.execution_dir = EXECUTIONS_DIR / str(nickname)
            viz_dir = self.execution_dir / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            self.visualizer = LogVisualizer(plot_dir=viz_dir)
        return self.run_id is not None

    def _cleanup(self, model_to_clean=None):
        target = model_to_clean if model_to_clean else self.model
        if target:
            del target
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _load_and_tensorize_dataset(self, dataset_path, encoder_model, encoder_tokenizer, progress_state):
        self._log(f"Loading and embedding {dataset_path.name}...")

        all_embeddings = []
        all_labels = []

        try:
            total_chunks = sum(1 for _ in pd.read_csv(dataset_path, chunksize=100000))
        except Exception as e:
            self._log(f"Could not read dataset {dataset_path}: {e}")
            return None

        with pd.read_csv(dataset_path, chunksize=100000) as reader:
            pbar = tqdm(reader, desc=f"Embedding {dataset_path.name}", disable=self.is_gui_mode, unit=" chunks", total=total_chunks)
            for i, chunk_df in enumerate(pbar):
                if self.is_test_run:
                    chunk_df = chunk_df.sample(frac=self.test_run_percentage, random_state=42)

                source_dataset = LogDataset(dataframe=chunk_df)
                sequences_in_chunk = source_dataset.sequences
                labels_in_chunk = source_dataset.get_all_labels()
                all_logs_flat, start_positions = merge_data(sequences_in_chunk)

                if not all_logs_flat:
                    all_embeddings.extend([torch.empty(0)] * len(labels_in_chunk))
                    all_labels.extend(labels_in_chunk)
                    continue

                log_message_batch_size = 512
                all_line_embeddings = []
                num_log_batches = (len(all_logs_flat) + log_message_batch_size - 1) // log_message_batch_size

                for k, batch_start in enumerate(range(0, len(all_logs_flat), log_message_batch_size)):
                    if self.is_gui_mode and k % 20 == 0:
                        status_msg = f"Embedding {dataset_path.name} [Chunk {i+1}/{total_chunks}]: Processing log batch {k+1}/{num_log_batches}"
                        self.callback({"status": status_msg})

                    log_batch = all_logs_flat[batch_start : batch_start + log_message_batch_size]
                    with torch.no_grad():
                        inputs = encoder_tokenizer(log_batch, return_tensors="pt", padding=True, truncation=True, max_length=self.hp['max_content_len']).to(self.device)
                        model_output = encoder_model(**inputs)
                        line_embeddings = self._mean_pooling(model_output, inputs['attention_mask'])
                        all_line_embeddings.append(line_embeddings.cpu())

                if not all_line_embeddings:
                    all_embeddings.extend([torch.empty(0)] * len(labels_in_chunk))
                    all_labels.extend(labels_in_chunk)
                    continue

                all_line_embeddings_tensor = torch.cat(all_line_embeddings, dim=0)
                sequence_tensors = list(torch.tensor_split(all_line_embeddings_tensor, start_positions[1:-1]))
                all_embeddings.extend(sequence_tensors)
                all_labels.extend(labels_in_chunk)

        self._log("Embedding complete. Tensorizing sequences...")
        
        tensorized_sequences, tensorized_labels = [], []
        max_seq_len = self.hp['max_seq_len']
        
        if not all_embeddings:
             raise RuntimeError(f"No log sequences found after processing {dataset_path.name}.")
        
        first_valid_tensor = next((t for t in all_embeddings if t.shape[0] > 0), None)
        if first_valid_tensor is None:
            raise RuntimeError(f"All log sequences are empty in {dataset_path.name}.")
        sample_embedding_dim = first_valid_tensor.shape[1]

        num_sequences = len(all_embeddings)
        for i in range(num_sequences):
            if self.is_gui_mode and i % 5000 == 0:
                status_msg = f"Tensorizing {dataset_path.name}: Processing sequence {i}/{num_sequences}"
                self.callback({"status": status_msg})
            
            seq_embeddings, seq_label = all_embeddings[i], all_labels[i]
            
            if seq_embeddings is None or seq_embeddings.shape[0] == 0:
                padding = torch.zeros(max_seq_len, sample_embedding_dim, dtype=torch.float32)
                tensorized_sequences.append(padding)
                tensorized_labels.append(torch.tensor(seq_label, dtype=torch.long))
                continue
                
            seq_len = seq_embeddings.shape[0]
            if seq_len > max_seq_len:
                tensorized_seq = seq_embeddings[-max_seq_len:]
            elif seq_len < max_seq_len:
                padding = torch.zeros(max_seq_len - seq_len, sample_embedding_dim, dtype=torch.float32)
                tensorized_seq = torch.cat([seq_embeddings, padding], dim=0)
            else:
                tensorized_seq = seq_embeddings
                
            tensorized_sequences.append(tensorized_seq)
            tensorized_labels.append(torch.tensor(seq_label, dtype=torch.long))

        if not tensorized_sequences:
            raise RuntimeError(f"No sequences were tensorized for {dataset_path.name}.")

        self._log(f"Created TensorDataset for {dataset_path.name} with {len(tensorized_labels)} samples.")
        return TensorDataset(torch.stack(tensorized_sequences), torch.stack(tensorized_labels))

    def run(self):
        monitor = ResourceMonitor()
        monitor.start()
        final_status = 'FAILED'
        total_training_time, total_testing_time = 0, 0
        final_model_metrics = {}

        try:
            if not self._initialize_run():
                raise RuntimeError("Failed to create a new run record.")
            self.run_start_time = time.time()

            self._log(f"Using Encoder: {self.encoder_path_str}")
            self._log(f"Using LLM: {self.llama_model_path}")

            encoder_config = AutoConfig.from_pretrained(self.encoder_path_str)
            self.hp['encoder_hidden_size'] = encoder_config.hidden_size

            self._log("Loading embedding model...")
            encoder_tokenizer = AutoTokenizer.from_pretrained(self.encoder_path_str)
            encoder_model = AutoModel.from_pretrained(self.encoder_path_str).to(self.device).eval()

            self._log("Preparing datasets...")
            train_dataset_path = Path("datasets") / self.dataset_name / 'train.csv'
            test_dataset_path = Path("datasets") / self.dataset_name / 'test.csv'

            full_train_dataset = self._load_and_tensorize_dataset(train_dataset_path, encoder_model, encoder_tokenizer, {})
            test_dataset = self._load_and_tensorize_dataset(test_dataset_path, encoder_model, encoder_tokenizer, {}) if test_dataset_path.exists() else None

            self._log("Creating a 90/10 train/validation split from the training data.")
            train_size = int(0.9 * len(full_train_dataset))
            validation_size = len(full_train_dataset) - train_size
            train_dataset, validation_dataset = random_split(full_train_dataset, [train_size, validation_size])
            self._log(f"New training set size: {len(train_dataset)}")
            self._log(f"New validation set size: {len(validation_dataset)}")
            
            self._cleanup(model_to_clean=encoder_model)
            del encoder_tokenizer

            ft_path = None
            progress_state = {'global_step': 0, 'phase_start_time': 0}
            
            # Calculate total training steps for the progress bar
            train_steps = 0
            for phase_epochs in [self.hp.get('n_epochs_phase_adapters', 0), self.hp.get('n_epochs_phase_full', 0)]:
                if phase_epochs > 0:
                    sampler = BalancedSampler(train_dataset.dataset.tensors[1][train_dataset.indices].cpu().numpy(), self.hp.get('min_less_portion', 0.5))
                    loader = DataLoader(train_dataset, batch_size=self.hp['micro_batch_size'], sampler=sampler, num_workers=self.num_workers)
                    train_steps += len(loader) * phase_epochs
            progress_state['total_steps'] = train_steps

            self.model = LogSentinelModel(self.llama_model_path, self.hp['encoder_hidden_size'], self.hp, ft_path, True, self.device, self._log)
            self.model = torch.compile(self.model, mode="max-autotune")
            
            self.model.set_train_projector_and_classifier()
            success, ft_path, duration = train_phase(self, "Adapters", self.hp.get('n_epochs_phase_adapters', 0), self.hp.get('lr_phase_adapters', 5e-5), train_dataset, validation_dataset, progress_state)
            total_training_time += duration

            if success:
                self._cleanup()
                self.model = LogSentinelModel(self.llama_model_path, self.hp['encoder_hidden_size'], self.hp, ft_path, True, self.device, self._log)
                self.model = torch.compile(self.model, mode="max-autotune")
                self.model.set_finetuning_all()
                _, ft_path, duration = train_phase(self, "Full_Fine_Tuning", self.hp.get('n_epochs_phase_full', 0), self.hp.get('lr_phase_full', 2e-5), train_dataset, validation_dataset, progress_state)
                total_training_time += duration
                self._cleanup()

            self._log("\n>>>> CONFIGURING MODEL FOR FINAL EVALUATION <<<<")
            self.model = LogSentinelModel(self.llama_model_path, self.hp['encoder_hidden_size'], self.hp, ft_path, False, self.device, self._log)
            self.model = torch.compile(self.model, mode="max-autotune")
            
            if validation_dataset:
                val_metrics, val_duration = evaluate_and_visualize(self, validation_dataset, "validation")
                final_model_metrics.update(val_metrics['validation'])
            
            if test_dataset:
                test_metrics, test_duration = evaluate_and_visualize(self, test_dataset, "test")
                final_model_metrics.update(test_metrics['test'])
                total_testing_time = test_duration
            
            self.visualizer.plot_training_loss(self.batch_losses)
            
            if ft_path and os.path.exists(ft_path):
                final_model_path = self.execution_dir / 'output_model'
                shutil.copytree(ft_path, final_model_path, dirs_exist_ok=True)
                self._log(f"Final model saved to: {final_model_path}")
            
            final_status = 'COMPLETED'
        
        except Exception as e:
            tb_str = traceback.format_exc()
            error_msg = f"CRITICAL ERROR in run {self.run_id}: {e}\n{tb_str}"
            self._log(error_msg)
            if self.callback: self.callback({"error": f"{e}\n{tb_str}"})
            final_status = 'FAILED'
            
        finally:
            total_run_time = time.time() - self.run_start_time
            resource_metrics_history = monitor.stop()
            
            if self.run_id:
                try:
                    if resource_metrics_history and 'summary' in resource_metrics_history and 'time_series' in resource_metrics_history:
                        summary = resource_metrics_history['summary']
                        ram_summary = summary.get('ram', {})
                        gpu_summary = summary.get('gpu', {})
                        db_metrics = {
                            "total_run_time_sec": total_run_time, "training_time_sec": total_training_time,
                            "testing_time_sec": total_testing_time, "accuracy": final_model_metrics.get(f"{'test' if test_dataset else 'validation'}_accuracy"),
                            "precision": final_model_metrics.get(f"{'test' if test_dataset else 'validation'}_precision"), "f1_score": final_model_metrics.get(f"{'test' if test_dataset else 'validation'}_f1_score"),
                            "recall": final_model_metrics.get(f"{'test' if test_dataset else 'validation'}_recall"), "avg_ram_usage_gb": ram_summary.get('avg_ram_usage_gb'),
                            "peak_95_ram_usage_gb": ram_summary.get('p95_ram_usage_gb'), "avg_gpu_vram_gb": gpu_summary.get('avg_gpu_vram_gb'),
                            "peak_95_gpu_vram_gb": gpu_summary.get('p95_gpu_vram_gb')
                        }
                        self.db.save_final_metrics(self.run_id, db_metrics)
                        if self.visualizer:
                            time_series_df = pd.DataFrame(resource_metrics_history['time_series'])
                            self.visualizer.plot_resource_usage(time_series_df)
                    else:
                        self._log("No resource metrics recorded.")
                except Exception as e:
                    self._log(f"Failed to save metrics or plots: {e}")
                report_path_str = str(self.execution_dir) if final_status == 'COMPLETED' else None
                self.db.update_run_status(self.run_id, final_status, report_path_str)
                
            if self.model: self._cleanup()
            if self.callback: self.callback({"status": final_status, "done": True})