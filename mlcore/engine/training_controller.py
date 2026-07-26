import os
import gc
import json
import torch
import time
import shutil
import pandas as pd
import traceback
import numpy as np
import h5py
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.utils.data import DataLoader, random_split
from datetime import datetime

from mlcore.config import (
    DATA_DIR, EXECUTIONS_DIR, MODELS_DIR, DEFAULT_ENCODER_MODEL,
    DEFAULT_LLAMA_MODEL, get_hyperparameters
)
from mlcore.utils.data_loader import replace_patterns
from mlcore.utils.resource_monitor import ResourceMonitor
from mlcore.logsentinel_model import LogSentinelModel
from mlcore.utils.helpers import merge_data, get_eta, format_time
from mlcore.utils.runtime_compat import maybe_compile_model
from mlcore.engine.phase_manager import train_phase, evaluate_and_visualize
from mlcore.engine.data_utils import BalancedSampler, HDF5Dataset

torch.backends.cuda.matmul.allow_tf32 = True

class TrainingController:
    def __init__(self, dataset_name, callback=None, is_test_run=False, test_run_percentage=0.3):
        self.dataset_name = dataset_name
        self.hp = get_hyperparameters(dataset_name)
        self.llama_model_path = str(MODELS_DIR / DEFAULT_LLAMA_MODEL.split('/')[-1])
        self.encoder_path_str = str(MODELS_DIR / DEFAULT_ENCODER_MODEL.split('/')[-1])
        self.callback = callback or (lambda *args: 'CONTINUE')
        self.run_id = None
        self.run_nickname = None
        self.model = None
        self.execution_dir = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.run_start_time = 0
        self.is_test_run = is_test_run
        self.test_run_percentage = test_run_percentage
        self.batch_losses = []
        self.run_metrics = {"training_loss": [], "resource_usage": {}, "evaluation": {}}
        self.num_workers = self.hp.get('dataloader_num_workers', 0)
        self.is_gui_mode = callback is not None

        if self.is_test_run:
            self._log(f"--- QUICK TEST RUN MODE ACTIVATED ({self.test_run_percentage*100:.0f}% data) ---")

    def _emit_callback(self, payload):
        if self.callback:
            self.callback(payload)

    def _log(self, message):
        print(message)
        self._emit_callback({"log": message})

    def _generate_nickname(self):
        if self.run_nickname:
            return self.run_nickname

        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        base_nickname = f"{self.dataset_name}_{now}"
        if self.is_test_run:
            base_nickname += f"_{int(self.test_run_percentage * 100)}pct_TEST"
        self.run_nickname = base_nickname
        return self.run_nickname

    def _initialize_run(self):
        nickname = self._generate_nickname()
        self.execution_dir = EXECUTIONS_DIR / str(nickname)
        self.execution_dir.mkdir(parents=True, exist_ok=True)
        self.run_metrics = {"training_loss": [], "resource_usage": {}, "evaluation": {}}
        self._log(f"Created new training execution directory: {self.execution_dir}")
        return True

    def _to_serializable(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: self._to_serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_serializable(item) for item in value]
        return value

    def _write_run_metrics(self):
        if not self.execution_dir:
            return

        metrics_path = self.execution_dir / "run_metrics.json"
        with metrics_path.open('w', encoding='utf-8') as metrics_file:
            json.dump(self._to_serializable(self.run_metrics), metrics_file, indent=2)

    def _cleanup(self, model_to_clean=None):
        if model_to_clean is None or model_to_clean is self.model:
            self.model = None
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _embed_and_save_to_hdf5(self, dataset_path, h5_path, encoder_model, encoder_tokenizer, progress_start=0.0, progress_end=1.0):
        self._log(f"Starting memory-safe embedding to {h5_path.name}...")
        embedding_chunk_size = self.hp['embedding_chunk_size']
        
        max_seq_len = self.hp['max_seq_len']
        embedding_dim = self.hp['encoder_hidden_size']

        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('sequences', (0, max_seq_len, embedding_dim), maxshape=(None, max_seq_len, embedding_dim), dtype='f4', chunks=(64, max_seq_len, embedding_dim))
            f.create_dataset('labels', (0,), maxshape=(None,), dtype='i8', chunks=(1024,))

        try:
            total_chunks = sum(1 for _ in pd.read_csv(dataset_path, chunksize=embedding_chunk_size))
        except Exception as e:
            self._log(f"Could not read dataset {dataset_path}: {e}")
            return

        with pd.read_csv(dataset_path, chunksize=embedding_chunk_size, dtype={'Content': str}) as reader:
            pbar = tqdm(reader, desc=f"Processing {dataset_path.name}", disable=self.is_gui_mode, unit=" chunks", total=total_chunks)
            for i, chunk_df in enumerate(pbar):
                if self.is_gui_mode:
                    local_progress = (i + 0.5) / total_chunks if total_chunks > 0 else 0
                    global_progress = progress_start + (local_progress * (progress_end - progress_start))
                    self._emit_callback({"status": f"Embedding {dataset_path.name}: Chunk {i+1}/{total_chunks}", "progress": global_progress})

                if self.is_test_run: chunk_df = chunk_df.sample(frac=self.test_run_percentage, random_state=42)

                chunk_df['Processed_Content'] = chunk_df['Content'].apply(replace_patterns)
                sequences_in_chunk = [content.split(' ;-; ') for content in chunk_df['Processed_Content'].values]
                chunk_labels = chunk_df['Label'].fillna(-1).astype(int).values
                all_logs_flat, start_positions = merge_data(sequences_in_chunk)

                if not all_logs_flat: continue

                all_line_embeddings = []
                for k in range(0, len(all_logs_flat), 512):
                    log_batch = all_logs_flat[k : k + 512]
                    with torch.no_grad():
                        inputs = encoder_tokenizer(log_batch, return_tensors="pt", padding=True, truncation=True, max_length=self.hp['max_content_len']).to(self.device)
                        model_output = encoder_model(**inputs)
                        embeddings = self._mean_pooling(model_output, inputs['attention_mask'])
                        all_line_embeddings.append(embeddings.cpu())

                if not all_line_embeddings: continue

                all_line_embeddings_tensor = torch.cat(all_line_embeddings, dim=0)
                chunk_embeddings = list(torch.tensor_split(all_line_embeddings_tensor, start_positions[1:-1]))

                tensorized_sequences_list, tensorized_labels_list = [], []
                for seq_embeddings, seq_label in zip(chunk_embeddings, chunk_labels):
                    seq_len = seq_embeddings.shape[0]
                    if seq_len == 0:
                        tensorized_seq = torch.zeros(max_seq_len, embedding_dim, dtype=torch.float32)
                    elif seq_len > max_seq_len:
                        tensorized_seq = seq_embeddings[-max_seq_len:]
                    else:
                        tensorized_seq = torch.cat([seq_embeddings, torch.zeros(max_seq_len - seq_len, embedding_dim, dtype=torch.float32)], dim=0)
                    tensorized_sequences_list.append(tensorized_seq)
                    tensorized_labels_list.append(torch.tensor(seq_label, dtype=torch.long))

                if not tensorized_sequences_list: continue
                
                sequences_to_save = torch.stack(tensorized_sequences_list).numpy()
                labels_to_save = torch.stack(tensorized_labels_list).numpy()

                with h5py.File(h5_path, 'a') as f:
                    num_new = len(labels_to_save)
                    f['sequences'].resize((f['sequences'].shape[0] + num_new, max_seq_len, embedding_dim))
                    f['sequences'][-num_new:] = sequences_to_save
                    f['labels'].resize((f['labels'].shape[0] + num_new,))
                    f['labels'][-num_new:] = labels_to_save
                
                del chunk_df, sequences_in_chunk, chunk_labels, all_logs_flat, all_line_embeddings, all_line_embeddings_tensor, chunk_embeddings
                gc.collect()

        self._log(f"Finished embedding to HDF5 file: {h5_path.name}")

    def run(self):
        monitor = ResourceMonitor()
        monitor.start()
        final_status = 'FAILED'
        total_training_time, total_testing_time = 0, 0
        final_model_metrics = {}
        embedding_root_dir = None
        resource_metrics = {}
        test_dataset = None
        result = None

        try:
            if not self._initialize_run():
                raise RuntimeError("Failed to create a new run record.")
            self.run_start_time = time.time()

            embedding_root_dir = self.execution_dir / "_temp_embeddings"
            embedding_root_dir.mkdir(exist_ok=True)
            self._log(f"Using controlled temp directory for embeddings: {embedding_root_dir}")

            encoder_config = AutoConfig.from_pretrained(self.encoder_path_str)
            self.hp['encoder_hidden_size'] = encoder_config.hidden_size

            self._log("Loading embedding model...")
            encoder_tokenizer = AutoTokenizer.from_pretrained(self.encoder_path_str)
            encoder_model = AutoModel.from_pretrained(self.encoder_path_str).to(self.device).eval()

            self._log("Preparing datasets...")
            train_dataset_path = DATA_DIR / self.dataset_name / 'train.csv'
            test_dataset_path = DATA_DIR / self.dataset_name / 'test.csv'
            
            train_h5_path = embedding_root_dir / "train.h5"
            self._embed_and_save_to_hdf5(train_dataset_path, train_h5_path, encoder_model, encoder_tokenizer, progress_start=0.0, progress_end=0.45)
            full_train_dataset = HDF5Dataset(train_h5_path)

            test_dataset = None
            if test_dataset_path.exists():
                test_h5_path = embedding_root_dir / "test.h5"
                self._embed_and_save_to_hdf5(test_dataset_path, test_h5_path, encoder_model, encoder_tokenizer, progress_start=0.45, progress_end=0.5)
                test_dataset = HDF5Dataset(test_h5_path)

            self._log("Creating a 90/10 train/validation split from the training data.")
            train_size = int(0.9 * len(full_train_dataset))
            validation_size = len(full_train_dataset) - train_size
            train_dataset, validation_dataset = random_split(full_train_dataset, [train_size, validation_size])

            self._cleanup(model_to_clean=encoder_model)
            del encoder_model, encoder_tokenizer

            ft_path = None
            
            self.model = LogSentinelModel(self.llama_model_path, self.hp['encoder_hidden_size'], self.hp, ft_path, True, self.device, self._log)
            self.model = maybe_compile_model(self.model, self._log)

            self.model.set_train_projector_and_classifier()
            success, ft_path, duration = train_phase(self, "Adapters", self.hp.get('n_epochs_phase_adapters', 0), self.hp.get('lr_phase_adapters', 5e-5), train_dataset, validation_dataset, {}, progress_start=0.5, progress_end=0.75)
            total_training_time += duration

            if success:
                self._cleanup()
                self.model = LogSentinelModel(self.llama_model_path, self.hp['encoder_hidden_size'], self.hp, ft_path, True, self.device, self._log)
                self.model = maybe_compile_model(self.model, self._log)

                self.model.set_finetuning_all()
                _, ft_path, duration = train_phase(self, "Full_Fine_Tuning", self.hp.get('n_epochs_phase_full', 0), self.hp.get('lr_phase_full', 2e-5), train_dataset, validation_dataset, {}, progress_start=0.75, progress_end=1.0)
                total_training_time += duration
                self._cleanup()

            self._log("\n>>>> CONFIGURING MODEL FOR FINAL EVALUATION <<<<")
            self.model = LogSentinelModel(self.llama_model_path, self.hp['encoder_hidden_size'], self.hp, ft_path, False, self.device, self._log)
            self.model = maybe_compile_model(self.model, self._log)
            
            if validation_dataset:
                val_metrics, val_duration = evaluate_and_visualize(self, validation_dataset, "validation")
                final_model_metrics.update(val_metrics['validation'])

            if test_dataset:
                test_metrics, test_duration = evaluate_and_visualize(self, test_dataset, "test")
                final_model_metrics.update(test_metrics['test'])
                total_testing_time = test_duration

            if ft_path and os.path.exists(ft_path):
                shutil.copytree(ft_path, self.execution_dir / 'output_model', dirs_exist_ok=True)
                self._log(f"Final model saved to: {self.execution_dir / 'output_model'}")

            final_status = 'COMPLETED'

        except Exception as e:
            tb_str = traceback.format_exc()
            self._log(f"CRITICAL ERROR in run {self.run_id}: {e}\n{tb_str}")
            self._emit_callback({"status": "FAILED", "error": f"{e}\n{tb_str}", "run_id": self.run_id, "execution_dir": str(self.execution_dir) if self.execution_dir else None})
            final_status = 'FAILED'

        finally:
            if embedding_root_dir and embedding_root_dir.exists():
                self._log(f"Cleaning up temporary embedding directory: {embedding_root_dir}")
                shutil.rmtree(embedding_root_dir)

            total_run_time = time.time() - self.run_start_time if self.run_start_time else 0
            resource_metrics = monitor.stop()
            self.run_metrics['training_loss'] = self._to_serializable(self.batch_losses)
            self.run_metrics['resource_usage'] = self._to_serializable(resource_metrics.get('time_series', {})) if resource_metrics else {}
            if self.execution_dir:
                try:
                    self._write_run_metrics()
                except Exception as e:
                    self._log(f"Failed to write run metrics JSON: {e}")

            result = {
                "status": final_status,
                "execution_dir": str(self.execution_dir) if self.execution_dir else None,
                "run_metrics_path": str(self.execution_dir / 'run_metrics.json') if self.execution_dir else None,
                "metrics": self._to_serializable(final_model_metrics),
                "resource_summary": self._to_serializable(resource_metrics.get('summary', {})) if resource_metrics else {},
                "total_run_time_sec": total_run_time,
                "training_time_sec": total_training_time,
                "testing_time_sec": total_testing_time,
            }

            self._cleanup()
            self._emit_callback({
                "status": final_status,
                "progress": 1.0 if final_status == 'COMPLETED' else None,
                "done": True,
                "run_id": self.run_id,
                "execution_dir": str(self.execution_dir) if self.execution_dir else None,
                "metrics": result["metrics"] if result else None,
            })

        return result