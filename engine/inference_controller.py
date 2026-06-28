import os
import gc
import torch
import time
import shutil
import pandas as pd
import traceback
import platform
import numpy as np
import h5py
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datetime import datetime

from config import (
    EXECUTIONS_DIR, MODELS_DIR, DEFAULT_ENCODER_MODEL,
    DEFAULT_LLAMA_MODEL, get_hyperparameters
)
from utils.data_loader import replace_patterns
from logsentinel_model import LogSentinelModel
from utils.helpers import merge_data, format_time
from engine.data_utils import HDF5Dataset
from utils.resource_monitor import ResourceMonitor
from utils.log_visualizer import LogVisualizer

torch.backends.cuda.matmul.allow_tf32 = True

class InferenceController:
    def __init__(self, model_run_path, dataset_name, db_manager, callback=None, is_test_run=False, test_run_percentage=0.3, manual_nickname=None):
        self.model_run_path = Path(model_run_path)
        self.dataset_name = dataset_name
        self.db = db_manager
        self.manual_nickname = manual_nickname
        self.hp = get_hyperparameters(dataset_name)
        self.encoder_path_str = str(MODELS_DIR / DEFAULT_ENCODER_MODEL.split('/')[-1])
        self.callback = callback or (lambda *args: 'CONTINUE')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.is_test_run = is_test_run
        self.test_run_percentage = test_run_percentage
        self.is_gui_mode = callback is not None
        self.num_workers = self.hp.get('dataloader_num_workers', 0)
        self.model = None
        self.run_id = None
        self.execution_dir = None
        self.visualizer = None

    def _log(self, message):
        print(message)
        if self.callback: self.callback({"log": message})

    def _generate_nickname(self):
        if self.manual_nickname:
            return f"{self.manual_nickname}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        base_nickname = f"Inference_{self.dataset_name}_{now}"
        if self.is_test_run:
            base_nickname += f"_{int(self.test_run_percentage * 100)}pct_TEST"
        return base_nickname

    def _initialize_run(self):
        nickname = self._generate_nickname()
        self.run_id = self.db.create_new_run('Inference', DEFAULT_LLAMA_MODEL, self.dataset_name, self.hp, nickname)
        if self.run_id:
            self._log(f"Created new inference run with ID: {self.run_id} (Nickname: {nickname})")
            self.execution_dir = EXECUTIONS_DIR / str(nickname)
            (self.execution_dir / "visualizations").mkdir(parents=True, exist_ok=True)
            self.visualizer = LogVisualizer(plot_dir=self.execution_dir / "visualizations")
        return self.run_id is not None

    def _cleanup(self, model_to_clean=None):
        target = model_to_clean if model_to_clean else self.model
        if target: del target
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _embed_and_save_to_hdf5(self, dataset_path, h5_path, encoder_model, encoder_tokenizer):
        self._log(f"Starting memory-safe embedding to {h5_path.name}...")
        embedding_chunk_size = self.hp['embedding_chunk_size']
        
        encoder_config = AutoConfig.from_pretrained(self.encoder_path_str)
        max_seq_len = self.hp['max_seq_len']
        embedding_dim = encoder_config.hidden_size

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

    def _evaluate_and_visualize(self, dataset, dataset_name):
        self._log(f"Running final evaluation on {dataset_name} dataset...")
        start_time = time.time()
        self.model.eval()
        loader = DataLoader(dataset, batch_size=self.hp['micro_batch_size'], num_workers=self.num_workers, pin_memory=True)
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for i, (sequences, labels) in enumerate(tqdm(loader, desc=f"Evaluating {dataset_name}", disable=self.is_gui_mode)):
                sequences = sequences.to(self.device, non_blocking=True)
                logits, _ = self.model.get_logits(sequences)
                
                probs = torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy()
                preds = (probs > 0.5).astype(int)

                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        eval_time = time.time() - start_time
        accuracy_val = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
        
        metrics = {
            f'{dataset_name}_accuracy': accuracy_val, f'{dataset_name}_precision': precision,
            f'{dataset_name}_recall': recall, f'{dataset_name}_f1_score': f1,
            f'{dataset_name}_inference_time_sec': eval_time,
            f'{dataset_name}_samples_per_sec': len(dataset) / eval_time if eval_time > 0 else 0
        }
        
        self._log(f"Final {dataset_name.upper()} Metrics: " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
        if self.callback: self.callback({"validation_metrics": metrics})
        
        if self.visualizer:
            try:
                self.visualizer.plot_confusion_matrix(all_labels, all_preds, f"{dataset_name}_confusion_matrix")
                self.visualizer.plot_roc_curve(all_labels, all_probs, f"{dataset_name}_roc_curve")
                self.visualizer.plot_precision_recall_curve(all_labels, all_probs, f"{dataset_name}_pr_curve")
                self.visualizer.plot_distributions(all_probs, all_labels, f"{dataset_name}_distributions")
            except Exception as e:
                self._log(f"Failed to generate plots for {dataset_name}: {e}")

        return metrics, all_preds, all_probs

    def run_inference(self):
        monitor = ResourceMonitor()
        monitor.start()
        final_status = 'FAILED'
        temp_embedding_dir = None
        run_start_time = time.time()
        final_model_metrics = {}
        dataset = None

        try:
            if not self._initialize_run():
                raise RuntimeError("Failed to create a new run record.")

            temp_embedding_dir = self.execution_dir / "_temp_inference_embeddings"
            temp_embedding_dir.mkdir(exist_ok=True)
            self._log(f"Using controlled temp directory for embeddings: {temp_embedding_dir}")

            encoder_config = AutoConfig.from_pretrained(self.encoder_path_str)
            encoder_tokenizer = AutoTokenizer.from_pretrained(self.encoder_path_str)
            encoder_model = AutoModel.from_pretrained(self.encoder_path_str).to(self.device).eval()

            self._log("Preparing dataset for inference...")
            dataset_path = Path("datasets") / self.dataset_name / 'test.csv'
            h5_path = temp_embedding_dir / "inference_data.h5"
            self._embed_and_save_to_hdf5(dataset_path, h5_path, encoder_model, encoder_tokenizer)
            dataset = HDF5Dataset(h5_path)
            self._cleanup(model_to_clean=encoder_model)
            del encoder_tokenizer

            if not dataset: raise RuntimeError("Inference dataset could not be loaded.")

            self._log(f"Loading fine-tuned model from: {self.model_run_path}")
            self.model = LogSentinelModel(
                llama_model_path=str(MODELS_DIR / DEFAULT_LLAMA_MODEL.split('/')[-1]),
                encoder_hidden_size=encoder_config.hidden_size, hyperparameters=self.hp,
                ft_path=str(self.model_run_path / 'output_model'), is_train_mode=False, device=self.device, log_callback=self._log
            )
            
            if platform.system() == "Linux":
                self._log("Optimizing for Linux: Enabling torch.compile() for optimized performance.")
                self.model = torch.compile(self.model, mode="max-autotune")
            else:
                self._log("Skipping torch.compile() on non-Linux system for compatibility.")

            test_metrics, all_preds, all_probs = self._evaluate_and_visualize(dataset, "test")
            final_model_metrics.update(test_metrics)
            
            output_df = pd.DataFrame({'prediction': all_preds, 'anomaly_probability': all_probs})
            output_path = self.execution_dir / "predictions.csv"
            output_df.to_csv(output_path, index=False)
            self._log(f"Predictions saved to: {output_path}")

            final_status = 'COMPLETED'

        except Exception as e:
            tb_str = traceback.format_exc()
            self._log(f"CRITICAL ERROR in inference: {e}\n{tb_str}")
            if self.callback: self.callback({"error": f"{e}\n{tb_str}"})
            final_status = 'FAILED'
        
        finally:
            if temp_embedding_dir and temp_embedding_dir.exists():
                self._log(f"Cleaning up temporary inference directory: {temp_embedding_dir}")
                shutil.rmtree(temp_embedding_dir)

            total_run_time = time.time() - run_start_time
            resource_metrics = monitor.stop()
            if self.run_id:
                if final_status == 'COMPLETED':
                    try:
                        num_records = len(dataset) if dataset else 0
                        inference_time = final_model_metrics.get("test_inference_time_sec", 0)
                        time_per_record = (inference_time / num_records) if num_records > 0 else 0
                        
                        summary = resource_metrics.get('summary', {})
                        ram_summary = summary.get('ram', {})
                        gpu_summary = summary.get('gpu', {})

                        report_data = {
                            'Metric': [
                                'Accuracy', 'Precision', 'Recall', 'F1-Score',
                                'Total Run Time (s)', 'Inference Time (s)', 'Inference Time per Record (s)',
                                'Average RAM Usage (GB)', 'Average VRAM Usage (GB)', 'Total Records Evaluated'
                            ],
                            'Value': [
                                final_model_metrics.get("test_accuracy"), final_model_metrics.get("test_precision"),
                                final_model_metrics.get("test_recall"), final_model_metrics.get("test_f1_score"),
                                total_run_time, inference_time, time_per_record,
                                ram_summary.get('avg_ram_usage_gb'), gpu_summary.get('avg_gpu_vram_gb'),
                                num_records
                            ]
                        }
                        report_df = pd.DataFrame(report_data)
                        report_path_csv = self.execution_dir / 'inference_summary_metrics.csv'
                        report_df.to_csv(report_path_csv, index=False)
                        self._log(f"Inference summary metrics saved to: {report_path_csv}")

                    except Exception as e:
                        self._log(f"Failed to generate summary CSV report: {e}")

                try:
                    if resource_metrics and 'summary' in resource_metrics:
                        summary = resource_metrics['summary']
                        db_metrics = {
                            "total_run_time_sec": total_run_time, "testing_time_sec": final_model_metrics.get("test_inference_time_sec"),
                            "accuracy": final_model_metrics.get("test_accuracy"), "precision": final_model_metrics.get("test_precision"),
                            "f1_score": final_model_metrics.get("test_f1_score"), "recall": final_model_metrics.get("test_recall"),
                            "avg_ram_usage_gb": summary.get('ram', {}).get('avg_ram_usage_gb'), "peak_95_ram_usage_gb": summary.get('ram', {}).get('p95_ram_usage_gb'),
                            "avg_gpu_vram_gb": summary.get('gpu', {}).get('avg_gpu_vram_gb'), "peak_95_gpu_vram_gb": summary.get('gpu', {}).get('p95_gpu_vram_gb')
                        }
                        self.db.save_final_metrics(self.run_id, db_metrics)
                        if self.visualizer:
                            self.visualizer.plot_resource_usage(pd.DataFrame(resource_metrics['time_series']))
                except Exception as e:
                    self._log(f"Failed to save final metrics or plots: {e}")
                
                report_path = str(self.execution_dir) if final_status == 'COMPLETED' else None
                self.db.update_run_status(self.run_id, final_status, report_path)

            self._cleanup()
            if self.callback: self.callback({"status": final_status, "done": True})