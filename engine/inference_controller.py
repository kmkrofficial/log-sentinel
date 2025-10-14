import os
import gc
import time
import shutil
import platform
import tempfile
import numpy as np
import pandas as pd
import torch
import traceback
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from config import REPORTS_DIR, DEFAULT_BERT_PATH
from utils.database_manager import DatabaseManager
from utils.data_loader import LogDataset, replace_patterns
from utils.helpers import merge_data
from utils.embedding_cacher import EmbeddingCacher
from prepareData.tensorize_embeddings import tensorize_dataset
from logsentinel_model import LogSentinelModel


class InferenceController:
    def __init__(self, trained_run_id, db_manager, callback=None):
        self.trained_run_id = trained_run_id
        self.db = db_manager
        self.callback = callback or (lambda *args: 'CONTINUE')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.run_id = None
        self.trained_model_report_dir = REPORTS_DIR / str(self.trained_run_id)
        self.ft_path = self.trained_model_report_dir / 'final_model'
        
        run_details = self.db.get_run_details(self.trained_run_id)
        if not run_details:
            raise FileNotFoundError(f"Could not find details for run ID '{self.trained_run_id}'")
        
        self.model_name = run_details['run_info'].get('model_name')
        self.hyperparameters = run_details.get('hyperparameters', {})
        
        if not self.ft_path.exists() or not self.model_name:
            raise FileNotFoundError(f"Fine-tuned model not found for run '{self.trained_run_id}' in {self.ft_path}")

    def _log(self, message):
        print(message)
        if self.callback: self.callback({"log": message})

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _prepare_inference_data(self, input_file_path):
        self._log("Preparing inference data...")
        df = pd.read_csv(input_file_path)
        
        # This is the critical bug fix: apply the same preprocessing as in training
        df['Processed_Content'] = df['Content'].apply(replace_patterns)
        
        sequences = [content.split(' ;-; ') for content in df['Processed_Content'].values]
        labels = df['Label'].fillna(-1).astype(int).values if 'Label' in df.columns else np.full(len(sequences), -1, dtype=int)

        # Generate embeddings
        encoder_path = DEFAULT_BERT_PATH
        encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_path)
        encoder_model = AutoModel.from_pretrained(encoder_path).to(self.device).eval()

        all_logs_flat, start_positions = merge_data(sequences)
        all_line_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(all_logs_flat), 256), desc="Generating Embeddings"):
                batch_logs = all_logs_flat[i:i+256]
                inputs = encoder_tokenizer(batch_logs, return_tensors="pt", padding=True, truncation=True, max_length=self.hyperparameters['max_content_len']).to(self.device)
                model_output = encoder_model(**inputs)
                line_embeddings = self._mean_pooling(model_output, inputs['attention_mask'])
                all_line_embeddings.append(line_embeddings.cpu())

        all_line_embeddings_tensor = torch.cat(all_line_embeddings, dim=0)
        embeddings = list(torch.tensor_split(all_line_embeddings_tensor, start_positions[1:]))

        del encoder_model, encoder_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        # Tensorize the embeddings
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
            torch.save({'embeddings': embeddings, 'labels': torch.from_numpy(labels)}, tmp.name)
            tmp_path = Path(tmp.name)
        
        tensorize_dataset(tmp_path, self.hyperparameters['max_seq_len'])
        
        tensorized_path = tmp_path.with_suffix('.tensor.pt')
        if tensorized_path.exists():
            data = torch.load(tensorized_path)
            tmp_path.unlink()
            tensorized_path.unlink()
            return TensorDataset(data['sequences'], data['labels'])
        else:
            raise RuntimeError("Failed to tensorize inference dataset.")

    def run(self, input_file_path: str, mode: str, internal_batch_size: int = 32):
        from utils.resource_monitor import ResourceMonitor
        from utils.log_visualizer import LogVisualizer
        from engine.phase_manager import evaluate_and_visualize

        self.run_start_time = time.time()
        run_type = 'Testing' if mode == 'testing' else 'Inference'
        
        self.run_id = self.db.create_new_run(run_type, f"run_{self.trained_run_id}", os.path.basename(input_file_path), {"mode": mode, "batch_size": internal_batch_size})
        if not self.run_id: raise RuntimeError("Failed to create new run in the database.")
        
        report_dir = REPORTS_DIR / str(self.run_id)
        report_dir.mkdir(exist_ok=True)
        self.visualizer = LogVisualizer(plot_dir=report_dir)
        monitor = ResourceMonitor()
        monitor.start()
        
        final_status, results = 'FAILED', None
        try:
            dataset = self._prepare_inference_data(input_file_path)
            
            model = LogSentinelModel(self.model_name, self.hyperparameters['encoder_hidden_size'], self.hyperparameters, self.ft_path, False, self.device)
            if platform.system() == "Linux": model = torch.compile(model)
            
            if mode == 'testing':
                if (dataset.tensors[1] == -1).all():
                    raise ValueError("Testing mode requires a 'Label' column with valid labels.")
                perf_metrics = evaluate_and_visualize(self, dataset, "test")
            else: # Inference mode
                all_probas = []
                loader = DataLoader(dataset, batch_size=internal_batch_size)
                with torch.no_grad():
                    for sequences, _ in tqdm(loader, desc="Inference"):
                        sequences = sequences.to(self.device)
                        logits, _ = model.get_logits(sequence_tensor_batch=sequences)
                        probas = torch.softmax(logits, dim=-1)
                        all_probas.extend(probas[:, 1].cpu().numpy())
                
                df = pd.read_csv(input_file_path)
                df['Prediction'] = ["Anomalous" if p > 0.5 else "Normal" for p in all_probas]
                df['Confidence'] = all_probas
                
                output_csv_path = report_dir / f"inference_results_{self.run_id}.csv"
                df.to_csv(output_csv_path, index=False)
                results = str(output_csv_path)
                perf_metrics = {}

            total_run_time = time.time() - self.run_start_time
            time_per_record_ms = (total_run_time / len(dataset)) * 1000 if len(dataset) > 0 else 0
            
            perf_metrics.setdefault('overall', {}).update({
                "total_run_time_sec": total_run_time,
                "time_per_record_ms": time_per_record_ms
            })
            
            self.db.save_performance_metrics(self.run_id, perf_metrics)
            final_status = 'COMPLETED'
        except Exception as e:
            tb_str = traceback.format_exc()
            error_msg = f"CRITICAL ERROR in run {self.run_id}: {e}\n{tb_str}"
            self._log(error_msg)
            if self.callback: self.callback({"error": f"{e}\n{tb_str}"})
            final_status = 'FAILED'
        finally:
            resource_metrics = monitor.stop()
            if self.run_id:
                self.db.save_resource_metrics(self.run_id, resource_metrics)
                if self.visualizer: self.visualizer.plot_resource_usage(resource_metrics)
                self.db.update_run_status(self.run_id, final_status, str(report_dir) if final_status == 'COMPLETED' else None)
            
            if self.callback: self.callback({"status": final_status, "done": True, "result": results})

