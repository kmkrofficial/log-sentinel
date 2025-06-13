import torch
import os
import pandas as pd
import time
import tempfile
import gc
import numpy as np
from pathlib import Path

from config import REPORTS_DIR, DEFAULT_BERT_PATH
from logsentinel_model import LogSentinelModel
from utils.data_loader import replace_patterns

class InferenceController:
    def __init__(self, run_id, callback=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.run_id = run_id
        self.model = None
        self.callback = callback or (lambda *args: 'CONTINUE')
        
        run_dir = REPORTS_DIR / self.run_id
        self.model_path = run_dir / 'final_model'
        self.model_name = self._get_model_name_from_run(run_id) 
        if not self.model_path.exists() or not self.model_name:
            raise FileNotFoundError(f"Fine-tuned model for run '{run_id}' not found.")
        self._load_model()

    def _log(self, message):
        print(message)
        self.callback({"log": message})

    def _get_model_name_from_run(self, run_id):
        from utils.database_manager import DatabaseManager
        db = DatabaseManager()
        details = db.get_run_details(run_id)
        db.close()
        return details['run_info'].get('model_name') if details else None

    def _load_model(self):
        self._log(f"Loading fine-tuned model from {self.model_path}...")
        self.model = LogSentinelModel(DEFAULT_BERT_PATH, self.model_name, str(self.model_path), False, self.device)
        self.model.eval()
        self._log("Model loaded in evaluation mode.")

    def cleanup(self):
        self._log("Cleaning up inference resources...")
        if hasattr(self, 'model') and self.model is not None:
            del self.model; self.model = None
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        self._log("Cleanup complete.")

    def _preprocess_sequence(self, raw_sequence_str):
        processed_content = replace_patterns(raw_sequence_str)
        return [line for line in processed_content.split(' ;-; ') if line]

    def predict_single(self, raw_sequence_str):
        if not self.model: raise RuntimeError("Model is not loaded.")
        preprocessed_sequence = self._preprocess_sequence(raw_sequence_str)
        with torch.no_grad():
            logits = self.model([preprocessed_sequence])
            prediction = torch.argmax(logits, dim=-1).item()
        result = "Anomalous" if prediction == 1 else "Normal"
        confidence = torch.softmax(logits, dim=-1)[0].max().item()
        return {"prediction": result, "confidence": confidence}

    def predict_batch(self, uploaded_file, internal_batch_size=16, chunk_size=512):
        if not self.model: raise RuntimeError("Model is not loaded.")
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv', encoding='utf-8') as tmp:
            temp_file_path = tmp.name
        self._log(f"Created temporary output file at: {temp_file_path}")

        uploaded_file.seek(0, os.SEEK_END); total_size = uploaded_file.tell(); uploaded_file.seek(0)
        start_time = time.time(); rows_processed = 0
        all_preds, all_probas, all_gt_labels = [], [], []
        
        try:
            header = pd.read_csv(uploaded_file, nrows=0).columns.tolist()
            has_labels = 'Label' in header
        except Exception as e: self._log(f"Error reading CSV header: {e}"); raise
        uploaded_file.seek(0)

        pd.DataFrame(columns=header + ['Prediction', 'Confidence']).to_csv(temp_file_path, index=False, mode='w')
        
        for df_chunk in pd.read_csv(uploaded_file, chunksize=chunk_size, low_memory=False):
            if 'Content' not in df_chunk.columns: raise ValueError("'Content' column not found.")
            sequences = [self._preprocess_sequence(row) for row in df_chunk['Content']]
            
            with torch.no_grad():
                for i in range(0, len(sequences), internal_batch_size):
                    batch_seq = sequences[i:i+internal_batch_size]
                    logits = self.model(batch_seq)
                    probas = torch.softmax(logits, dim=-1)
                    all_preds.extend(torch.argmax(probas, dim=-1).cpu().numpy())
                    all_probas.extend(probas[:, 1].cpu().numpy())

            df_chunk['Prediction'] = ["Anomalous" if p == 1 else "Normal" for p in all_preds[-len(df_chunk):]]
            df_chunk['Confidence'] = [p if pr == 1 else 1-p for pr, p in zip(all_preds[-len(df_chunk):], all_probas[-len(df_chunk):])]
            df_chunk.to_csv(temp_file_path, mode='a', header=False, index=False)
            
            if has_labels: all_gt_labels.extend(df_chunk['Label'].values)
            rows_processed += len(df_chunk)
            progress = uploaded_file.tell() / total_size if total_size > 0 else 1
            elapsed = time.time() - start_time
            etc = (elapsed / progress) * (1 - progress) if progress > 0 else 0
            if self.callback({"progress": progress, "rows_processed": rows_processed, "etc": etc}) == 'STOP':
                self._log("Inference aborted by user."); os.remove(temp_file_path); return None, None, None
        
        total_time = time.time() - start_time
        performance_metrics = {"total_rows": rows_processed, "total_time_sec": total_time, "rows_per_second": rows_processed / total_time if total_time > 0 else float('inf'), "time_per_prediction_ms": (total_time / rows_processed) * 1000 if rows_processed > 0 else 0}
        
        classification_data = None
        if has_labels:
            classification_data = (np.array(all_gt_labels), np.array(all_preds), np.array(all_probas))

        return temp_file_path, performance_metrics, classification_data