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
        db = DatabaseManager(); details = db.get_run_details(run_id); db.close()
        return details['run_info'].get('model_name') if details else None

    def _load_model(self):
        self._log(f"Loading fine-tuned model from {self.model_path}...")
        self.model = LogSentinelModel(DEFAULT_BERT_PATH, self.model_name, str(self.model_path), False, self.device)
        self.model.eval()
        self._log("Model loaded in evaluation mode.")

    def cleanup(self):
        self._log("Cleaning up inference resources from memory...");
        if hasattr(self, 'model') and self.model is not None: del self.model; self.model = None
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        self._log("Cleanup complete.")

    def _preprocess_sequence(self, raw_sequence_str):
        if not isinstance(raw_sequence_str, str): raw_sequence_str = str(raw_sequence_str)
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

    def predict_batch(self, uploaded_file, internal_batch_size=32):
        if not self.model: raise RuntimeError("Model is not loaded.")
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv', encoding='utf-8') as tmp:
            temp_file_path = tmp.name
        self._log(f"Using true streaming. Output file: {temp_file_path}")

        uploaded_file.seek(0, os.SEEK_END); total_size = uploaded_file.tell(); uploaded_file.seek(0)
        start_time = time.time(); rows_processed = 0
        
        try:
            header = pd.read_csv(uploaded_file, nrows=0).columns.tolist()
            if 'Content' not in header: raise ValueError("'Content' column not found in the input file.")
        except Exception as e: self._log(f"Error reading CSV header: {e}"); raise
        
        uploaded_file.seek(0)
        pd.DataFrame(columns=header + ['Prediction', 'Confidence']).to_csv(temp_file_path, index=False, mode='w')
        
        pd_iterator = pd.read_csv(uploaded_file, chunksize=internal_batch_size, iterator=True, low_memory=False)

        for df_batch in pd_iterator:
            sequences = [self._preprocess_sequence(row) for row in df_batch['Content']]
            
            with torch.no_grad():
                logits = self.model(sequences)
                probas = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probas, dim=-1).cpu().numpy()
                confidences = probas.max(dim=-1).values.cpu().numpy()

            df_batch['Prediction'] = ["Anomalous" if p == 1 else "Normal" for p in predictions]
            df_batch['Confidence'] = confidences
            df_batch.to_csv(temp_file_path, mode='a', header=False, index=False)
            
            rows_processed += len(df_batch)
            current_pos = uploaded_file.tell()
            progress = current_pos / total_size if total_size > 0 else 1
            elapsed = time.time() - start_time
            etc = (elapsed / progress) * (1 - progress) if progress > 0 and progress < 1 else 0

            if self.callback({"progress": progress, "rows_processed": rows_processed, "etc": etc}) == 'STOP':
                self._log("Inference aborted by user."); os.remove(temp_file_path); return None, None
        
        total_time = time.time() - start_time
        performance_metrics = {
            "total_time_sec": total_time,
            "time_per_prediction_ms": (total_time / rows_processed) * 1000 if rows_processed > 0 else 0,
            "total_rows": rows_processed, 
            "rows_per_second": rows_processed / total_time if total_time > 0 else float('inf')
        }
        
        return temp_file_path, performance_metrics