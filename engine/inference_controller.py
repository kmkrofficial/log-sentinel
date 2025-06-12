import torch
import os
import pandas as pd
from pathlib import Path

# Local project imports
from config import REPORTS_DIR, DEFAULT_BERT_PATH
from logsentinel_model import LogSentinelModel
from utils.data_loader import replace_patterns

class InferenceController:
    def __init__(self, run_id):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.run_id = run_id
        self.model = None
        
        run_dir = REPORTS_DIR / self.run_id
        self.model_path = run_dir / 'final_model'

        # These would be stored in the DB in a full implementation,
        # but for now we assume we can find them or use defaults.
        # This part needs to be improved by fetching from DB.
        self.model_name = self._get_model_name_from_run(run_id) 

        if not self.model_path.exists() or not self.model_name:
            raise FileNotFoundError(f"Fine-tuned model for run '{run_id}' not found.")
            
        self._load_model()

    def _get_model_name_from_run(self, run_id):
        # Placeholder: In a real app, you'd query the DB for the base model name.
        # For now, we assume the model ID is part of the directory structure or a config file.
        # This is a simplification.
        from utils.database_manager import DatabaseManager
        db = DatabaseManager()
        details = db.get_run_details(run_id)
        db.close()
        return details['run_info'].get('model_name') if details else None

    def _load_model(self):
        """Loads the fine-tuned model for inference."""
        print(f"Loading fine-tuned model from {self.model_path}...")
        self.model = LogSentinelModel(
            bert_path=DEFAULT_BERT_PATH,
            llama_path=self.model_name,
            ft_path=str(self.model_path),
            is_train_mode=False,
            device=self.device
        )
        self.model.eval()
        print("Model loaded in evaluation mode.")

    def _preprocess_sequence(self, raw_sequence_str):
        """Applies the same preprocessing used during training."""
        processed_content = replace_patterns(raw_sequence_str)
        return [line for line in processed_content.split(' ;-; ') if line]

    def predict_single(self, raw_sequence_str):
        """Performs inference on a single raw log sequence string."""
        if not self.model:
            raise RuntimeError("Model is not loaded.")
        
        preprocessed_sequence = self._preprocess_sequence(raw_sequence_str)
        
        with torch.no_grad():
            logits = self.model([preprocessed_sequence]) # Model expects a batch
            prediction = torch.argmax(logits, dim=-1).item()

        result = "Anomalous" if prediction == 1 else "Normal"
        confidence = torch.softmax(logits, dim=-1)[0].max().item()
        
        return {"prediction": result, "confidence": confidence}

    def predict_batch(self, df_batch):
        """Performs inference on a DataFrame of log sequences."""
        if not self.model or 'Content' not in df_batch.columns:
            raise RuntimeError("Model is not loaded or DataFrame is missing 'Content' column.")

        sequences = [self._preprocess_sequence(row) for row in df_batch['Content']]
        
        all_preds = []
        all_confs = []

        with torch.no_grad():
            for i in range(0, len(sequences), 32): # Internal batching for large files
                batch_seq = sequences[i:i+32]
                logits = self.model(batch_seq)
                predictions = torch.argmax(logits, dim=-1)
                confidences = torch.softmax(logits, dim=-1).max(dim=-1).values
                
                all_preds.extend(predictions.cpu().numpy())
                all_confs.extend(confidences.cpu().numpy())
        
        df_batch['Prediction'] = ["Anomalous" if p == 1 else "Normal" for p in all_preds]
        df_batch['Confidence'] = all_confs

        return df_batch