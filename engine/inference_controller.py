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
from torch.utils.data import TensorDataset, DataLoader

from config import (
    EXECUTIONS_DIR, MODELS_DIR, DEFAULT_ENCODER_MODEL,
    DEFAULT_LLAMA_MODEL, get_hyperparameters
)
from utils.data_loader import replace_patterns
from logsentinel_model import LogSentinelModel
from utils.helpers import merge_data, format_time

torch.backends.cuda.matmul.allow_tf32 = True

class InferenceController:
    def __init__(self, model_run_path, dataset_name, output_filename, callback=None, is_test_run=False, test_run_percentage=0.3):
        self.model_run_path = Path(model_run_path)
        self.dataset_name = dataset_name
        self.output_filename = output_filename
        self.hp = get_hyperparameters(dataset_name)
        self.encoder_path_str = str(MODELS_DIR / DEFAULT_ENCODER_MODEL.split('/')[-1])
        self.callback = callback or (lambda *args: 'CONTINUE')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.is_test_run = is_test_run
        self.test_run_percentage = test_run_percentage
        self.is_gui_mode = callback is not None
        self.num_workers = self.hp.get('dataloader_num_workers', 0)
        self.model = None

    def _log(self, message):
        print(message)
        if self.callback: self.callback({"log": message})

    def _cleanup(self, model_to_clean=None):
        target = model_to_clean if model_to_clean else self.model
        if target: del target
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _load_and_tensorize_dataset(self, dataset_path, encoder_model, encoder_tokenizer, temp_dir):
        self._log(f"Starting memory-safe embedding to disk for {dataset_path.name}...")
        embedding_chunk_size = self.hp['embedding_chunk_size']
        
        try:
            total_chunks = sum(1 for _ in pd.read_csv(dataset_path, chunksize=embedding_chunk_size))
        except Exception as e:
            self._log(f"Could not read dataset {dataset_path}: {e}")
            return None

        chunk_files = []
        with pd.read_csv(dataset_path, chunksize=embedding_chunk_size, dtype={'Content': str}) as reader:
            pbar = tqdm(reader, desc=f"Processing {dataset_path.name}", disable=self.is_gui_mode, unit=" chunks", total=total_chunks)
            for i, chunk_df in enumerate(pbar):
                if self.is_test_run:
                    chunk_df = chunk_df.sample(frac=self.test_run_percentage, random_state=42)

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
                        all_line_embeddings.append(self._mean_pooling(encoder_model(**inputs), inputs['attention_mask']).cpu())
                
                if not all_line_embeddings: continue
                    
                all_line_embeddings_tensor = torch.cat(all_line_embeddings, dim=0)
                chunk_embeddings = list(torch.tensor_split(all_line_embeddings_tensor, start_positions[1:-1]))
                
                tensorized_sequences, tensorized_labels = [], []
                max_seq_len, sample_embedding_dim = self.hp['max_seq_len'], all_line_embeddings_tensor.shape[1]

                for seq_embeddings, seq_label in zip(chunk_embeddings, chunk_labels):
                    seq_len = seq_embeddings.shape[0]
                    if seq_len == 0:
                        tensorized_seq = torch.zeros(max_seq_len, sample_embedding_dim, dtype=torch.float32)
                    elif seq_len > max_seq_len:
                        tensorized_seq = seq_embeddings[-max_seq_len:]
                    else:
                        tensorized_seq = torch.cat([seq_embeddings, torch.zeros(max_seq_len - seq_len, sample_embedding_dim, dtype=torch.float32)], dim=0)
                    tensorized_sequences.append(tensorized_seq)
                    tensorized_labels.append(torch.tensor(seq_label, dtype=torch.long))

                if tensorized_sequences:
                    chunk_file = Path(temp_dir) / f"chunk_{i}.pt"
                    torch.save((torch.stack(tensorized_sequences), torch.stack(tensorized_labels)), chunk_file)
                    chunk_files.append(chunk_file)
                
                del chunk_df, sequences_in_chunk, chunk_labels, all_logs_flat, all_line_embeddings, all_line_embeddings_tensor, chunk_embeddings
                gc.collect()

        self._log("Assembling final TensorDataset from disk...")
        all_tensors = [torch.load(f) for f in tqdm(chunk_files, desc="Assembling dataset")]
        if not all_tensors: raise RuntimeError(f"No data processed for {dataset_path.name}.")
        return TensorDataset(torch.cat([t[0] for t in all_tensors]), torch.cat([t[1] for t in all_tensors]))
            
    def run_inference(self):
        start_time = time.time()
        temp_embedding_dir = None
        try:
            # --- START OF FIX: Create a controlled temporary directory ---
            temp_embedding_dir = self.model_run_path / "_temp_inference_embeddings"
            temp_embedding_dir.mkdir(exist_ok=True)
            self._log(f"Using controlled temp directory for embeddings: {temp_embedding_dir}")
            # --- END OF FIX ---

            encoder_config = AutoConfig.from_pretrained(self.encoder_path_str)
            encoder_tokenizer = AutoTokenizer.from_pretrained(self.encoder_path_str)
            encoder_model = AutoModel.from_pretrained(self.encoder_path_str).to(self.device).eval()

            self._log("Preparing dataset for inference...")
            dataset_path = Path("datasets") / self.dataset_name / 'test.csv'
            dataset = self._load_and_tensorize_dataset(dataset_path, encoder_model, encoder_tokenizer, temp_embedding_dir)
            self._cleanup(model_to_clean=encoder_model)
            del encoder_tokenizer
            
            if not dataset: raise RuntimeError("Inference dataset could not be loaded.")

            self._log(f"Loading fine-tuned model from: {self.model_run_path}")
            ft_model_path = self.model_run_path / 'output_model'
            if not ft_model_path.exists():
                raise FileNotFoundError(f"Fine-tuned model not found at {ft_model_path}")

            self.model = LogSentinelModel(
                llama_model_path=str(MODELS_DIR / DEFAULT_LLAMA_MODEL.split('/')[-1]),
                encoder_hidden_size=encoder_config.hidden_size, hyperparameters=self.hp,
                ft_path=str(ft_model_path), is_train_mode=False, device=self.device, log_callback=self._log
            )
            self.model = torch.compile(self.model, mode="max-autotune")
            self.model.eval()

            loader = DataLoader(dataset, batch_size=self.hp['micro_batch_size'] * 2, num_workers=self.num_workers, pin_memory=True)
            all_preds, all_probs = [], []
            
            with torch.no_grad():
                for i, (sequences, _) in enumerate(tqdm(loader, desc="Running inference", disable=self.is_gui_mode)):
                    sequences = sequences.to(self.device, non_blocking=True)
                    logits, _ = self.model.get_logits(sequences)
                    probs = torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy()
                    all_preds.extend((probs > 0.5).astype(int))
                    all_probs.extend(probs)
            
            output_df = pd.DataFrame({'prediction': all_preds, 'anomaly_probability': all_probs})
            output_path = self.model_run_path / self.output_filename
            output_df.to_csv(output_path, index=False)
            
            self._log(f"Inference complete in {format_time(time.time() - start_time)}.")
            self._log(f"Predictions saved to: {output_path}")

        except Exception as e:
            tb_str = traceback.format_exc()
            self._log(f"CRITICAL ERROR in inference: {e}\n{tb_str}")
            if self.callback: self.callback({"error": f"{e}\n{tb_str}"})
        finally:
            # --- START OF FIX: Robust cleanup ---
            if temp_embedding_dir and temp_embedding_dir.exists():
                self._log(f"Cleaning up temporary inference directory: {temp_embedding_dir}")
                shutil.rmtree(temp_embedding_dir)
            # --- END OF FIX ---
            self._cleanup()