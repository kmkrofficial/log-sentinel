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
from utils.database_manager import DatabaseManager
from utils.data_loader import LogDataset
from utils.resource_monitor import ResourceMonitor
from utils.log_visualizer import LogVisualizer
from logsentinel_model import LogSentinelModel
from utils.helpers import merge_data, get_eta, format_time
from engine.data_utils import BalancedSampler

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
        if target:
            del target
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _load_and_tensorize_dataset(self, dataset_path, encoder_model, encoder_tokenizer):
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

                for k in range(0, len(all_logs_flat), log_message_batch_size):
                    log_batch = all_logs_flat[k : k + log_message_batch_size]

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

                if self.is_gui_mode:
                    progress = min((i + 1) / total_chunks, 1.0)
                    self.callback({"status": f"Embedding {dataset_path.name}: Chunk {i+1}/{total_chunks}"})

        self._log("Embedding complete. Tensorizing sequences...")

        tensorized_sequences = []
        tensorized_labels = []
        max_seq_len = self.hp['max_seq_len']

        if not all_embeddings:
             raise RuntimeError(f"No log sequences found after processing {dataset_path.name}.")


        first_valid_tensor = next((t for t in all_embeddings if t.shape[0] > 0), None)
        if first_valid_tensor is None:
            raise RuntimeError(f"All log sequences are empty in {dataset_path.name}.")
        sample_embedding_dim = first_valid_tensor.shape[1]


        for i in tqdm(range(len(all_embeddings)), desc="Tensorizing sequences", disable=self.is_gui_mode):
            seq_embeddings = all_embeddings[i]
            seq_label = all_labels[i]

            if seq_embeddings is None or seq_embeddings.shape[0] == 0:
                padding_len = max_seq_len
                padding = torch.zeros(padding_len, sample_embedding_dim, dtype=torch.float32)
                tensorized_sequences.append(padding)
                tensorized_labels.append(torch.tensor(seq_label, dtype=torch.long))
                continue

            seq_len = seq_embeddings.shape[0]

            if seq_len > max_seq_len:
                tensorized_seq = seq_embeddings[-max_seq_len:]
            elif seq_len < max_seq_len:
                padding_len = max_seq_len - seq_len
                padding = torch.zeros(padding_len, sample_embedding_dim, dtype=torch.float32)
                tensorized_seq = torch.cat([seq_embeddings, padding], dim=0)
            else:
                tensorized_seq = seq_embeddings

            tensorized_sequences.append(tensorized_seq)
            tensorized_labels.append(torch.tensor(seq_label, dtype=torch.long))

        if not tensorized_sequences:
            raise RuntimeError(f"No sequences were tensorized for {dataset_path.name}.")

        sequences_tensor = torch.stack(tensorized_sequences)
        labels_tensor = torch.stack(tensorized_labels)

        self._log(f"Created TensorDataset for {dataset_path.name} with {len(labels_tensor)} samples.")
        return TensorDataset(sequences_tensor, labels_tensor)

    def run_inference(self):
        start_time = time.time()
        try:
            self._log(f"Loading base encoder: {self.encoder_path_str}")
            encoder_config = AutoConfig.from_pretrained(self.encoder_path_str)
            encoder_tokenizer = AutoTokenizer.from_pretrained(self.encoder_path_str)
            encoder_model = AutoModel.from_pretrained(self.encoder_path_str).to(self.device).eval()

            self._log("Preparing dataset for inference...")
            dataset_path = Path("datasets") / self.dataset_name / 'test.csv'

            dataset = self._load_and_tensorize_dataset(dataset_path, encoder_model, encoder_tokenizer)
            self._cleanup(model_to_clean=encoder_model)
            del encoder_tokenizer

            if not dataset:
                raise RuntimeError("Inference dataset could not be loaded.")

            self._log(f"Loading fine-tuned model from: {self.model_run_path}")

            ft_model_path = self.model_run_path / 'output_model'
            if not ft_model_path.exists():
                raise FileNotFoundError(f"Fine-tuned model not found at {ft_model_path}")

            self.model = LogSentinelModel(
                llama_model_path=str(MODELS_DIR / DEFAULT_LLAMA_MODEL.split('/')[-1]),
                encoder_hidden_size=encoder_config.hidden_size,
                hyperparameters=self.hp,
                ft_path=str(ft_model_path),
                is_train_mode=False,
                device=self.device,
                log_callback=self._log
            )
            self.model = torch.compile(self.model, mode="max-autotune")
            self.model.eval()

            loader = DataLoader(
                dataset,
                batch_size=self.hp['micro_batch_size'] * 2,
                num_workers=self.num_workers,
                pin_memory=True
            )

            all_preds = []
            all_probs = []

            with torch.no_grad():
                for sequences, _ in tqdm(loader, desc="Running inference", disable=self.is_gui_mode):
                    sequences = sequences.to(self.device, non_blocking=True)
                    logits, _ = self.model.get_logits(sequences)

                    probs = torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy()
                    preds = (probs > 0.5).astype(int)

                    all_preds.extend(preds)
                    all_probs.extend(probs)

            output_df = pd.DataFrame({
                'prediction': all_preds,
                'anomaly_probability': all_probs
            })

            output_path = self.model_run_path / self.output_filename
            output_df.to_csv(output_path, index=False)

            total_time = time.time() - start_time
            self._log(f"Inference complete in {format_time(total_time)}.")
            self._log(f"Predictions saved to: {output_path}")

        except Exception as e:
            tb_str = traceback.format_exc()
            error_msg = f"CRITICAL ERROR in inference: {e}\n{tb_str}"
            self._log(error_msg)
            if self.callback: self.callback({"error": f"{e}\n{tb_str}"})
        finally:
            self._cleanup(self.model)