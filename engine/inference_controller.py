import torch
import os
import pandas as pd
import time
import tempfile
import gc
import shutil
import numpy as np
from pathlib import Path
from torch import nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc

from config import REPORTS_DIR, DEFAULT_BERT_PATH, MODELS_DIR
from utils.data_loader import LogDataset, replace_patterns
from utils.model_loader import load_model_and_tokenizer
from utils.helpers import merge_data, stack_and_pad_left, safe_np_array
from transformers import BertTokenizerFast, BertModel, AutoConfig
from peft import PeftModel

class InferenceController:
    def __init__(self, trained_run_id, db_manager, callback=None):
        self.trained_run_id = trained_run_id
        self.db = db_manager
        self.callback = callback or (lambda *args: 'CONTINUE')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.run_id = None
        self.trained_model_report_dir = REPORTS_DIR / self.trained_run_id
        self.ft_path = self.trained_model_report_dir / 'final_model'
        self.model_name = self._get_model_name_from_db(self.trained_run_id)
        if not self.ft_path.exists() or not self.model_name:
            raise FileNotFoundError(f"Fine-tuned model components not found for run '{self.trained_run_id}' in {self.ft_path}")

    def _log(self, message):
        print(message)
        self.callback({"log": message})

    def _update_progress(self, processed, total, total_start_time):
        if self.callback({"stop_requested": True}) == "STOP":
            raise InterruptedError("Stop request received by controller.")
        progress = processed / total if total > 0 else 0
        elapsed_total = time.time() - total_start_time
        etc = (elapsed_total / progress) * (1 - progress) if progress > 0 else 0
        status = {"progress": progress, "rows_processed": processed, "etc": etc}
        return self.callback(status)

    def _get_model_name_from_db(self, run_id):
        details = self.db.get_run_details(run_id)
        return details['run_info'].get('model_name') if details else None

    def _run_bert_phase(self, dataset, temp_embedding_dir, sequence_batch_size):
        self._log("Phase 1: Starting BERT embedding generation.")
        bert_model = BertModel.from_pretrained(
            DEFAULT_BERT_PATH, 
            gradient_checkpointing=True
        ).to(self.device).eval()
        bert_tokenizer = BertTokenizerFast.from_pretrained(DEFAULT_BERT_PATH)
        llama_config = AutoConfig.from_pretrained(MODELS_DIR / self.model_name if (MODELS_DIR / self.model_name).exists() else self.model_name)
        projector = nn.Sequential(nn.Linear(bert_model.config.hidden_size, llama_config.hidden_size), nn.GELU(), nn.Linear(llama_config.hidden_size, llama_config.hidden_size)).to(self.device).eval()
        projector.load_state_dict(torch.load(os.path.join(self.ft_path, 'projector.pt')))
        
        total_rows, num_sequence_batches, processed_rows = len(dataset), 0, 0
        
        with torch.inference_mode():
            for i in tqdm(range(0, total_rows, sequence_batch_size), desc="BERT Phase (Sequence Batches)"):
                end_idx = min(i + sequence_batch_size, total_rows)
                seqs, _ = dataset.get_batch(list(range(i, end_idx)))
                
                merged_logs, start_positions = merge_data(seqs)
                
                if not merged_logs:
                    processed_rows += len(seqs)
                    continue

                # --- FIX: Inner-loop batching to process log lines ---
                all_projected_outputs = []
                physical_log_batch_size = 64 # Process 64 log lines at a time
                for j in range(0, len(merged_logs), physical_log_batch_size):
                    log_batch = merged_logs[j:j+physical_log_batch_size]
                    inputs = bert_tokenizer(log_batch, return_tensors="pt", max_length=128, padding=True, truncation=True).to(self.device)
                    bert_outputs = bert_model(**inputs).pooler_output
                    projected_batch_outputs = projector(bert_outputs)
                    all_projected_outputs.append(projected_batch_outputs.cpu())
                
                if not all_projected_outputs:
                    processed_rows += len(seqs)
                    continue

                projected_outputs = torch.cat(all_projected_outputs, dim=0)
                # --- END FIX ---
                
                torch.save({'data': projected_outputs, 'pos': start_positions}, os.path.join(temp_embedding_dir, f"batch_{num_sequence_batches}.pt"))
                num_sequence_batches += 1
                processed_rows += len(seqs)

                if self._update_progress(processed_rows, total_rows, self.run_start_time) == "STOP":
                    raise InterruptedError("Stop request received.")
        
        del bert_model, projector, bert_tokenizer, llama_config; gc.collect(); torch.cuda.empty_cache()
        self._log("Phase 1 Complete. BERT and Projector released from memory.")
        return num_sequence_batches

    def _run_llama_phase(self, dataset, temp_embedding_dir, num_batches):
        self._log("Phase 2: Starting Llama classification.")
        llama_model, llama_tokenizer = load_model_and_tokenizer(self.model_name, is_train_mode=False)
        llama_model = PeftModel.from_pretrained(llama_model, os.path.join(self.ft_path, 'Llama_ft'), is_trainable=False).eval()
        classifier = nn.Linear(llama_model.config.hidden_size, 2).to(self.device).eval()
        classifier.load_state_dict(torch.load(os.path.join(self.ft_path, 'classifier.pt')))
        instruc_tokens = llama_tokenizer(['Below is a sequence of system log messages:'], return_tensors="pt", padding=True).to(self.device)
        
        all_preds, all_probas, processed_rows, total_rows = [], [], 0, len(dataset)
        with torch.inference_mode():
            for i in tqdm(range(num_batches), desc="Llama Phase"):
                batch_data = torch.load(os.path.join(temp_embedding_dir, f"batch_{i}.pt"))
                projected_outputs, start_positions = batch_data['data'].to(self.device), batch_data['pos']
                if not start_positions: continue
                
                seq_embeddings = list(torch.tensor_split(projected_outputs, start_positions[1:], dim=0)) if len(start_positions) > 1 else [projected_outputs]
                valid_embeddings = [torch.cat([llama_model.get_input_embeddings()(instruc_tokens['input_ids'])[0], se], dim=0) for se in seq_embeddings if se is not None and se.shape[0] > 0]
                if not valid_embeddings:
                    processed_rows += len(seq_embeddings)
                    continue
                
                inputs_embeds, attention_mask = stack_and_pad_left(valid_embeddings)
                outputs = llama_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True)
                sequence_lengths = attention_mask.sum(dim=1) - 1
                cls_input_hidden_state = outputs.hidden_states[-1][torch.arange(len(valid_embeddings), device=self.device), sequence_lengths]
                probas = torch.softmax(classifier(cls_input_hidden_state), dim=-1)
                all_preds.extend(torch.argmax(probas, dim=-1).cpu().numpy())
                all_probas.extend(probas[:, 1].cpu().numpy())
                
                processed_rows += len(seq_embeddings)
                if self._update_progress(processed_rows, total_rows, self.run_start_time) == "STOP":
                    raise InterruptedError("Stop request received.")

        del llama_model, classifier, llama_tokenizer; gc.collect(); torch.cuda.empty_cache()
        self._log("Phase 2 Complete. Llama and Classifier released from memory.")
        return all_preds, all_probas

    def run(self, input_file_path: str, mode: str, internal_batch_size: int = 32):
        from utils.resource_monitor import ResourceMonitor
        from utils.log_visualizer import LogVisualizer

        self.run_start_time = time.time()
        run_type = 'Testing' if mode == 'testing' else 'Inference'
        self.run_id = self.db.create_new_run(run_type, self.model_name, os.path.basename(input_file_path), {"mode": mode, "batch_size": internal_batch_size})
        if not self.run_id: raise RuntimeError("Failed to create new run in the database.")
        
        report_dir = REPORTS_DIR / self.run_id
        report_dir.mkdir(exist_ok=True)
        visualizer = LogVisualizer(plot_dir=report_dir)
        monitor = ResourceMonitor()
        monitor.start()
        
        temp_embedding_dir = tempfile.mkdtemp()
        final_status, results = 'FAILED', None
        
        try:
            dataset = LogDataset(input_file_path)
            if mode == 'testing' and (dataset.labels == -1).all():
                raise ValueError("Testing mode requires a 'Label' column, but it was missing or all values were invalid.")

            num_batches = self._run_bert_phase(dataset, temp_embedding_dir, internal_batch_size)
            all_preds, all_probas = self._run_llama_phase(dataset, temp_embedding_dir, num_batches)
            
            total_run_time, total_records = time.time() - self.run_start_time, len(dataset)
            time_per_record_ms = (total_run_time / total_records) * 1000 if total_records > 0 else 0

            perf_metrics = { "overall": { "total_run_time_sec": total_run_time, "time_per_record_ms": time_per_record_ms } }
            if mode == 'testing':
                gt_labels = dataset.get_all_labels()
                preds_numeric, probas_numeric, gt_numeric = safe_np_array(all_preds), safe_np_array(all_probas), gt_labels
                p, r, f1, _ = precision_recall_fscore_support(gt_numeric, preds_numeric, average='binary', pos_label=1, zero_division=0)
                perf_metrics["overall"].update({ "accuracy": accuracy_score(gt_numeric, preds_numeric), "precision": p, "recall": r, "f1_score": f1 })
                visualizer.plot_confusion_matrix(confusion_matrix(gt_numeric, preds_numeric), ['Normal', 'Anomalous'])
                visualizer.plot_roc_curve(gt_numeric, probas_numeric)
                visualizer.plot_overall_metrics({k: v for k, v in perf_metrics['overall'].items() if k in ['accuracy', 'precision', 'recall', 'f1_score']})
            else:
                df = pd.read_csv(input_file_path); df['Prediction'] = ["Anomalous" if p == 1 else "Normal" for p in all_preds]; df['Confidence'] = all_probas
                output_csv_path = report_dir / f"inference_results_{self.run_id}.csv"; df.to_csv(output_csv_path, index=False)
                results = str(output_csv_path)
                perf_metrics["overall"]["total_rows"] = total_records
            
            self.db.save_performance_metrics(self.run_id, perf_metrics)
            final_status = 'COMPLETED'
        except InterruptedError:
            final_status = 'ABORTED'; self._log(f"Run {self.run_id} aborted by user.")
        except Exception as e:
            final_status = 'FAILED'; error_msg = f"CRITICAL ERROR in run {self.run_id}: {e}"; print(error_msg); self._log(error_msg)
            self.callback({"error": str(e)})
        finally:
            if os.path.exists(temp_embedding_dir): shutil.rmtree(temp_embedding_dir)
            resource_metrics = monitor.stop()
            if self.run_id:
                self.db.save_resource_metrics(self.run_id, resource_metrics)
                visualizer.plot_resource_usage(resource_metrics)
                self.db.update_run_status(self.run_id, final_status, str(report_dir) if final_status == 'COMPLETED' else None)
            
            self._log(f"Run {self.run_id} finished with status: {final_status}")
            self.callback({"status": final_status, "done": True, "result": results})