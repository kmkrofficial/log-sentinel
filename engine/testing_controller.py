import os
import gc
import torch
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from config import REPORTS_DIR, DEFAULT_BERT_PATH
from utils.database_manager import DatabaseManager
from utils.data_loader import LogDataset
from utils.resource_monitor import ResourceMonitor
from utils.log_visualizer import LogVisualizer
from logsentinel_model import LogSentinelModel

class TestingController:
    def __init__(self, trained_run_id, test_filename, db_manager, callback=None):
        self.trained_run_id = trained_run_id; self.test_filename = test_filename; self.db = db_manager
        self.callback = callback or (lambda *args: 'CONTINUE'); self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.run_id, self.model, self.test_dataset = None, None, None

    def _log(self, message): print(message); self.callback({"log": message})
    def _initialize_run(self):
        trained_model_details = self.db.get_run_details(self.trained_run_id)
        if not trained_model_details: raise ValueError(f"Could not find details for trained run ID: {self.trained_run_id}")
        model_name, dataset_name = trained_model_details['run_info']['model_name'], f"Test on {os.path.basename(self.test_filename)}"
        self.run_id = self.db.create_new_run('Testing', model_name, dataset_name, None)
        if self.run_id: self._log(f"Created new test run with ID: {self.run_id}"); self.report_dir = REPORTS_DIR / self.run_id; self.report_dir.mkdir(exist_ok=True); self.visualizer = LogVisualizer(plot_dir=self.report_dir)
        return self.run_id is not None
    def cleanup(self):
        self._log("Cleaning up testing resources..."); del self.model, self.test_dataset; self.model, self.test_dataset = None, None
        gc.collect();
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        self._log("Testing cleanup complete.")
    def run(self):
        monitor = ResourceMonitor(); monitor.start(); final_status = 'FAILED'; run_start_time = time.time()
        try:
            if not self._initialize_run(): raise RuntimeError("Failed to create a new test run record in the database.")
            model_path = str(REPORTS_DIR / self.trained_run_id); model_name = self.db.get_run_details(self.trained_run_id)['run_info']['model_name']
            self._log(f"Loading test dataset from {self.test_filename}..."); self.test_dataset = LogDataset(self.test_filename)
            self._log(f"Loading model for testing..."); self.model = LogSentinelModel(DEFAULT_BERT_PATH, model_name, model_path, False, self.device)
            self.model.eval(); all_preds, all_probas, gt_labels = [], [], self.test_dataset.get_all_labels()
            total_rows, batch_size = len(self.test_dataset), 16
            self._log(f"Starting evaluation on {total_rows} test samples...")
            with torch.inference_mode(): # Use inference_mode for speed
                for i in tqdm(range(0, total_rows, batch_size), desc="Testing"):
                    end_idx = min(i + batch_size, total_rows); seqs, _ = self.test_dataset.get_batch(list(range(i, end_idx)))
                    logits = self.model(seqs); probas = torch.softmax(logits, dim=-1); all_preds.extend(torch.argmax(probas, dim=-1).cpu().numpy()); all_probas.extend(probas[:, 1].cpu().numpy())
                    progress, elapsed = (i + len(seqs)) / total_rows, time.time() - run_start_time; etc = (elapsed / progress) * (1 - progress) if progress > 0 else 0
                    if self.callback({"progress": progress, "rows_processed": i + len(seqs), "etc": etc}) == 'STOP': final_status = 'ABORTED'; raise InterruptedError("Test run aborted by user.")
            self._log("Evaluation complete. Calculating metrics..."); valid_indices = (np.array(all_preds) != -1)
            preds_numeric, probas_numeric, gt_numeric = np.array(all_preds)[valid_indices], np.array(all_probas)[valid_indices], gt_labels[valid_indices]
            p, r, f1, _ = precision_recall_fscore_support(gt_numeric, preds_numeric, average='binary', pos_label=1, zero_division=0)
            p_det, r_det, f1_det, s_det = precision_recall_fscore_support(gt_numeric, preds_numeric, labels=[0, 1], zero_division=0)
            total_run_time_sec = time.time() - run_start_time; time_per_record_ms = (total_run_time_sec / len(gt_numeric)) * 1000 if len(gt_numeric) > 0 else 0
            perf_metrics = {"overall": {"accuracy": accuracy_score(gt_numeric, preds_numeric), "precision": p, "recall": r, "f1_score": f1, "total_run_time_sec": total_run_time_sec, "time_per_record_ms": time_per_record_ms}, "per_class": {"normal": {"precision": p_det[0], "recall": r_det[0], "f1": f1_det[0], "support": int(s_det[0])}, "anomalous": {"precision": p_det[1], "recall": r_det[1], "f1": f1_det[1], "support": int(s_det[1])}}, "confusion_matrix": confusion_matrix(gt_numeric, preds_numeric, labels=[0, 1]).tolist()}
            self.db.save_performance_metrics(self.run_id, perf_metrics); self.visualizer.plot_confusion_matrix(np.array(perf_metrics['confusion_matrix']), ['Normal', 'Anomalous'])
            metrics_for_plot = {k: v for k, v in perf_metrics['overall'].items() if k in ['accuracy', 'precision', 'recall', 'f1_score']}
            self.visualizer.plot_overall_metrics(metrics_for_plot); self.visualizer.plot_roc_curve(gt_numeric, probas_numeric); final_status = 'COMPLETED'
        except InterruptedError as e: self._log(f"\n{e}")
        except Exception as e: error_msg = f"CRITICAL ERROR in test run {self.run_id}: {e}"; self._log(error_msg); self.callback({"error": str(e)})
        finally:
            self._log("Stopping resource monitor and saving metrics..."); resource_metrics = monitor.stop()
            if self.run_id:
                self.db.save_resource_metrics(self.run_id, resource_metrics)
                if hasattr(self, 'visualizer'): self._log("Generating resource usage plots..."); self.visualizer.plot_resource_usage(resource_metrics)
                self.db.update_run_status(self.run_id, final_status, str(self.report_dir) if final_status == 'COMPLETED' else None); self._log(f"Test run {self.run_id} finished with status: {final_status}")
            self.cleanup(); self.callback({"status": final_status, "done": True})