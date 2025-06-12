import os
import gc
import random
import numpy as np
import torch
import time
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
from transformers import get_linear_schedule_with_warmup

from config import REPORTS_DIR, DEFAULT_BERT_PATH
from utils.database_manager import DatabaseManager
from utils.data_loader import LogDataset
from utils.resource_monitor import ResourceMonitor
from utils.log_visualizer import LogVisualizer
from logsentinel_model import LogSentinelModel

class TrainingController:
    def __init__(self, model_name, dataset_name, hyperparameters, db_manager, callback=None):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.hp = hyperparameters
        self.db = db_manager
        self.callback = callback or (lambda *args, **kwargs: 'CONTINUE')
        self.run_id = None
        self.model = None
        self.train_dataset = None
        self.test_dataset = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.global_step_count = 0
        self.total_training_steps = 0
        self.run_start_time = 0

    def _initialize_run(self):
        self.run_id = self.db.create_new_run(self.model_name, self.dataset_name, self.hp)
        if self.run_id:
            self.report_dir = REPORTS_DIR / self.run_id
            self.report_dir.mkdir(exist_ok=True)
            self.visualizer = LogVisualizer(plot_dir=self.report_dir)
        return self.run_id is not None

    def _train_phase(self, phase_name, n_epochs, lr):
        if not n_epochs > 0:
            print(f"Skipping Phase: {phase_name}")
            return True

        print(f"\n--- Starting Training Phase: {phase_name} ---")
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            print("Warning: No trainable parameters for this phase. Skipping.")
            return True

        optimizer = torch.optim.AdamW(trainable_params, lr=lr)
        
        indexes = list(range(len(self.train_dataset)))
        if self.train_dataset.num_less > 0 and self.train_dataset.num_less / len(self.train_dataset) < self.hp.get('min_less_portion', 0.5):
            add_num = int((self.hp['min_less_portion'] * self.train_dataset.num_majority) / (1 - self.hp['min_less_portion'])) - self.train_dataset.num_less
            if add_num > 0:
                oversampled_indices = np.random.choice(self.train_dataset.less_indexes, add_num, replace=True).tolist()
                indexes.extend(oversampled_indices)
        
        micro_batch_size = self.hp['micro_batch_size']
        grad_accum_steps = self.hp['batch_size'] // micro_batch_size
        
        for epoch in range(int(n_epochs)):
            epoch_str = f"Epoch {epoch + 1}/{int(n_epochs)} ({phase_name})"
            print(f"\n--- {epoch_str} ---")
            random.shuffle(indexes)
            optimizer.zero_grad()
            
            num_batches = len(indexes) // micro_batch_size if micro_batch_size > 0 else 0
            
            for i_th, start_idx in enumerate(range(0, len(indexes), micro_batch_size)):
                self.global_step_count += 1
                end_idx = min(start_idx + micro_batch_size, len(indexes))
                batch_indexes = indexes[start_idx:end_idx]
                if not batch_indexes: continue

                seqs, labels = self.train_dataset.get_batch(batch_indexes)
                logits, int_labels = self.model.train_helper(seqs, labels)
                
                if logits.shape[0] == 0: continue

                loss = criterion(logits, int_labels)
                loss_val = loss.item()
                loss = loss / grad_accum_steps
                loss.backward()

                if (i_th + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                elapsed_time = time.time() - self.run_start_time
                avg_time_per_step = elapsed_time / self.global_step_count if self.global_step_count > 0 else 0
                remaining_steps = self.total_training_steps - self.global_step_count
                etc_seconds = remaining_steps * avg_time_per_step
                
                status_update = {"epoch": epoch_str, "progress": self.global_step_count / self.total_training_steps, "loss": loss_val, "etc": etc_seconds}
                if self.callback(status_update) == 'STOP':
                    print("Stop request received from UI. Aborting training.")
                    return False
        
        torch.cuda.empty_cache()
        gc.collect()
        return True

    def _evaluate(self):
        # ... (This method remains unchanged) ...
        print("\n--- Starting Final Evaluation ---")
        self.model.eval()
        all_preds, gt_labels = [], self.test_dataset.get_all_labels()
        
        with torch.no_grad():
            for i in tqdm(range(0, len(self.test_dataset), self.hp['batch_size']), desc="Evaluating"):
                end_idx = min(i + self.hp['batch_size'], len(self.test_dataset))
                seqs, _ = self.test_dataset.get_batch(list(range(i, end_idx)))
                logits = self.model(seqs)
                all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        
        valid_indices = (np.array(all_preds) != -1)
        preds_numeric, gt_numeric = np.array(all_preds)[valid_indices], gt_labels[valid_indices]
        p, r, f1, _ = precision_recall_fscore_support(gt_numeric, preds_numeric, average='binary', pos_label=1, zero_division=0)
        p_det, r_det, f1_det, s_det = precision_recall_fscore_support(gt_numeric, preds_numeric, labels=[0, 1], zero_division=0)
        
        return {
            "overall": {"accuracy": accuracy_score(gt_numeric, preds_numeric), "precision": p, "recall": r, "f1_score": f1},
            "per_class": {"normal": {"precision": p_det[0], "recall": r_det[0], "f1": f1_det[0], "support": int(s_det[0])},
                          "anomalous": {"precision": p_det[1], "recall": r_det[1], "f1": f1_det[1], "support": int(s_det[1])}},
            "confusion_matrix": confusion_matrix(gt_numeric, preds_numeric, labels=[0, 1]).tolist()
        }

    def run(self):
        if not self._initialize_run():
            self.callback({"error": "Failed to create run in database."})
            return

        monitor = ResourceMonitor()
        monitor.start()
        final_status = 'FAILED'
        self.run_start_time = time.time()

        try:
            self.train_dataset = LogDataset(os.path.join("datasets", self.dataset_name, 'train.csv'))
            self.test_dataset = LogDataset(os.path.join("datasets", self.dataset_name, 'test.csv'))
            self.model = LogSentinelModel(DEFAULT_BERT_PATH, self.model_name, None, True, self.device, self.hp['max_content_len'], self.hp['max_seq_len'])
            
            batches_per_epoch = len(self.train_dataset) // self.hp['micro_batch_size']
            self.total_training_steps = sum([self.hp.get(f'n_epochs_phase{i+1}', 0) * batches_per_epoch for i in range(4)])
            
            training_phases = [
                ("Projector", self.model.set_train_only_projector, self.hp['n_epochs_phase1'], self.hp['lr_phase1']),
                ("Classifier", self.model.set_train_only_classifier, self.hp['n_epochs_phase2'], self.hp['lr_phase2']),
                ("Projector+Classifier", self.model.set_train_projector_and_classifier, self.hp['n_epochs_phase3'], self.hp['lr_phase3']),
                ("Fine-tuning All", self.model.set_finetuning_all, self.hp['n_epochs_phase4'], self.hp['lr_phase4'])
            ]

            all_phases_completed = True
            for name, setup_func, epochs, lr in training_phases:
                setup_func()
                if not self._train_phase(name, epochs, lr):
                    all_phases_completed = False
                    final_status = 'ABORTED'
                    break
            
            if all_phases_completed:
                final_model_path = REPORTS_DIR / self.run_id / 'final_model'
                self.model.save_ft_model(str(final_model_path))
                perf_metrics = self._evaluate()
                self.db.save_performance_metrics(self.run_id, perf_metrics)
                self.visualizer.plot_confusion_matrix(np.array(perf_metrics['confusion_matrix']), ['Normal', 'Anomalous'])
                self.visualizer.plot_overall_metrics(perf_metrics['overall'])
                final_status = 'COMPLETED'

        except Exception as e:
            print(f"\nCRITICAL ERROR in run {self.run_id}: {e}")
            final_status = 'FAILED'
        finally:
            resource_metrics = monitor.stop()
            self.db.save_resource_metrics(self.run_id, resource_metrics)
            if hasattr(self, 'visualizer'): self.visualizer.plot_resource_usage(resource_metrics)
            self.db.update_run_status(self.run_id, final_status, str(self.report_dir) if final_status == 'COMPLETED' else None)
            self.callback({"status": final_status, "done": True})