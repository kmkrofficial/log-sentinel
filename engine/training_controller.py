import os
import gc
import torch
import time
import shutil
import pandas as pd
import traceback
import platform
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.utils.data import TensorDataset, DataLoader

from config import REPORTS_DIR, DEFAULT_BERT_PATH
from utils.database_manager import DatabaseManager
from utils.data_loader import LogDataset, replace_patterns
from utils.resource_monitor import ResourceMonitor
from utils.log_visualizer import LogVisualizer
from logsentinel_model import LogSentinelModel
from utils.embedding_cacher import EmbeddingCacher
from utils.helpers import merge_data
from engine.phase_manager import train_phase, evaluate_and_visualize
from engine.data_utils import BalancedSampler
from prepareData.tensorize_embeddings import tensorize_dataset

torch.backends.cuda.matmul.allow_tf32 = True

class TrainingController:
    def __init__(self, model_name, dataset_name, hyperparameters, db_manager, callback=None, use_cached_embeddings=True, is_test_run=False, test_run_percentage=0.3):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.hp = hyperparameters
        self.db = db_manager
        self.callback = callback or (lambda *args: 'CONTINUE')
        self.run_id = None
        self.model = None
        self.visualizer = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.run_start_time = 0
        self.use_cached_embeddings = use_cached_embeddings
        self.is_test_run = is_test_run
        self.test_run_percentage = test_run_percentage
        self.batch_losses = []
        
        if self.is_test_run:
            self.use_cached_embeddings = False
            self._log(f"--- QUICK TEST RUN MODE ACTIVATED ({self.test_run_percentage*100:.0f}% data): Caching is enabled for this test run. ---")

    def _log(self, message):
        print(message)
        if self.callback: self.callback({"log": message})

    def _generate_nickname(self):
        model_name_short = self.model_name.split('/')[-1]
        base_nickname = f"{model_name_short}-{self.dataset_name}"
        if self.is_test_run:
            base_nickname += f"-{int(self.test_run_percentage * 100)}pct_TEST"
        
        existing_runs = self.db.get_runs_by_nickname_prefix(base_nickname)
        count = len(existing_runs)
        
        if count == 0:
            return base_nickname
        else:
            return f"{base_nickname}_{count + 1}"

    def _initialize_run(self):
        if self.is_test_run:
            self.hp['is_test_run'] = True
            self.hp['test_run_percentage'] = self.test_run_percentage
        
        nickname = self._generate_nickname()

        self.run_id = self.db.create_new_run('Training', self.model_name, self.dataset_name, self.hp, nickname)
        if self.run_id:
            self._log(f"Created new training run with ID: {self.run_id} (Nickname: {nickname})")
            self.report_dir = REPORTS_DIR / str(self.run_id)
            self.report_dir.mkdir(exist_ok=True)
            self.visualizer = LogVisualizer(plot_dir=self.report_dir)
        return self.run_id is not None

    def _cleanup(self, model_to_clean=None):
        target = model_to_clean if model_to_clean else self.model
        if target:
            self._log(f"Cleaning up model: {type(target).__name__}...")
            del target
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        self._log("Cleanup complete.")

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _get_or_create_embeddings(self, dataset_path, encoder_path):
        cacher = EmbeddingCacher(
            encoder_name=encoder_path.name,
            dataset_path=dataset_path,
            is_test_run=self.is_test_run,
            test_run_percentage=self.test_run_percentage if self.is_test_run else None
        )

        tensorized_path = cacher.cache_file_path.with_suffix('.tensor.pt')
        
        use_cache = self.use_cached_embeddings or self.is_test_run

        if tensorized_path.exists() and use_cache:
            self._log(f"Loading pre-tensorized dataset from {tensorized_path}")
            data = torch.load(tensorized_path)
            return TensorDataset(data['sequences'], data['labels'])

        embeddings, labels = cacher.load_embeddings()
        if not (embeddings and labels is not None and use_cache):
            self._log(f"No valid cache for {dataset_path.name}. Generating new embeddings...")
            
            df = pd.read_csv(dataset_path)
            if self.is_test_run:
                df = df.sample(frac=1, random_state=42).reset_index(drop=True)
                subset_size = int(self.test_run_percentage * len(df))
                df = df.head(subset_size)
            
            source_dataset = LogDataset(dataframe=df)
            
            encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_path)
            encoder_model = AutoModel.from_pretrained(encoder_path).to(self.device).eval()

            all_logs_flat, start_positions = merge_data(source_dataset.sequences)
            all_line_embeddings = []
            with torch.no_grad():
                for i in tqdm(range(0, len(all_logs_flat), 256), desc=f"Embedding {dataset_path.name}"):
                    batch_logs = all_logs_flat[i:i+256]
                    inputs = encoder_tokenizer(batch_logs, return_tensors="pt", padding=True, truncation=True, max_length=self.hp['max_content_len']).to(self.device)
                    model_output = encoder_model(**inputs)
                    line_embeddings = self._mean_pooling(model_output, inputs['attention_mask'])
                    all_line_embeddings.append(line_embeddings.cpu())
            
            all_line_embeddings_tensor = torch.cat(all_line_embeddings, dim=0)
            embeddings = list(torch.tensor_split(all_line_embeddings_tensor, start_positions[1:]))
            labels = source_dataset.get_all_labels()
            
            cacher.save_embeddings(embeddings, labels)
            self._cleanup(model_to_clean=encoder_model)
            del encoder_tokenizer

        tensorize_dataset(cacher.cache_file_path, self.hp['max_seq_len'])
        
        if tensorized_path.exists():
            data = torch.load(tensorized_path)
            return TensorDataset(data['sequences'], data['labels'])
        else:
            raise RuntimeError(f"Failed to create tensorized dataset for {dataset_path.name}")

    def run(self):
        monitor = ResourceMonitor()
        monitor.start()
        final_status = 'FAILED'
        try:
            if not self._initialize_run():
                raise RuntimeError("Failed to create a new run record.")
            self.run_start_time = time.time()
            
            encoder_path = DEFAULT_BERT_PATH
            encoder_config = AutoConfig.from_pretrained(encoder_path)
            self.hp['encoder_hidden_size'] = encoder_config.hidden_size

            self._log("Preparing datasets (on-demand)...")
            train_dataset_path = Path("datasets") / self.dataset_name / 'train.csv'
            val_dataset_path = Path("datasets") / self.dataset_name / 'validation.csv'
            test_dataset_path = Path("datasets") / self.dataset_name / 'test.csv'

            train_dataset = self._get_or_create_embeddings(train_dataset_path, encoder_path)
            validation_dataset = self._get_or_create_embeddings(val_dataset_path, encoder_path) if val_dataset_path.exists() else None
            test_dataset = self._get_or_create_embeddings(test_dataset_path, encoder_path) if test_dataset_path.exists() else None

            if not train_dataset:
                raise RuntimeError("Training dataset could not be loaded or created.")

            ft_path = None
            
            compile_model = platform.system() == "Linux"
            if compile_model:
                self._log("Linux detected. Enabling torch.compile() for optimized performance.")

            # Initialize progress state
            progress_state = {
                'global_step': 0,
                'total_steps': 0,
                'phase_steps': 0,
                'phase_total_steps': 0,
                'phase_start_time': 0
            }

            # Calculate total steps for the progress bar
            for phase_epochs in [self.hp.get('n_epochs_phase_adapters', 0), self.hp.get('n_epochs_phase_full', 0)]:
                if phase_epochs > 0:
                    sampler = BalancedSampler(train_dataset.tensors[1].numpy(), self.hp.get('min_less_portion', 0.5))
                    loader = DataLoader(train_dataset, batch_size=self.hp['micro_batch_size'], sampler=sampler)
                    progress_state['total_steps'] += len(loader) * phase_epochs

            # Phase 1
            self.model = LogSentinelModel(self.model_name, self.hp['encoder_hidden_size'], self.hp, ft_path, True, self.device)
            if compile_model: self.model = torch.compile(self.model, mode="max-autotune")
            self.model.set_train_projector_and_classifier()
            success, ft_path = train_phase(self, "Adapters", self.hp.get('n_epochs_phase_adapters', 0), self.hp.get('lr_phase_adapters', 5e-5), train_dataset, validation_dataset, progress_state)

            # Phase 2
            if success:
                self._cleanup(self.model)
                self.model = LogSentinelModel(self.model_name, self.hp['encoder_hidden_size'], self.hp, ft_path, True, self.device)
                if compile_model: self.model = torch.compile(self.model, mode="max-autotune")
                self.model.set_finetuning_all()
                _, ft_path = train_phase(self, "Full_Fine_Tuning", self.hp.get('n_epochs_phase_full', 0), self.hp.get('lr_phase_full', 2e-5), train_dataset, validation_dataset, progress_state)
                self._cleanup(self.model)

            # Evaluation
            self._log("\n>>>> CONFIGURING MODEL FOR FINAL EVALUATION <<<<")
            self.model = LogSentinelModel(self.model_name, self.hp['encoder_hidden_size'], self.hp, ft_path, False, self.device)
            if compile_model: self.model = torch.compile(self.model, mode="max-autotune")
            
            final_metrics = {}
            if validation_dataset:
                final_metrics.update(evaluate_and_visualize(self, validation_dataset, "validation"))
            if test_dataset:
                final_metrics.update(evaluate_and_visualize(self, test_dataset, "test"))
            
            final_metrics['overall'] = { "total_run_time_sec": time.time() - self.run_start_time }
            self.visualizer.plot_training_loss(self.batch_losses)
            self.db.save_performance_metrics(self.run_id, final_metrics)

            if ft_path and os.path.exists(ft_path):
                final_model_path = self.report_dir / 'final_model'
                shutil.copytree(ft_path, final_model_path, dirs_exist_ok=True)
                self._log(f"Consolidated best model saved to: {final_model_path}")
            
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
                self.db.update_run_status(self.run_id, final_status, str(self.report_dir) if final_status == 'COMPLETED' else None)
            if self.model: self._cleanup(self.model)
            if self.callback: self.callback({"status": final_status, "done": True})

