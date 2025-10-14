from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

DATA_CACHE_DIR = ROOT_DIR / 'logsentinel_data'

DATA_DIR = ROOT_DIR / 'datasets'
MODELS_DIR = ROOT_DIR / 'models'
REPORTS_DIR = ROOT_DIR / 'reports'
UTILS_DIR = ROOT_DIR / 'utils'

EMBEDDING_CACHE_DIR = DATA_CACHE_DIR / 'embedding_cache'
TEMP_MODELS_DIR = DATA_CACHE_DIR / 'temp_models'

DB_PATH = ROOT_DIR / 'logsentinel.db'

DEFAULT_BERT_PATH = MODELS_DIR / 'sentence-transformers/all-MiniLM-L6-v2'

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TEMP_MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_HYPERPARAMETERS = {
    "n_epochs_phase_adapters": 5,
    "lr_phase_adapters": 5e-5,
    "n_epochs_phase_full": 15,
    "lr_phase_full": 2e-5,
    "lora_r": 64,
    "batch_size": 128,
    "micro_batch_size": 32,
    "max_content_len": 100,
    "max_seq_len": 128,
    "min_less_portion": 0.5,
    "early_stopping_patience": 2,
    "early_stopping_metric": "f1_score",
    "early_stopping_min_delta": 0.01
}