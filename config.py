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

for dir_path in [DATA_DIR, MODELS_DIR, REPORTS_DIR, EMBEDDING_CACHE_DIR, TEMP_MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

DEFAULT_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLAMA_MODEL = "meta-llama/Meta-Llama-3.2-1B"

BASE_HYPERPARAMETERS = {
    "n_epochs_phase_adapters": 5,
    "lr_phase_adapters": 5e-5,
    "n_epochs_phase_full": 15,
    "lr_phase_full": 2e-5,
    "lora_r": 64,
    "batch_size": 128,
    "micro_batch_size": 32,
    "max_content_len": 100,
    "max_seq_len": 128,
    "early_stopping_patience": 3,
    "early_stopping_metric": "f1_score",
    "early_stopping_min_delta": 0.005,
    "dataloader_num_workers": 4,
}

DATASET_HYPERPARAMETERS = {
    "BGL": {
        "min_less_portion": 0.3,
        "max_seq_len": 100,
    },
    "Liberty": {
        "min_less_portion": 0.3,
        "max_seq_len": 100,
    },
    "HDFS": {
        "min_less_portion": 0.3,
        "max_seq_len": 128,
        "n_epochs_phase_adapters": 3,
        "n_epochs_phase_full": 10,
    },
    "Thunderbird": {
        "min_less_portion": 0.3,
        "max_seq_len": 100,
        "early_stopping_patience": 2,
    },
    "default": {
        "min_less_portion": 0.5,
    }
}

def get_hyperparameters(dataset_name: str) -> dict:
    if dataset_name not in DATASET_HYPERPARAMETERS:
        dataset_name = "default"
        
    hp = BASE_HYPERPARAMETERS.copy()
    hp.update(DATASET_HYPERPARAMETERS[dataset_name])
    return hp