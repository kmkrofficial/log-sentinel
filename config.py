from pathlib import Path
import psutil

ROOT_DIR = Path(__file__).resolve().parent

DATA_CACHE_DIR = ROOT_DIR / 'logsentinel_data'
DATA_DIR = ROOT_DIR / 'datasets'
MODELS_DIR = ROOT_DIR / 'models'
EXECUTIONS_DIR = ROOT_DIR / 'executions'
UTILS_DIR = ROOT_DIR / 'utils'

TEMP_MODELS_DIR = DATA_CACHE_DIR / 'temp_models'
DB_PATH = ROOT_DIR / 'logsentinel.db'

for dir_path in [DATA_DIR, MODELS_DIR, EXECUTIONS_DIR, DATA_CACHE_DIR, TEMP_MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

DEFAULT_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLAMA_MODEL = "meta-llama/Llama-3.2-1B"

def get_optimal_workers():
    try:
        cpu_count = psutil.cpu_count(logical=True)
        # Use half of the available cores, up to a reasonable limit
        return max(4, min(cpu_count // 2, 16))
    except Exception:
        return 4

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
    "dataloader_num_workers": get_optimal_workers(),
}

DATASET_HYPERPARAMETERS = {
    "BGL": {
        "min_less_portion": 0.3,
        "max_seq_len": 100,
    },
    "Liberty": {
        "min_less_portion": 0.5,
        "max_seq_len": 100,
        "lr_phase_full": 8e-6
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
    specific_hp = DATASET_HYPERPARAMETERS.get(dataset_name, DATASET_HYPERPARAMETERS["default"])
        
    hp = BASE_HYPERPARAMETERS.copy()
    hp.update(specific_hp)
    return hp