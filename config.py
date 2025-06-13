from pathlib import Path

# Project Root Directory
ROOT_DIR = Path(__file__).resolve().parent

# Core Directories
DATA_DIR = ROOT_DIR / 'datasets'
MODELS_DIR = ROOT_DIR / 'models'
REPORTS_DIR = ROOT_DIR / 'reports'
UTILS_DIR = ROOT_DIR / 'utils'

# Database Path
DB_PATH = ROOT_DIR / 'logsentinel.db'

# Default Model and Tokenizer Paths
DEFAULT_BERT_PATH = MODELS_DIR / 'bert-base-uncased'

# Ensure core directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# Default training & model configurations
DEFAULT_HYPERPARAMETERS = {
    "n_epochs_phase1": 2,
    "n_epochs_phase2": 2,
    "n_epochs_phase3": 2,
    "n_epochs_phase4": 2,
    "lr_phase1": 1e-4,
    "lr_phase2": 5e-4,
    "lr_phase3": 7e-5,
    "lr_phase4": 1e-5,
    "batch_size": 8,
    "micro_batch_size": 4,
    "max_content_len": 100,
    "max_seq_len": 128,
    "min_less_portion": 0.5,
}