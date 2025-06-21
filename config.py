from pathlib import Path

# Project Root Directory
ROOT_DIR = Path(__file__).resolve().parent

# --- NEW: Centralized directory for all generated data ---
DATA_CACHE_DIR = ROOT_DIR / 'logsentinel_data'

# Core Directories
DATA_DIR = ROOT_DIR / 'datasets'
MODELS_DIR = ROOT_DIR / 'models'
REPORTS_DIR = ROOT_DIR / 'reports'
UTILS_DIR = ROOT_DIR / 'utils'

# --- NEW: Specific paths for generated data ---
EMBEDDING_CACHE_DIR = DATA_CACHE_DIR / 'embedding_cache'
TEMP_MODELS_DIR = DATA_CACHE_DIR / 'temp_models'

# Database Path
DB_PATH = ROOT_DIR / 'logsentinel.db'

# Default Model and Tokenizer Paths
DEFAULT_BERT_PATH = MODELS_DIR / 'sentence-transformers/all-MiniLM-L6-v2'

# Ensure core directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TEMP_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- FINAL, OPTIMIZED HYPERPARAMETERS ---
# Based on analysis of successful runs with Early Stopping.
# These values provide a fast and effective training cycle, prioritizing generalization.
DEFAULT_HYPERPARAMETERS = {
    "n_epochs_phase1": 2,
    "n_epochs_phase2": 1,
    "n_epochs_phase3": 3,
    "n_epochs_phase4": 8,                  # A reasonable budget for the final phase
    "lr_phase1": 8e-05,
    "lr_phase2": 5e-05,
    "lr_phase3": 3e-05,
    "lr_phase4": 2e-05,
    "batch_size": 8,
    "micro_batch_size": 4,
    "max_content_len": 100,
    "max_seq_len": 128,
    "min_less_portion": 0.5,
    "early_stopping_patience": 2,
    "early_stopping_metric": "f1_score",
    "early_stopping_min_delta": 0.01       # Assertive delta to prevent overfitting
}