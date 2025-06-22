from pathlib import Path

# Project Root Directory
ROOT_DIR = Path(__file__).resolve().parent

# Centralized directory for all generated data
DATA_CACHE_DIR = ROOT_DIR / 'logsentinel_data'

# Core Directories
DATA_DIR = ROOT_DIR / 'datasets'
MODELS_DIR = ROOT_DIR / 'models'
REPORTS_DIR = ROOT_DIR / 'reports'
UTILS_DIR = ROOT_DIR / 'utils'

# Specific paths for generated data
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

# --- HYPERPARAMETERS TUNED FOR PEAK PERFORMANCE (r=64) ---
DEFAULT_HYPERPARAMETERS = {
    # Phase 1: Train the input/output adapters (Projector + Classifier)
    "n_epochs_phase_adapters": 5,
    "lr_phase_adapters": 5e-5,
    
    # Phase 2: Fine-tune the entire pipeline (LoRA + Projector + Classifier)
    # --- FIX: Increased epoch budget to allow the more powerful r=64 model to fully converge ---
    "n_epochs_phase_full": 15,
    "lr_phase_full": 2e-5,
    
    "batch_size": 8,
    "micro_batch_size": 4,
    "max_content_len": 100,
    "max_seq_len": 128,
    "min_less_portion": 0.5,
    "early_stopping_patience": 2,
    "early_stopping_metric": "f1_score",
    "early_stopping_min_delta": 0.01
}