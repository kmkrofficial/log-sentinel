from pathlib import Path
import sys


BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlcore.config import (  # noqa: E402
    DATA_CACHE_DIR,
    DATA_DIR,
    MODELS_DIR,
    EXECUTIONS_DIR,
    UTILS_DIR,
    TEMP_MODELS_DIR,
    DEFAULT_ENCODER_MODEL,
    DEFAULT_LLAMA_MODEL,
    BASE_HYPERPARAMETERS,
    DATASET_HYPERPARAMETERS,
    get_hyperparameters,
    get_optimal_workers,
)


DB_PATH = BACKEND_ROOT / 'logsentinel.db'