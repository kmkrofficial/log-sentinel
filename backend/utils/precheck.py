import csv
import importlib.util
import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import (
    DATA_DIR,
    DEFAULT_ENCODER_MODEL,
    DEFAULT_LLAMA_MODEL,
    EXECUTIONS_DIR,
    MODELS_DIR,
    get_hyperparameters,
)
from mlcore.setup_catalog import DATASET_SPECS
from mlcore.setup_manager import scan_setup_state
from mlcore.utils.runtime_compat import bitsandbytes_version, resolve_bitsandbytes_cuda_binary, torch_compile_readiness


REQUIRED_DATASET_COLUMNS = {"Content", "Label"}
VALID_DATASET_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
GIB = 1024**3


def _check(key: str, label: str, status: str, detail: str) -> dict[str, str]:
    return {"key": key, "label": label, "status": status, "detail": detail}


def _is_dataset_name_safe(dataset_name: str) -> bool:
    return bool(VALID_DATASET_NAME.fullmatch(dataset_name))


def _read_csv_contract(csv_path: Path, key: str, label: str, required: bool) -> dict[str, str]:
    if not csv_path.is_file():
        status = "failed" if required else "warning"
        requirement = "is required" if required else "is optional for training"
        return _check(key, label, status, f"{csv_path.name} {requirement} but was not found.")

    if csv_path.stat().st_size == 0:
        return _check(key, label, "failed", f"{csv_path.name} is empty.")

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as dataset_file:
            reader = csv.reader(dataset_file)
            headers = next(reader, [])
            first_record = next(reader, None)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        return _check(key, label, "failed", f"Could not read {csv_path.name}: {error}")

    missing_columns = sorted(REQUIRED_DATASET_COLUMNS - set(headers))
    if missing_columns:
        return _check(
            key,
            label,
            "failed",
            f"{csv_path.name} is missing required columns: {', '.join(missing_columns)}.",
        )
    if first_record is None:
        return _check(key, label, "failed", f"{csv_path.name} has headers but no data rows.")

    return _check(key, label, "passed", f"{csv_path.name} has Content and Label columns with data rows.")


def _count_dataset_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as dataset_file:
        return max(0, sum(1 for _ in dataset_file) - 1)


def _find_model_weight(model_dir: Path) -> Path | None:
    for candidate_name in ("model.safetensors", "pytorch_model.bin", "model.bin"):
        candidate_path = model_dir / candidate_name
        if candidate_path.is_file():
            return candidate_path
    return None


def _model_asset_check(model_name: str, key: str, label: str) -> dict[str, str]:
    model_dir = MODELS_DIR / model_name.split("/")[-1]
    config_path = model_dir / "config.json"
    weight_path = _find_model_weight(model_dir)

    if not model_dir.is_dir():
        return _check(key, label, "failed", f"Model directory was not found: {model_dir}.")
    if not config_path.is_file():
        return _check(key, label, "failed", f"Model config was not found: {config_path}.")
    if weight_path is None:
        return _check(key, label, "failed", f"No supported model weight file was found in {model_dir}.")

    return _check(key, label, "passed", f"Found {config_path.name} and {weight_path.name} in {model_dir.name}.")


def _check_runtime_dependencies() -> list[dict[str, str]]:
    required_modules = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "peft": "PEFT",
        "bitsandbytes": "bitsandbytes",
        "h5py": "HDF5 support",
        "sklearn": "scikit-learn",
    }
    missing_modules = [label for module_name, label in required_modules.items() if importlib.util.find_spec(module_name) is None]

    if missing_modules:
        dependency_check = _check(
            "python_dependencies",
            "Python dependencies",
            "failed",
            f"Missing required modules: {', '.join(missing_modules)}.",
        )
    else:
        dependency_check = _check(
            "python_dependencies",
            "Python dependencies",
            "passed",
            "PyTorch, Transformers, PEFT, bitsandbytes, HDF5, and scikit-learn are importable.",
        )

    return [dependency_check]


def _check_cuda_runtime() -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []

    try:
        import torch
    except Exception as error:
        return [_check("cuda_runtime", "CUDA runtime", "failed", f"PyTorch could not be imported: {error}")]

    if not torch.cuda.is_available():
        return [_check("cuda_runtime", "CUDA runtime", "failed", "CUDA is unavailable, but the model loader requires GPU execution.")]

    device = torch.cuda.get_device_properties(0)
    capability = f"{device.major}.{device.minor}"
    free_memory, total_memory = torch.cuda.mem_get_info(0)
    checks.append(
        _check(
            "cuda_runtime",
            "CUDA runtime",
            "passed",
            f"{device.name} is available with CUDA {torch.version.cuda}, compute capability {capability}, and {free_memory / GIB:.1f}/{total_memory / GIB:.1f} GiB free.",
        )
    )

    if device.major < 8:
        checks.append(
            _check(
                "flash_attention_hardware",
                "FlashAttention 3 hardware",
                "failed",
                f"Compute capability {capability} is below the FlashAttention 3 minimum of 8.0.",
            )
        )
    else:
        checks.append(
            _check(
                "flash_attention_hardware",
                "FlashAttention 3 hardware",
                "passed",
                f"Compute capability {capability} satisfies the FlashAttention 3 minimum of 8.0.",
            )
        )

    if torch.cuda.is_bf16_supported():
        checks.append(_check("bf16_support", "BF16 support", "passed", "The active GPU supports bfloat16 execution."))
    else:
        checks.append(_check("bf16_support", "BF16 support", "failed", "The active GPU does not report bfloat16 support."))

    return checks


def _check_flash_attention() -> dict[str, str]:
    try:
        from transformers.utils.import_utils import is_flash_attn_3_available

        if is_flash_attn_3_available():
            return _check(
                "flash_attention_3",
                "FlashAttention 3",
                "passed",
                "Transformers recognizes the installed FlashAttention 3 backend.",
            )
    except Exception as error:
        return _check("flash_attention_3", "FlashAttention 3", "failed", f"FlashAttention 3 check failed: {error}")

    return _check(
        "flash_attention_3",
        "FlashAttention 3",
        "failed",
        "Transformers cannot use FlashAttention 3. Install a compatible flash_attn_3 build.",
    )


def _check_bitsandbytes() -> dict[str, str]:
    try:
        import torch
    except Exception as error:
        return _check("bitsandbytes_cuda", "bitsandbytes CUDA binary", "failed", f"PyTorch could not be imported: {error}")

    installed_version = bitsandbytes_version()
    if installed_version is None:
        return _check("bitsandbytes_cuda", "bitsandbytes CUDA binary", "failed", "bitsandbytes is not installed.")

    binary = resolve_bitsandbytes_cuda_binary(torch.version.cuda)
    if binary is None:
        return _check(
            "bitsandbytes_cuda",
            "bitsandbytes CUDA binary",
            "failed",
            f"bitsandbytes {installed_version} is installed, but it has no compatible CUDA binary for PyTorch CUDA {torch.version.cuda}.",
        )

    if binary.source == "configured":
        return _check(
            "bitsandbytes_cuda",
            "bitsandbytes CUDA binary",
            "passed",
            f"bitsandbytes {installed_version} is using the configured {binary.path.name} binary.",
        )
    if binary.source == "compatible":
        return _check(
            "bitsandbytes_cuda",
            "bitsandbytes CUDA binary",
            "passed",
            f"bitsandbytes {installed_version} will use compatible {binary.path.name} for PyTorch CUDA {torch.version.cuda}.",
        )

    return _check(
        "bitsandbytes_cuda",
        "bitsandbytes CUDA binary",
        "passed",
        f"bitsandbytes {installed_version} has {binary.path.name} for PyTorch CUDA {torch.version.cuda}.",
    )


def _check_triton_runtime() -> dict[str, str]:
    triton_spec = importlib.util.find_spec("triton")
    if triton_spec is None:
        return _check(
            "triton_runtime",
            "Triton runtime",
            "failed",
            "Triton is not installed.",
        )

    return _check(
        "triton_runtime",
        "Triton runtime",
        "passed",
        "Triton is installed. See torch.compile acceleration for optional local compiler requirements.",
    )


def _check_torch_compile_acceleration() -> dict[str, str]:
    readiness = torch_compile_readiness()
    if not readiness.ready:
        return _check(
            "torch_compile_acceleration",
            "torch.compile acceleration",
            "warning",
            readiness.detail,
        )

    return _check(
        "torch_compile_acceleration",
        "torch.compile acceleration",
        "passed",
        readiness.detail,
    )


def build_environment_checks() -> list[dict[str, str]]:
    """Return model-runtime checks that are independent of a chosen dataset."""
    checks: list[dict[str, str]] = []
    checks.extend(_check_runtime_dependencies())
    checks.extend(_check_cuda_runtime())
    checks.append(_check_flash_attention())
    checks.append(_check_bitsandbytes())
    checks.append(_check_triton_runtime())
    checks.append(_check_torch_compile_acceleration())
    return checks


def _check_execution_storage(dataset_dir: Path, dataset_name: str, is_test_run: bool, test_run_percentage: float) -> list[dict[str, str]]:
    try:
        EXECUTIONS_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=EXECUTIONS_DIR, prefix=".precheck-", delete=True):
            pass
    except OSError as error:
        return [_check("execution_write_access", "Execution directory write access", "failed", f"Cannot write to {EXECUTIONS_DIR}: {error}")]

    train_path = dataset_dir / "train.csv"
    test_path = dataset_dir / "test.csv"
    if not train_path.is_file():
        return [_check("execution_storage", "Temporary embedding storage", "warning", "Storage estimate skipped because train.csv is unavailable.")]

    try:
        train_rows = _count_dataset_rows(train_path)
        test_rows = _count_dataset_rows(test_path) if test_path.is_file() else 0
        sampling_fraction = test_run_percentage if is_test_run else 1.0
        hyperparameters = get_hyperparameters(dataset_name)
        encoder_config = json.loads((MODELS_DIR / DEFAULT_ENCODER_MODEL.split("/")[-1] / "config.json").read_text(encoding="utf-8"))
        hidden_size = int(encoder_config["hidden_size"])
        embedding_bytes = int((train_rows + test_rows) * sampling_fraction * hyperparameters["max_seq_len"] * hidden_size * 4)
        required_bytes = max(2 * GIB, int(embedding_bytes * 1.3))
        free_bytes = shutil.disk_usage(EXECUTIONS_DIR).free
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        return [_check("execution_storage", "Temporary embedding storage", "warning", f"Could not estimate embedding storage: {error}")]

    if free_bytes < required_bytes:
        return [
            _check(
                "execution_storage",
                "Temporary embedding storage",
                "failed",
                f"Estimated {required_bytes / GIB:.1f} GiB is required for embeddings, but only {free_bytes / GIB:.1f} GiB is free in {EXECUTIONS_DIR}.",
            )
        ]

    return [
        _check(
            "execution_storage",
            "Temporary embedding storage",
            "passed",
            f"Estimated embedding requirement is {required_bytes / GIB:.1f} GiB; {free_bytes / GIB:.1f} GiB is free in {EXECUTIONS_DIR}.",
        ),
        _check("execution_write_access", "Execution directory write access", "passed", f"{EXECUTIONS_DIR} accepts temporary files."),
    ]


def _check_inference_checkpoint(model_run_path: str | None) -> dict[str, str]:
    if not model_run_path:
        return _check("inference_checkpoint", "Inference checkpoint", "failed", "A trained model run path is required for inference.")

    supplied_path = Path(model_run_path).expanduser()
    run_path = supplied_path.parent if supplied_path.name == "output_model" else supplied_path
    output_model_path = run_path / "output_model"
    required_paths = [
        output_model_path / "projector.pt",
        output_model_path / "classifier.pt",
        output_model_path / "Llama_ft" / "adapter_config.json",
    ]
    missing_paths = [str(path.relative_to(run_path)) for path in required_paths if not path.is_file()]

    if missing_paths:
        return _check(
            "inference_checkpoint",
            "Inference checkpoint",
            "failed",
            f"Missing required checkpoint files: {', '.join(missing_paths)}.",
        )

    return _check("inference_checkpoint", "Inference checkpoint", "passed", f"Found reusable model artifacts in {output_model_path}.")


def _check_setup_dataset_state(dataset_name: str) -> dict[str, str] | None:
    if dataset_name not in DATASET_SPECS:
        return None

    setup_state = scan_setup_state()
    dataset = next((item for item in setup_state["datasets"] if item["id"] == dataset_name), None)
    if dataset is None:
        return _check("setup_dataset_state", "Configuration setup", "failed", f"{dataset_name} is missing from the setup catalog.")
    if not dataset["raw"]["ready"]:
        return _check(
            "setup_dataset_state",
            "Configuration setup",
            "failed",
            f"Verified raw assets for {dataset_name} are not ready. Complete configuration setup first.",
        )
    if dataset.get("preparation_blocker"):
        return _check(
            "setup_dataset_state",
            "Configuration setup",
            "failed",
            dataset["preparation_blocker"],
        )
    return _check(
        "setup_dataset_state",
        "Configuration setup",
        "passed",
        f"Verified {dataset['raw']['layout']} raw assets are available for {dataset_name}.",
    )


def _check_complete_setup_state() -> dict[str, str]:
    setup_state = scan_setup_state()
    missing_datasets = [dataset["id"] for dataset in setup_state["datasets"] if not dataset["raw"]["ready"]]
    missing_models = [model["model_id"] for model in setup_state["models"] if not model["ready"]]

    missing_assets = [*missing_datasets, *missing_models]
    if missing_assets:
        return _check(
            "complete_configuration_setup",
            "Complete configuration setup",
            "failed",
            f"Complete Step 1 before execution. Missing or unverified assets: {', '.join(missing_assets)}.",
        )

    return _check(
        "complete_configuration_setup",
        "Complete configuration setup",
        "passed",
        "All catalog datasets and default model assets are available.",
    )


def build_precheck_report(
    phase: str,
    dataset_name: str,
    model_run_path: str | None = None,
    is_test_run: bool = False,
    test_run_percentage: float = 0.3,
) -> dict[str, Any]:
    checks: list[dict[str, str]] = []
    normalized_phase = phase.strip().lower()
    normalized_dataset_name = dataset_name.strip()

    if normalized_phase not in {"training", "inference"}:
        checks.append(_check("phase", "Execution phase", "failed", f"Unsupported pre-check phase: {phase}."))
        return {
            "phase": normalized_phase,
            "ready": False,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
        }

    if not _is_dataset_name_safe(normalized_dataset_name):
        checks.append(_check("dataset_name", "Dataset name", "failed", "Dataset name contains unsupported path characters."))
        dataset_dir = None
    else:
        dataset_dir = DATA_DIR / normalized_dataset_name
        if dataset_dir.is_dir():
            checks.append(_check("dataset_directory", "Dataset directory", "passed", f"Found {dataset_dir}."))
        else:
            checks.append(_check("dataset_directory", "Dataset directory", "failed", f"Dataset directory was not found: {dataset_dir}."))

    if not 0 < test_run_percentage <= 1:
        checks.append(
            _check(
                "test_run_percentage",
                "Quick test percentage",
                "failed",
                "Quick test percentage must be greater than 0 and no greater than 1.",
            )
        )
    else:
        checks.append(
            _check(
                "test_run_percentage",
                "Quick test percentage",
                "passed",
                f"Quick test sampling fraction is {test_run_percentage:.2f}.",
            )
        )

    if dataset_dir is not None:
        checks.append(_check_complete_setup_state())
        setup_dataset_check = _check_setup_dataset_state(normalized_dataset_name)
        if setup_dataset_check is not None:
            checks.append(setup_dataset_check)
        checks.append(_read_csv_contract(dataset_dir / "train.csv", "training_dataset", "Training dataset", required=normalized_phase == "training"))
        checks.append(_read_csv_contract(dataset_dir / "test.csv", "test_dataset", "Test dataset", required=normalized_phase == "inference"))
        validation_check = _read_csv_contract(dataset_dir / "validation.csv", "validation_dataset", "Validation dataset", required=False)
        checks.append(validation_check)
        checks.extend(_check_execution_storage(dataset_dir, normalized_dataset_name, is_test_run, test_run_percentage))

    checks.append(_model_asset_check(DEFAULT_ENCODER_MODEL, "encoder_model", "Sentence encoder assets"))
    checks.append(_model_asset_check(DEFAULT_LLAMA_MODEL, "llama_model", "Llama backbone assets"))
    checks.extend(build_environment_checks())

    if normalized_phase == "inference":
        checks.append(_check_inference_checkpoint(model_run_path))

    return {
        "phase": normalized_phase,
        "ready": not any(check["status"] == "failed" for check in checks),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }