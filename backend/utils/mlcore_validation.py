from pathlib import Path

from fastapi import HTTPException

from config import DEFAULT_ENCODER_MODEL, DEFAULT_LLAMA_MODEL, MODELS_DIR


def expected_model_directories() -> dict[str, Path]:
    return {
        DEFAULT_ENCODER_MODEL: MODELS_DIR / DEFAULT_ENCODER_MODEL.split('/')[-1],
        DEFAULT_LLAMA_MODEL: MODELS_DIR / DEFAULT_LLAMA_MODEL.split('/')[-1],
    }


def find_missing_model_directories() -> list[str]:
    missing: list[str] = []
    for model_name, model_path in expected_model_directories().items():
        if not model_path.exists() or not any(model_path.iterdir()):
            missing.append(f"{model_name} -> {model_path}")
    return missing


def assert_required_models_present() -> None:
    missing = find_missing_model_directories()
    if not missing:
        return

    missing_list = '; '.join(missing)
    raise HTTPException(
        status_code=503,
        detail=(
            "Required local model assets are missing. Run scripts/setup.ps1 and choose model download, "
            f"or execute python -m mlcore.download_models first. Missing: {missing_list}"
        ),
    )