import json
import math
import statistics
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from api.schemas import (
    DatasetListResponse,
    ModelListResponse,
    RunDetailResponse,
    RunHistoryResponse,
    SyncRunsResponse,
)
from config import DATA_DIR, DB_PATH, EXECUTIONS_DIR
from mlcore.config import DEFAULT_LLAMA_MODEL
from utils.database_manager import DatabaseManager


router = APIRouter(prefix="/api", tags=["metadata"])
db_manager = DatabaseManager(DB_PATH)


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _resolve_report_path(report_path: str | None) -> Path | None:
    if not report_path:
        return None

    candidate = Path(report_path)
    if candidate.exists():
        return candidate

    fallback = EXECUTIONS_DIR / candidate.name
    if fallback.exists():
        return fallback

    return candidate


def _read_run_metrics(report_path: str | None) -> dict[str, Any] | None:
    resolved_report_path = _resolve_report_path(report_path)
    if resolved_report_path is None:
        return None

    metrics_path = resolved_report_path / "run_metrics.json"
    if not metrics_path.is_file():
        return None

    with metrics_path.open("r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None

    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)

    if lower_index == upper_index:
        return ordered[lower_index]

    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    weight = position - lower_index
    return lower_value + ((upper_value - lower_value) * weight)


def _extract_resource_summary(run_metrics: dict[str, Any]) -> dict[str, Any]:
    resource_usage = run_metrics.get("resource_usage", {}) if isinstance(run_metrics, dict) else {}
    ram_values = [float(value) for value in resource_usage.get("ram_usage_gb", []) if value is not None]
    gpu_values = [float(value) for value in resource_usage.get("gpu_mem_used_gb", []) if value is not None]

    return {
        "ram": {
            "avg_ram_usage_gb": statistics.mean(ram_values) if ram_values else None,
            "p95_ram_usage_gb": _percentile(ram_values, 0.95),
        },
        "gpu": {
            "avg_gpu_vram_gb": statistics.mean(gpu_values) if gpu_values else None,
            "p95_gpu_vram_gb": _percentile(gpu_values, 0.95),
        },
    }


def _select_primary_evaluation(run_metrics: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    evaluation = run_metrics.get("evaluation", {}) if isinstance(run_metrics, dict) else {}

    if "test" in evaluation:
        return "test", evaluation["test"]
    if "validation" in evaluation:
        return "validation", evaluation["validation"]

    first_entry = next(iter(evaluation.items()), (None, {}))
    return first_entry[0], first_entry[1]


def _infer_dataset_name(nickname: str) -> str | None:
    dataset_names = sorted((path.name for path in DATA_DIR.iterdir() if path.is_dir()), key=len, reverse=True)
    for dataset_name in dataset_names:
        if nickname.startswith(f"Inference_{dataset_name}_") or nickname.startswith(f"{dataset_name}_"):
            return dataset_name
    return None


def _infer_run_type(run_metrics: dict[str, Any]) -> str:
    training_loss = run_metrics.get("training_loss") if isinstance(run_metrics, dict) else None
    return "Training" if training_loss else "Inference"


def _build_synced_db_metrics(run_metrics: dict[str, Any]) -> dict[str, Any]:
    metric_prefix, evaluation_payload = _select_primary_evaluation(run_metrics)
    metrics = evaluation_payload.get("metrics", {}) if isinstance(evaluation_payload, dict) else {}
    resource_summary = _extract_resource_summary(run_metrics)
    timestamps = run_metrics.get("resource_usage", {}).get("timestamps", []) if isinstance(run_metrics, dict) else []

    return {
        "total_run_time_sec": timestamps[-1] if timestamps else None,
        "training_time_sec": None,
        "testing_time_sec": metrics.get("test_inference_time_sec") or metrics.get("validation_inference_time_sec"),
        "accuracy": metrics.get(f"{metric_prefix}_accuracy") if metric_prefix else None,
        "precision": metrics.get(f"{metric_prefix}_precision") if metric_prefix else None,
        "f1_score": metrics.get(f"{metric_prefix}_f1_score") if metric_prefix else None,
        "recall": metrics.get(f"{metric_prefix}_recall") if metric_prefix else None,
        "avg_ram_usage_gb": resource_summary.get("ram", {}).get("avg_ram_usage_gb"),
        "peak_95_ram_usage_gb": resource_summary.get("ram", {}).get("p95_ram_usage_gb"),
        "avg_gpu_vram_gb": resource_summary.get("gpu", {}).get("avg_gpu_vram_gb"),
        "peak_95_gpu_vram_gb": resource_summary.get("gpu", {}).get("p95_gpu_vram_gb"),
    }


@router.get("/datasets", response_model=DatasetListResponse)
def get_datasets() -> DatasetListResponse:
    datasets = sorted(path.name for path in DATA_DIR.iterdir() if path.is_dir())
    return DatasetListResponse(datasets=datasets)


@router.get("/models", response_model=ModelListResponse)
def get_models() -> ModelListResponse:
    model_paths: set[str] = set()

    runs_df = db_manager.get_all_runs()
    run_ids = [] if runs_df.empty else runs_df["id"].dropna().tolist()
    for run_id in run_ids:
        run_details = db_manager.get_run_details(int(run_id))
        if not run_details:
            continue
        if run_details.get("run_type") != "Training" or run_details.get("status") != "COMPLETED":
            continue

        report_path = run_details.get("report_path")
        resolved_report_path = _resolve_report_path(report_path)
        if not resolved_report_path:
            continue

        output_model_path = resolved_report_path / "output_model"
        if output_model_path.is_dir():
            model_paths.add(str(output_model_path.resolve()))

    for execution_dir in EXECUTIONS_DIR.iterdir():
        if not execution_dir.is_dir():
            continue

        output_model_path = execution_dir / "output_model"
        if output_model_path.is_dir():
            model_paths.add(str(output_model_path.resolve()))

    return ModelListResponse(models=sorted(model_paths))


@router.get("/runs", response_model=RunHistoryResponse)
def get_runs() -> RunHistoryResponse:
    runs_df = db_manager.get_all_runs()
    if runs_df.empty:
        return RunHistoryResponse(runs=[])

    records = runs_df.to_dict(orient="records")
    sanitized_records = [_sanitize_value(record) for record in records]
    return RunHistoryResponse(runs=sanitized_records)


@router.get("/runs/{run_id}", response_model=RunDetailResponse)
def get_run_details(run_id: int) -> RunDetailResponse:
    run_details = db_manager.get_run_details(run_id)
    if not run_details:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return RunDetailResponse(
        run=_sanitize_value(run_details),
        run_metrics=_read_run_metrics(run_details.get("report_path")),
    )


@router.post("/sync-runs", response_model=SyncRunsResponse)
def sync_runs() -> SyncRunsResponse:
    execution_dirs = [path for path in EXECUTIONS_DIR.iterdir() if path.is_dir()]
    imported_nicknames: list[str] = []
    skipped_nicknames: list[str] = []

    for execution_dir in execution_dirs:
        nickname = execution_dir.name
        if db_manager.get_run_by_nickname(nickname):
            skipped_nicknames.append(nickname)
            continue

        metrics_path = execution_dir / "run_metrics.json"
        if not metrics_path.is_file():
            skipped_nicknames.append(nickname)
            continue

        try:
            with metrics_path.open("r", encoding="utf-8") as metrics_file:
                run_metrics = json.load(metrics_file)
        except Exception:
            skipped_nicknames.append(nickname)
            continue

        run_type = _infer_run_type(run_metrics)
        dataset_name = _infer_dataset_name(nickname)

        run_id = db_manager.create_new_run(
            run_type,
            DEFAULT_LLAMA_MODEL,
            dataset_name,
            {},
            nickname,
            status="COMPLETED",
        )
        db_manager.save_final_metrics(run_id, _build_synced_db_metrics(run_metrics))
        db_manager.update_run_status(run_id, "COMPLETED", str(execution_dir))
        imported_nicknames.append(nickname)

    return SyncRunsResponse(
        scanned_count=len(execution_dirs),
        imported_count=len(imported_nicknames),
        skipped_count=len(skipped_nicknames),
        imported_nicknames=imported_nicknames,
        skipped_nicknames=skipped_nicknames,
    )