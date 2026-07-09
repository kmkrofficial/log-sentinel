from typing import Any

from pydantic import BaseModel


class DatasetListResponse(BaseModel):
    datasets: list[str]


class ModelListResponse(BaseModel):
    models: list[str]


class RunSummary(BaseModel):
    id: int
    start_time: str
    nickname: str | None = None
    dataset_name: str | None = None
    status: str | None = None
    total_run_time_sec: float | None = None
    f1_score: float | None = None
    precision: float | None = None
    recall: float | None = None
    accuracy: float | None = None


class RunHistoryResponse(BaseModel):
    runs: list[RunSummary]


class RunDetailResponse(BaseModel):
    run: dict[str, Any]
    run_metrics: dict[str, Any] | None = None


class TrainRequest(BaseModel):
    dataset_name: str
    hyperparameters: dict[str, Any] | None = None
    is_test_run: bool = False
    test_run_percentage: float = 0.3


class InferenceRequest(BaseModel):
    model_run_path: str
    dataset_name: str
    is_test_run: bool = False
    test_run_percentage: float = 0.3
    manual_nickname: str | None = None


class JobStartedResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    progress: float
    done: bool
    latest_logs: list[str]
    error: str | None = None
    metrics: dict[str, Any] | None = None
    validation_metrics: dict[str, Any] | None = None
    run_id: int | None = None
    execution_dir: str | None = None


class SyncRunsResponse(BaseModel):
    scanned_count: int
    imported_count: int
    skipped_count: int
    imported_nicknames: list[str]
    skipped_nicknames: list[str]