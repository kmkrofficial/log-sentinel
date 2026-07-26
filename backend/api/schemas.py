from typing import Any, Literal

from pydantic import BaseModel, Field


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


class DataPrepRequest(BaseModel):
    dataset_name: str
    options: dict[str, Any] = Field(default_factory=dict)


class DataPrepStatusResponse(BaseModel):
    datasets: list[dict[str, Any]]


class SetupCheckItem(BaseModel):
    key: str
    label: str
    status: Literal["passed", "warning", "failed"]
    detail: str


class SetupProvisionRequest(BaseModel):
    datasets: list[str] = []
    models: list[str] = []
    force: bool = False

    class Response(BaseModel):
        job_id: str
        status: str


class SetupStatusResponse(BaseModel):
    checked_at: str
    hf_token_configured: bool
    downloads_path: str
    datasets: list[dict[str, Any]]
    models: list[dict[str, Any]]
    storage: dict[str, Any]
    runtime_checks: list[SetupCheckItem]


class PreCheckRequest(BaseModel):
    phase: Literal["training", "inference"]
    dataset_name: str
    model_run_path: str | None = None
    is_test_run: bool = False
    test_run_percentage: float = 0.3


class PreCheckItem(BaseModel):
    key: str
    label: str
    status: Literal["passed", "warning", "failed"]
    detail: str


class PreCheckResponse(BaseModel):
    phase: Literal["training", "inference"]
    ready: bool
    checked_at: str
    checks: list[PreCheckItem]


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


class ActiveJobsResponse(BaseModel):
    jobs: list[JobStatusResponse]


class SyncRunsResponse(BaseModel):
    scanned_count: int
    imported_count: int
    skipped_count: int
    imported_nicknames: list[str]
    skipped_nicknames: list[str]