from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from api.job_manager import build_job_callback, initialize_job, run_job, update_job
from api.schemas import DataPrepRequest, DataPrepStatusResponse, JobStartedResponse
from mlcore.prepareData.pipeline import get_preparation_status, prepare_dataset, validate_dataset_preparation
from mlcore.setup_catalog import DATASET_SPECS
from mlcore.setup_manager import record_setup_error


router = APIRouter(prefix="/api", tags=["data-preparation"])


def _run_data_prep_job(job_id: str, request_payload: dict) -> dict:
    callback = build_job_callback(job_id)
    callback({"status": "Preparing dataset", "progress": 0.0, "log": "Starting dataset preparation."})
    try:
        result = prepare_dataset(request_payload["dataset_name"], callback, request_payload.get("options", {}))
    except Exception as error:
        record_setup_error("dataset", request_payload["dataset_name"], str(error))
        raise
    callback({"status": "Finalizing prepared dataset", "progress": 1.0, "metrics": result["sequence_counts"]})
    return result


@router.get("/data-prep/status", response_model=DataPrepStatusResponse)
def get_data_prep_status() -> DataPrepStatusResponse:
    return DataPrepStatusResponse(datasets=[get_preparation_status(dataset_id) for dataset_id in DATASET_SPECS])


@router.post("/data-prep", response_model=JobStartedResponse, status_code=status.HTTP_202_ACCEPTED)
def start_data_prep_job(request: DataPrepRequest, background_tasks: BackgroundTasks) -> JobStartedResponse:
    try:
        validate_dataset_preparation(request.dataset_name, request.options)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    job_id = uuid4().hex
    request_payload = request.model_dump()
    initialize_job(job_id, "prep", request_payload)
    update_job(job_id, status="PENDING")
    background_tasks.add_task(run_job, job_id, lambda: _run_data_prep_job(job_id, request_payload))
    return JobStartedResponse(job_id=job_id, status="started")