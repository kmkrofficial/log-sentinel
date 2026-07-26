from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from api.job_manager import build_job_callback, initialize_job, run_job, update_job
from api.schemas import InferenceRequest, JobStartedResponse
from config import DB_PATH, EXECUTIONS_DIR
from mlcore.config import DEFAULT_LLAMA_MODEL, get_hyperparameters
from mlcore.engine.inference_controller import InferenceController
from utils.database_manager import DatabaseManager
from utils.precheck import build_precheck_report


router = APIRouter(prefix="/api", tags=["inference"])


def _generate_inference_nickname(dataset_name: str, is_test_run: bool, test_run_percentage: float, manual_nickname: str | None) -> str:
    if manual_nickname:
        return f"{manual_nickname}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"

    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    nickname = f"Inference_{dataset_name}_{now}"
    if is_test_run:
        nickname += f"_{int(test_run_percentage * 100)}pct_TEST"
    return nickname


def _build_inference_db_metrics(result: dict) -> dict:
    resource_summary = result.get("resource_summary", {}) or {}
    final_metrics = result.get("metrics", {}) or {}

    return {
        "total_run_time_sec": result.get("total_run_time_sec"),
        "testing_time_sec": result.get("testing_time_sec"),
        "accuracy": final_metrics.get("test_accuracy"),
        "precision": final_metrics.get("test_precision"),
        "f1_score": final_metrics.get("test_f1_score"),
        "recall": final_metrics.get("test_recall"),
        "avg_ram_usage_gb": resource_summary.get("ram", {}).get("avg_ram_usage_gb"),
        "peak_95_ram_usage_gb": resource_summary.get("ram", {}).get("p95_ram_usage_gb"),
        "avg_gpu_vram_gb": resource_summary.get("gpu", {}).get("avg_gpu_vram_gb"),
        "peak_95_gpu_vram_gb": resource_summary.get("gpu", {}).get("p95_gpu_vram_gb"),
    }


def _run_inference_job(job_id: str, run_id: int, nickname: str, request_payload: dict) -> dict | None:
    db_manager = DatabaseManager(DB_PATH)
    controller = InferenceController(
        model_run_path=request_payload["model_run_path"],
        dataset_name=request_payload["dataset_name"],
        callback=build_job_callback(job_id),
        is_test_run=request_payload.get("is_test_run", False),
        test_run_percentage=request_payload.get("test_run_percentage", 0.3),
        manual_nickname=request_payload.get("manual_nickname"),
    )
    controller.run_id = run_id
    controller.run_nickname = nickname

    update_job(job_id, status="RUNNING", run_id=run_id, execution_dir=str(EXECUTIONS_DIR / nickname))
    db_manager.update_run_status(run_id, "RUNNING", None)

    try:
        result = controller.run_inference() or {}
        status_value = result.get("status", "FAILED")

        db_manager.save_final_metrics(run_id, _build_inference_db_metrics(result))

        report_path = result.get("execution_dir") if status_value == "COMPLETED" else None
        db_manager.update_run_status(run_id, status_value, report_path)
        return result
    except Exception:
        db_manager.update_run_status(run_id, "FAILED", None)
        raise


@router.post("/inference", response_model=JobStartedResponse, status_code=status.HTTP_202_ACCEPTED)
def start_inference_job(request: InferenceRequest, background_tasks: BackgroundTasks) -> JobStartedResponse:
    precheck_report = build_precheck_report(
        phase="inference",
        dataset_name=request.dataset_name,
        model_run_path=request.model_run_path,
        is_test_run=request.is_test_run,
        test_run_percentage=request.test_run_percentage,
    )
    if not precheck_report["ready"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"message": "Inference pre-checks failed.", "precheck": precheck_report},
        )

    job_id = uuid4().hex
    request_payload = request.model_dump()
    nickname = _generate_inference_nickname(
        request.dataset_name,
        request.is_test_run,
        request.test_run_percentage,
        request.manual_nickname,
    )
    hyperparameters = get_hyperparameters(request.dataset_name)

    db_manager = DatabaseManager(DB_PATH)
    run_id = db_manager.create_new_run(
        "Inference",
        DEFAULT_LLAMA_MODEL,
        request.dataset_name,
        hyperparameters,
        nickname,
        status="PENDING",
    )

    initialize_job(job_id, "inference", request_payload)
    update_job(job_id, status="PENDING", run_id=run_id, execution_dir=str(EXECUTIONS_DIR / nickname))
    background_tasks.add_task(run_job, job_id, lambda: _run_inference_job(job_id, run_id, nickname, request_payload))
    return JobStartedResponse(job_id=job_id, status="started")