import importlib.metadata
import importlib.util
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from api.job_manager import build_job_callback, initialize_job, run_job
from api.schemas import SetupProvisionRequest, SetupStatusResponse
from mlcore.setup_catalog import get_dataset_spec, get_model_spec
from mlcore.setup_manager import ResourceBusyError, SetupError, provision_assets, record_setup_error, scan_setup_state
from utils.precheck import build_environment_checks


router = APIRouter(prefix="/api", tags=["setup"])


def _runtime_check(key: str, label: str, status_value: str, detail: str) -> dict[str, str]:
    return {"key": key, "label": label, "status": status_value, "detail": detail}


def _runtime_checks() -> list[dict[str, str]]:
    required_modules = {
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn",
        "torch": "PyTorch",
        "transformers": "Transformers",
        "huggingface_hub": "Hugging Face Hub",
        "h5py": "HDF5",
    }
    checks: list[dict[str, str]] = []
    missing_modules: list[str] = []
    installed_versions: list[str] = []

    for module_name, label in required_modules.items():
        if importlib.util.find_spec(module_name) is None:
            missing_modules.append(label)
            continue
        try:
            installed_versions.append(f"{label} {importlib.metadata.version(module_name.replace('_', '-'))}")
        except importlib.metadata.PackageNotFoundError:
            installed_versions.append(label)

    if missing_modules:
        checks.append(
            _runtime_check(
                "backend_dependencies",
                "Backend Python packages",
                "failed",
                f"Missing required packages: {', '.join(missing_modules)}.",
            )
        )
    else:
        checks.append(
            _runtime_check(
                "backend_dependencies",
                "Backend Python packages",
                "passed",
                f"Required packages are importable: {', '.join(installed_versions)}.",
            )
        )

    return checks


def build_setup_status() -> dict[str, Any]:
    setup_state = scan_setup_state()
    return {**setup_state, "runtime_checks": [*_runtime_checks(), *build_environment_checks()]}


def _run_setup_job(job_id: str, request_payload: dict[str, Any]) -> dict[str, Any]:
    callback = build_job_callback(job_id)
    callback({"status": "Provisioning setup assets", "progress": 0.0, "log": "Starting setup provisioning."})
    try:
        result = provision_assets(
            dataset_ids=request_payload["datasets"],
            model_keys=request_payload["models"],
            force=request_payload["force"],
            callback=callback,
        )
    except Exception as error:
        for dataset_id in request_payload["datasets"]:
            record_setup_error("dataset", dataset_id, str(error))
        for model_key in request_payload["models"]:
            record_setup_error("model", model_key, str(error))
        raise
    callback(
        {
            "status": "Setup provisioning complete",
            "progress": 1.0,
            "metrics": {"datasets": len(result["datasets"]), "models": len(result["models"])},
            "log": "Requested setup assets are ready.",
        }
    )
    return result


@router.get("/setup/status", response_model=SetupStatusResponse)
def get_setup_status() -> SetupStatusResponse:
    return SetupStatusResponse(**build_setup_status())


@router.post("/setup/provision", response_model=SetupProvisionRequest.Response, status_code=status.HTTP_202_ACCEPTED)
def start_setup_provision(request: SetupProvisionRequest, background_tasks: BackgroundTasks) -> SetupProvisionRequest.Response:
    if not request.datasets and not request.models:
        raise HTTPException(status_code=422, detail="Select at least one dataset or model to provision.")

    try:
        for dataset_id in request.datasets:
            get_dataset_spec(dataset_id)
        for model_key in request.models:
            model = get_model_spec(model_key)
            if model.requires_hf_token and not build_setup_status()["hf_token_configured"]:
                raise SetupError(f"HF_TOKEN must be configured on the backend before provisioning {model.model_id}.")
    except SetupError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    job_id = uuid4().hex
    request_payload = request.model_dump()
    initialize_job(job_id, "setup", request_payload)
    try:
        background_tasks.add_task(run_job, job_id, lambda: _run_setup_job(job_id, request_payload))
    except (SetupError, ResourceBusyError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error

    return SetupProvisionRequest.Response(job_id=job_id, status="started")