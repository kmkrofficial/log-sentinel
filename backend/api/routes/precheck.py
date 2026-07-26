from fastapi import APIRouter

from api.schemas import PreCheckRequest, PreCheckResponse
from utils.precheck import build_precheck_report


router = APIRouter(prefix="/api", tags=["pre-check"])


@router.post("/pre-check", response_model=PreCheckResponse)
def run_precheck(request: PreCheckRequest) -> PreCheckResponse:
    report = build_precheck_report(
        phase=request.phase,
        dataset_name=request.dataset_name,
        model_run_path=request.model_run_path,
        is_test_run=request.is_test_run,
        test_run_percentage=request.test_run_percentage,
    )
    return PreCheckResponse(**report)