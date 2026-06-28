from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.job_manager import get_job
from api.routes.inference import router as inference_router
from api.routes.metadata import router as metadata_router
from api.routes.training import router as training_router
from api.schemas import JobStatusResponse


app = FastAPI(title="Log Sentinel API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(metadata_router)
app.include_router(training_router)
app.include_router(inference_router)


@app.get("/api/status/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatusResponse(
        job_id=job["job_id"],
        job_type=job["job_type"],
        status=job["status"],
        progress=job["progress"],
        done=job["done"],
        latest_logs=job.get("logs", [])[-50:],
        error=job.get("error"),
        metrics=job.get("metrics"),
        validation_metrics=job.get("validation_metrics"),
        run_id=job.get("run_id"),
        execution_dir=job.get("execution_dir"),
    )