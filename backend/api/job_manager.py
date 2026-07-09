import copy
import threading
from datetime import datetime, timezone
from typing import Any, Callable


ACTIVE_JOBS: dict[str, dict[str, Any]] = {}

_JOB_LOCK = threading.Lock()
_MAX_LOG_LINES = 500


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_progress(progress: Any) -> float | None:
    if progress is None:
        return None

    try:
        normalized = float(progress)
    except (TypeError, ValueError):
        return None

    if normalized <= 1.0:
        normalized *= 100.0

    return max(0.0, min(normalized, 100.0))


def initialize_job(job_id: str, job_type: str, request_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    state = {
        "job_id": job_id,
        "job_type": job_type,
        "status": "PENDING",
        "progress": 0.0,
        "logs": [],
        "done": False,
        "error": None,
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
        "request": request_payload or {},
        "metrics": None,
        "validation_metrics": None,
        "run_id": None,
        "execution_dir": None,
    }
    with _JOB_LOCK:
        ACTIVE_JOBS[job_id] = state
        return copy.deepcopy(state)


def update_job(job_id: str, status: str | None = None, progress: Any = None, **extra: Any) -> dict[str, Any] | None:
    with _JOB_LOCK:
        job = ACTIVE_JOBS.get(job_id)
        if job is None:
            return None

        if status is not None:
            job["status"] = status

        normalized_progress = _normalize_progress(progress)
        if normalized_progress is not None:
            job["progress"] = normalized_progress

        for key, value in extra.items():
            if value is not None:
                job[key] = value

        job["updated_at"] = _timestamp()
        return copy.deepcopy(job)


def append_log(job_id: str, message: str) -> dict[str, Any] | None:
    if not message:
        return None

    with _JOB_LOCK:
        job = ACTIVE_JOBS.get(job_id)
        if job is None:
            return None

        logs = job.setdefault("logs", [])
        logs.append(str(message))
        if len(logs) > _MAX_LOG_LINES:
            del logs[:-_MAX_LOG_LINES]
        job["updated_at"] = _timestamp()
        return copy.deepcopy(job)


def mark_job_complete(job_id: str, status: str = "COMPLETED", **extra: Any) -> dict[str, Any] | None:
    return update_job(job_id, status=status, progress=100.0, done=True, error=None, **extra)


def mark_job_failed(job_id: str, error: str | None = None, status: str = "FAILED", **extra: Any) -> dict[str, Any] | None:
    if error:
        append_log(job_id, error)
    return update_job(job_id, status=status, done=True, error=error, **extra)


def get_job(job_id: str) -> dict[str, Any] | None:
    with _JOB_LOCK:
        job = ACTIVE_JOBS.get(job_id)
        return copy.deepcopy(job) if job is not None else None


def build_job_callback(job_id: str) -> Callable[[dict[str, Any]], None]:
    def _callback(payload: dict[str, Any] | Any) -> None:
        if not isinstance(payload, dict):
            append_log(job_id, str(payload))
            return

        extra = {
            "metrics": payload.get("metrics"),
            "validation_metrics": payload.get("validation_metrics"),
            "run_id": payload.get("run_id"),
            "execution_dir": payload.get("execution_dir"),
        }

        if payload.get("log"):
            append_log(job_id, payload["log"])

        if payload.get("error"):
            mark_job_failed(
                job_id,
                error=str(payload["error"]),
                status=payload.get("status") or "FAILED",
                progress=payload.get("progress"),
                **extra,
            )
            return

        if payload.get("done"):
            final_status = payload.get("status") or "COMPLETED"
            if final_status == "COMPLETED":
                mark_job_complete(job_id, status=final_status, **extra)
            else:
                current = get_job(job_id) or {}
                mark_job_failed(
                    job_id,
                    error=current.get("error"),
                    status=final_status,
                    progress=payload.get("progress"),
                    **extra,
                )
            return

        update_job(job_id, status=payload.get("status"), progress=payload.get("progress"), **extra)

    return _callback


def run_job(job_id: str, job_callable: Callable[[], None]) -> None:
    try:
        update_job(job_id, status="RUNNING")
        job_callable()
        current = get_job(job_id)
        if current and not current.get("done"):
            mark_job_complete(job_id)
    except Exception as exc:
        mark_job_failed(job_id, error=str(exc))


def iter_active_jobs() -> list[dict[str, Any]]:
    with _JOB_LOCK:
        return [copy.deepcopy(job) for job in ACTIVE_JOBS.values()]


def fail_unfinished_jobs(reason: str) -> list[dict[str, Any]]:
    affected_jobs: list[dict[str, Any]] = []

    for job in iter_active_jobs():
        if job.get("done"):
            continue

        updated = mark_job_failed(job["job_id"], error=reason, status="FAILED")
        if updated is not None:
            affected_jobs.append(updated)

    return affected_jobs