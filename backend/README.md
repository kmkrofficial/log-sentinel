# Backend

The `backend/` package is LogSentinel's control plane. It provisions datasets and models, runs preparation/training/inference jobs, tracks live job status, stores run metadata in SQLite, and exposes the state consumed by the React Run Model and Results sections.

The backend exists because the ML pipeline is too heavy and stateful to be coupled directly to the UI. By putting orchestration behind FastAPI, the repository can support a browser client, headless scripts, and future automation without changing the ML runtime itself.

## Why This Layer Exists

- It decouples web concerns from the model runtime in `mlcore/`.
- It provides a stable API for training, inference, status polling, and metadata lookup.
- It centralizes persistence in SQLite while keeping large metrics artifacts on disk.
- It lets the frontend stay thin: the browser submits payloads and renders results, while the backend owns job state and lifecycle rules.

## Directory Guide

| Path | Significance |
| --- | --- |
| `api/` | FastAPI entrypoint, request/response schemas, job-state machinery, and API routes. |
| `utils/` | Backend-only helpers for SQLite persistence and pre-flight validation. |
| `config.py` | Backend shim that imports path and hyperparameter config from `mlcore`, then adds the backend-local SQLite path. |
| `requirements.txt` | Consolidated Python dependency list used by the backend and, in practice, by most `mlcore` workflows too. |
| `chart_gen.py` | Research utility for generating comparison plots against other models; not part of the live API. |
| `dataset_analyzer.py` | Operational helper for auditing prepared dataset sizes and class balance. |
| `directory_scanner.py` | Snapshot utility for documentation/auditing; not part of runtime request handling. |

## API File Guide

| File | What it does | Why it exists |
| --- | --- | --- |
| `api/main.py` | Builds the FastAPI app, wires CORS, includes routers, exposes job status, and marks unfinished jobs as failed on shutdown. | Keeps the application boundary small and explicit. |
| `api/schemas.py` | Defines Pydantic request and response models for jobs, runs, datasets, models, and sync operations. | Gives the API a typed contract instead of passing loose dictionaries around. |
| `api/job_manager.py` | Maintains an in-memory registry of active jobs, normalizes callback payloads, appends logs, and handles terminal states. | The backend needs fast mutable job state while a process is running, and SQLite alone is not a good fit for high-frequency progress updates. |
| `api/routes/training.py` | Implements `POST /api/train`, creates the initial run record, launches `TrainingController`, and maps controller output back into database metrics. | Training jobs need API-specific glue such as job IDs, HTTP status codes, and database coordination. |
| `api/routes/inference.py` | Implements `POST /api/inference` with the same control-plane pattern used for training. | Keeps inference orchestration parallel to training while allowing distinct metric mapping. |
| `api/routes/setup.py` | Implements setup inventory and selected asset provisioning endpoints. | Gives the configuration step a non-secret view of archives, models, runtime checks, storage, and setup jobs. |
| `api/routes/data_prep.py` | Implements preparation status plus background preparation jobs for BGL, HDFS_v1, Thunderbird, and conditionally Hadoop. | Lets the browser start supported corpus construction while explicitly blocking unlabelled Hadoop. |
| `api/routes/precheck.py` | Implements `POST /api/pre-check` for training and inference readiness reports. | Gives the browser a complete requirement report before it starts an expensive workload. |
| `api/routes/metadata.py` | Exposes datasets, available models, run history, run details, and sync-from-artifacts endpoints. | The frontend needs a read API that is separate from job-launch endpoints. |

## Utility File Guide

| File | What it does | Why it exists |
| --- | --- | --- |
| `utils/database_manager.py` | Creates the SQLite schema and performs CRUD-style operations for runs. | Keeps persistence logic out of route handlers and makes the data model easy to audit. |
| `utils/mlcore_validation.py` | Checks whether the expected local encoder and backbone directories exist before a run is accepted. | Prevents expensive jobs from starting only to fail immediately on missing model assets. |
| `config.py` | Imports the active path and hyperparameter constants from `mlcore` and adds `DB_PATH`. | Makes the backend the owner of database location without duplicating ML configuration. |

## Job Lifecycle

1. A client provisions setup assets, prepares a dataset, trains, or evaluates a model.
2. The backend creates a matching in-memory job record; training and inference also create a `PENDING` SQLite run row.
3. FastAPI `BackgroundTasks` calls `run_job()`, which wraps the controller or preparation pipeline.
4. The worker emits structured callbacks such as log lines, progress, metric snapshots, run ID, and execution directory.
5. `api/job_manager.py` normalizes those updates into a polling-friendly state object.
6. Setup writes only non-secret inventory metadata; preparation writes generated CSV artifacts into the selected dataset directory; training and inference write summary metrics into SQLite.

## Design Decisions

### BackgroundTasks instead of a full queue system

The backend uses FastAPI `BackgroundTasks` and an in-memory job manager rather than Celery, Redis, or a broker-backed queue. That choice keeps local development simple and matches the workstation-oriented deployment model. The tradeoff is that jobs live inside the backend process, so restarting the backend kills active runs.

### SQLite plus filesystem artifacts

SQLite stores only summary data and indexing metadata. Detailed loss curves, probability outputs, and telemetry live in `run_metrics.json` and related execution artifacts. This split keeps the database queryable and lightweight while still preserving rich run evidence.

### Model validation before launch

The backend runs the same pre-check report used by the browser immediately before it accepts training or inference. It validates prepared CSV schema and availability, local model assets, inference checkpoints, CUDA/BF16/FlashAttention 3 compatibility, bitsandbytes CUDA binaries, Linux Triton build prerequisites, and embedding storage/write access. Failed required checks return `422` with the full report, so direct API clients cannot bypass the UI gate.

### Verified provisioning

The setup API accepts only assets from `mlcore.setup_catalog`. Dataset downloads are resumable, checked against Zenodo file size and MD5 metadata, extracted with ZIP/TAR path-traversal protections, and promoted atomically into `datasets/<id>/raw/`. Setup state is persisted under `logsentinel_data/setup-state.json` without credentials. The gated Llama download uses backend `HF_TOKEN`; no endpoint exposes or stores a token from the browser.

### Sync-from-artifacts endpoint

`POST /api/sync-runs` exists because execution directories are the durable source of truth for rich metrics. If a database row is lost or a run directory is copied in manually, the backend can reconstruct a useful run-history entry from the artifact bundle.