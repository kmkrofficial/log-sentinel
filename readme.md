# LogSentinel

LogSentinel is now organized as a three-part monorepo:

- `backend/`: FastAPI orchestration, async job management, and SQLite-backed run history.
- `mlcore/`: Stateless machine-learning package with controllers, datasets, models, filesystem artifacts, and the headless CLI.
- `frontend/`: React + Vite control plane for launching jobs, polling live status, browsing runs, and visualizing `run_metrics.json` artifacts.

## Structure

```text
log-sentinel/
├── backend/
│   ├── api/             # FastAPI app, async job manager, routes, schemas
│   ├── utils/           # Backend-only utilities such as DatabaseManager
│   ├── config.py        # Backend shim for DB path + mlcore paths
│   ├── logsentinel.db   # SQLite run history
│   └── requirements.txt
├── mlcore/
│   ├── engine/          # Training and inference controllers
│   ├── utils/           # Stateless ML helpers
│   ├── datasets/        # Local datasets
│   ├── models/          # Cached / local model weights
│   ├── executions/      # Run outputs including run_metrics.json
│   ├── prepareData/     # Data preparation helpers
│   ├── cli.py           # Headless CLI with train / inference subcommands
│   └── config.py        # Stateless filesystem and hyperparameter config
├── frontend/
│   ├── src/
│   │   ├── services/    # API integration helpers
│   │   ├── hooks/       # Polling / UI control hooks
│   │   ├── pages/       # Dashboard, train, job, run details views
│   │   └── components/  # Shared React UI pieces
│   └── package.json
└── README.md
```

## Backend Setup

Python 3.11 is recommended. GPU-backed training still depends on a CUDA-compatible PyTorch install and the model stack already used in this repository.

1. Create and activate a virtual environment.

```bash
python -m venv .venv
```

Windows:

```bash
.\.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

2. Install PyTorch for your CUDA/runtime target.

3. Install backend dependencies.

```bash
cd backend
pip install -r requirements.txt
```

4. Start the FastAPI server.

```bash
uvicorn api.main:app --reload
```

The backend will be available at `http://localhost:8000`.

## ML Core CLI

The ML package can also be run without the API or frontend.

Training:

```bash
python -m mlcore.cli train --dataset-name BGL
```

Inference:

```bash
python -m mlcore.cli inference --model-run-path mlcore/executions/<run-name> --dataset-name BGL
```

Optional training overrides:

```bash
python -m mlcore.cli train --dataset-name BGL --hyperparameters-json '{"micro_batch_size": 16}'
```

## Frontend Setup

From the repository root:

```bash
cd frontend
npm install
```

Start the Vite development server:

```bash
npm run dev
```

The frontend defaults to `http://localhost:5173` and talks to `http://localhost:8000/api`.

Optional override:

```bash
VITE_API_BASE_URL=http://localhost:8000/api
```

## Running Both Servers

Use two terminals.

Terminal 1:

```bash
cd backend
uvicorn api.main:app --reload
```

Terminal 2:

```bash
cd frontend
npm run dev
```

## Scripts

The repository now includes a dedicated [scripts](scripts) folder for setup, deployment prep, and local execution management.

- [scripts/setup.ps1](scripts/setup.ps1): verifies Python 3.11, creates or reuses `.venv`, verifies the virtual environment interpreter, lets you choose `frontend`, `mlcore`, `backend-mlcore`, or `all`, installs dependencies, and builds the frontend when requested.
- [scripts/start.ps1](scripts/start.ps1): starts `backend`, `frontend`, or `all`.
- [scripts/stop.ps1](scripts/stop.ps1): stops `backend`, `frontend`, or `all`.
- [scripts/restart.ps1](scripts/restart.ps1): restarts `backend`, `frontend`, or `all`.

Examples:

```powershell
PowerShell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
PowerShell -ExecutionPolicy Bypass -File .\scripts\start.ps1 all
PowerShell -ExecutionPolicy Bypass -File .\scripts\stop.ps1 backend
PowerShell -ExecutionPolicy Bypass -File .\scripts\restart.ps1 frontend
```

The scripts store local process state under `scripts/.runtime/`, which is ignored by Git.

## Current Frontend Surface

- Dashboard view: lists historical runs from `GET /api/runs`.
- Start Run view: selects a dataset from `GET /api/datasets`, submits `POST /api/train`, then redirects into live polling.
- Live Progress view: polls `GET /api/status/{job_id}` every 3 seconds until the job reaches a terminal state.
- Run Details view: loads `GET /api/runs/{run_id}` and renders training loss, RAM/VRAM series, and anomaly score distributions from `run_metrics.json`.

## API Notes

Key backend endpoints:

- `GET /api/datasets`
- `GET /api/models`
- `GET /api/runs`
- `GET /api/runs/{run_id}`
- `POST /api/train`
- `POST /api/inference`
- `GET /api/status/{job_id}`

The backend now owns SQLite persistence. `mlcore/` does not import the database layer.

## Operational Note

`ACTIVE_JOBS` is currently in-memory inside the FastAPI process. Live job state will be lost if the backend process restarts. That matches the current refactor phase, but it is not a durable job queue.

When the managed backend process is stopped through the scripts, any unfinished application-started jobs are marked `FAILED` during FastAPI shutdown. CLI-driven `mlcore` runs remain independent because they do not execute inside the backend process.