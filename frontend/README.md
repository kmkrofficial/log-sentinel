# Frontend

The `frontend/` package is the browser-based control plane for LogSentinel. It is built with React, React Router, Axios, Recharts, and Vite. Its job is not to run machine-learning logic. Its job is to submit well-formed API requests, poll live job state, and render run metadata plus artifact-backed charts.

The frontend exists because the repository outgrew the earlier Streamlit-only interaction model. A dedicated browser client makes route-based navigation, run inspection, and a richer visual layer easier without pushing UI concerns into the ML code.

## Directory Guide

| Path | Significance |
| --- | --- |
| `src/pages/` | Route-level Run Model, Results, job-status, and run-detail screens. |
| `src/components/` | Shared UI primitives such as the app shell, job panel, status badge, and run table. |
| `src/hooks/` | Reusable client-side behavior, currently focused on job-status polling. |
| `src/services/` | Axios-based API client wrappers for the backend. |
| `src/App.jsx` | Route table for the browser application. |
| `src/main.jsx` | React entrypoint that mounts the router. |
| `src/App.css`, `src/index.css` | Visual system and application styling. |
| `package.json` | Frontend dependencies and scripts. |
| `vite.config.js` | Vite build configuration. |

## File Guide

| File | What it does | Why it exists |
| --- | --- | --- |
| `src/main.jsx` | Mounts the app inside `BrowserRouter`. | Keeps startup small and conventional. |
| `src/App.jsx` | Declares the two primary routes, `Run Model` and `Results`, plus deep links for jobs and run details. | Replaces fragmented workflow navigation with one gated operational flow. |
| `src/pages/RunModelPage.jsx` | Coordinates the configuration, preparation, and execution stepper. | Ensures users cannot skip prerequisite work through UI routing. |
| `src/pages/ResultsPage.jsx` | Shows current setup/preparation/training/inference activity beside the historical run ledger. | Separates live operations from completed evidence. |
| `src/pages/JobStatusPage.jsx` | Displays live progress for a single job using polling. | Long-running ML jobs need a dedicated status screen instead of optimistic fire-and-forget UX. |
| `src/pages/RunDetailsPage.jsx` | Merges SQLite-backed run metadata with `run_metrics.json` and attempts to render charts. | Run inspection is a first-class product capability, not a debugging afterthought. |
| `src/components/AppShell.jsx` | Provides the app frame, branding, and primary navigation. | Keeps the route pages focused on their own content. |
| `src/components/WorkflowStepper.jsx` | Shows locked, required, active, running, and complete phases of the Run Model workflow. | Makes execution dependencies visible and prevents accidental ordering mistakes. |
| `src/components/ConfigurationSetupStep.jsx` | Renders the setup inventory, runtime audit, and selected asset provisioning controls. | Makes all dataset/model/package blockers visible before preparation. |
| `src/components/DatasetPreparationStep.jsx` | Renders strategy eligibility and controls for BGL, HDFS_v1, Thunderbird, and blocked Hadoop. | Preserves dataset-specific labeling rules instead of masking them behind one generic parser. |
| `src/components/ModelExecutionStep.jsx` | Hosts training and evaluation controls for prepared datasets, plus final pre-checks. | Keeps Step 3 focused on executable model work. |
| `src/components/CurrentActivityPanel.jsx` | Shows active setup, preparation, training, and inference jobs on Results. | Prevents live workload visibility from being lost while browsing history. |
| `src/components/LiveJobPanel.jsx` | Renders progress bars, terminal logs, run IDs, and validation metrics for an active job. | Concentrates the polling-driven UX in one component. |
| `src/components/PrecheckPanel.jsx` | Renders the complete pre-execution requirement report and readiness state. | Makes environment and input failures visible before the user launches training or inference. |
| `src/components/RunTable.jsx` | Displays the run ledger with core metrics and links to details. | Reuses one consistent table representation on the dashboard. |
| `src/components/StatusBadge.jsx` | Maps raw status values to visual tones. | Prevents status rendering logic from being duplicated across screens. |
| `src/hooks/useJobStatus.js` | Polls `GET /api/status/{job_id}` every three seconds and stops on terminal states. | Keeps side-effect-heavy polling behavior out of page components. |
| `src/services/api.js` | Defines the Axios client, base URL, and backend endpoint wrappers. | Centralizes HTTP behavior and endpoint paths. |
| `vite.config.js` | Enables the React plugin and Vite defaults. | Keeps the frontend build lightweight and easy to run locally. |

## Why The Frontend Is Structured This Way

### Thin client over a backend control plane

The browser does not attempt to manage process state itself. It delegates job lifecycle to FastAPI and treats the API as the system of coordination. That keeps the UI simpler and makes it possible to reuse the same backend from scripts or other clients.

### Polling instead of WebSockets

`useJobStatus.js` polls every three seconds. That is a deliberate tradeoff: the backend stays simpler, local development is easier, and the job manager does not need a push channel. The cost is a small amount of update latency.

### Artifact-driven charts

The run-details screen is designed around `run_metrics.json` rather than around backend-generated PNGs. That decision keeps the backend stateless with respect to chart rendering and lets the frontend compose multiple chart views from one stored artifact bundle.

### Gated run workflow

`/run` has three ordered stages: Configuration Setup, Data Preparation, and Model Training. The browser reads setup inventory and preparation eligibility from the backend, locks later stages until prior requirements are satisfied, and preserves active setup/preparation/training/inference job IDs through `JobContext` and local storage. Legacy workflow URLs redirect into the appropriate step.

### Explicit pre-check phase

Training and inference forms run `POST /api/pre-check` before enabling their execution controls. The report lists dataset, model, checkpoint, GPU, FlashAttention 3, bitsandbytes, Triton, storage, and write-access checks. Changing a form input invalidates the prior report, and the backend repeats the same checks at submission time.

## Current Frontend Notes

- `HF_TOKEN` is reported only as configured/not configured. It is never accepted, persisted, or displayed by the browser.
- Hadoop can appear as download-complete but preparation-blocked when no approved `Content`/`Label` source exists. This is intentional: the UI does not fabricate supervised labels.
