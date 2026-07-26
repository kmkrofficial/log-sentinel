# Engine

The `mlcore/engine/` package is the runtime orchestration layer for LogSentinel. It takes prepared datasets and configured paths from `mlcore.config`, drives the heavy training or inference workflow, emits structured progress callbacks, and writes artifact bundles that the backend and frontend can consume later.

This folder is intentionally separate from both the web API and the model definition. The backend should not know how to chunk embeddings into HDF5, and the model module should not know how to create execution directories or stop early on validation F1.

## File Guide

| File | What it does | Why it exists |
| --- | --- | --- |
| `training_controller.py` | Runs the full training lifecycle: dataset staging, embedding, train/validation split, two-phase fine-tuning, evaluation, telemetry capture, and artifact writing. | Training is the most stateful workflow in the repository and needs a dedicated orchestrator. |
| `inference_controller.py` | Rebuilds a saved model, stages test data, evaluates it, writes predictions, and emits a summary payload. | Inference shares part of the training data path but deserves a separate control flow and result contract. |
| `phase_manager.py` | Contains shared epoch loops, evaluation logic, early stopping, and per-phase checkpoint handling. | Keeps the controllers focused on orchestration rather than batch-level mechanics. |
| `data_utils.py` | Provides `HDF5Dataset`, `BalancedSampler`, and `FocalLoss`. | The engine needs memory-safe dataset access and imbalance-aware training primitives. |

## Who Calls The Engine

- `backend/api/routes/training.py` instantiates `TrainingController` inside a FastAPI background task.
- `backend/api/routes/inference.py` instantiates `InferenceController` the same way.
- `mlcore/cli.py` uses the same controllers directly for headless runs.

That call pattern is why the engine emits callback payloads instead of assuming a UI framework or a database adapter.

## Runtime Pattern

1. A caller constructs a controller with a dataset name, optional test-run flags, and an optional callback.
2. The controller creates an execution directory and begins resource monitoring.
3. Raw sequence CSVs are normalized, embedded in chunks, and saved into HDF5.
4. The controller rebuilds `LogSentinelModel`, optionally compiles it on Linux, and runs training or inference.
5. Evaluation outputs, training loss, and resource telemetry are written into `run_metrics.json`.
6. The controller returns a compact result dictionary for the caller, while richer artifacts stay on disk.

## Why Controllers Own The Workflow

The controllers are intentionally stateful because a single run needs to coordinate:

- dataset names and hyperparameters
- execution directory creation
- callback emission
- model load and cleanup cycles
- temporary HDF5 staging
- resource monitoring
- final artifact serialization

Keeping those concerns in one object makes the workflow easier to reason about and lets the backend stay thin.

## Why The Engine Is Database-Free

The engine never writes SQLite rows itself. That separation is deliberate: persistence is a backend concern, while the engine's job is to produce artifacts and structured summaries. The backend decides how those summaries map into rows, statuses, and API responses.

## Design Decisions That Show Up Here

### Two-phase training

The training controller first updates the projector and classifier, then reloads the model and enables LoRA fine-tuning. That sequence reflects the model architecture and helps avoid over-updating the backbone before the projection layer has learned a useful alignment.

### HDF5 everywhere before model execution

Both training and inference stage the dataset through HDF5 instead of feeding raw sequence strings directly into the final model path. This keeps the data path consistent, reduces peak memory pressure, and makes large datasets feasible on a single machine.

### Structured callback payloads

The callback contract uses plain dictionaries with keys such as `log`, `status`, `progress`, `metrics`, `validation_metrics`, `run_id`, and `execution_dir`. That keeps the engine agnostic about whether the caller is a backend job manager, a CLI wrapper, or a future scheduler.