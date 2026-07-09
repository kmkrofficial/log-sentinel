# Engine

The `engine/` directory contains the runtime orchestration layer for LogSentinel. It is the boundary between the Streamlit UI and the model pipeline: pages collect user intent, but controllers inside `engine/` decide how a run is created, how data is staged, when models are loaded or cleaned up, and where artifacts are written.

This separation is intentional. Training and inference are long-running workflows with cleanup, error handling, progress callbacks, and resource accounting. Keeping that logic out of `pages/` prevents the UI from becoming the real application layer.

## What Lives Here

| File | Responsibility |
| --- | --- |
| `training_controller.py` | Orchestrates an end-to-end training run: creates the run record, embeds datasets into HDF5, builds the model, executes both training phases, evaluates results, and saves artifacts. |
| `inference_controller.py` | Runs evaluation or prediction with a saved model: stages test data, rebuilds the model stack, loads fine-tuned weights, computes metrics, and writes predictions. |
| `phase_manager.py` | Holds shared training and evaluation loops, early stopping, batch-level progress reporting, and visualization hooks. |
| `data_utils.py` | Defines HDF5-backed datasets plus imbalance-handling primitives such as `BalancedSampler` and `FocalLoss`. |

## Why The Split Looks Like This

### Controllers own orchestration

The controllers are intentionally stateful. A single run needs a nickname, a database row, an execution directory, callbacks, cleanup behavior, and final summaries. Bundling those concerns into a controller object makes the workflow inspectable and keeps the page layer thin.

### `phase_manager.py` isolates the repeatable loop logic

The outer workflow in `training_controller.py` is already busy with dataset staging, model lifecycle, and artifact management. Moving the per-epoch loops, evaluation metrics, and early-stopping rules into `phase_manager.py` keeps the controller focused on orchestration rather than training math.

### `data_utils.py` exists because the runtime has two data problems

The first problem is scale: embedded log sequences can be too large to hold entirely in RAM, so `HDF5Dataset` lets the later stages stream data from disk. The second problem is imbalance: anomaly datasets are usually skewed, so `BalancedSampler` and `FocalLoss` push the runtime toward a more useful class distribution during training.

## Runtime Pattern

1. A page starts a controller in a background thread and passes a queue callback.
2. The controller creates a run record in SQLite and an execution directory under `executions/`.
3. Raw CSV content is normalized and embedded in chunks with the sentence encoder.
4. The controller writes those embeddings to HDF5 and reopens them through `HDF5Dataset`.
5. `LogSentinelModel` is created, optionally compiled on Linux, and trained or evaluated.
6. Metrics, plots, copied checkpoints, and predictions are written back into the execution directory.

## Why Training Is Two-Phase

Training starts by updating only the projector and classifier, then moves into broader LoRA-based fine-tuning. That design reduces the chance of destabilizing the backbone from the first step, and it matches the underlying model architecture: the projector has to learn how sentence-encoder embeddings should enter the Llama space before the LLM-side adapters can make full use of them.

## Why Inference Repeats The Embedding Stage

Inference still runs the encoder and writes HDF5 intermediates instead of reading raw strings directly inside the LLM path. That repetition is deliberate. It keeps the data path aligned with training, makes evaluation predictable on large datasets, and avoids having a separate code path with different batching or truncation rules.

## What This Folder Does Not Do

- It does not own Streamlit widgets or screen layout.
- It does not define raw dataset parsing rules; those live under `prepareData/`.
- It does not persist UI state; that belongs to `utils/global_state.py` and the page layer.

In short, `engine/` is the operational core of the application. If `pages/` represents the control panel, `engine/` is the machinery behind it.