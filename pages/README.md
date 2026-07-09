# Pages

The `pages/` directory contains the operator-facing Streamlit screens for LogSentinel. Each file maps to a single user workflow and intentionally stays thin: the page validates inputs, starts background work, consumes queue messages, and renders status, while the heavy runtime logic lives in `engine/`.

That separation matters because Streamlit reruns the page script frequently. If model training or inference logic lived directly in the page layer, the app would be much harder to reason about and much easier to break during rerenders.

## Page Guide

| File | User workflow | Why it exists |
| --- | --- | --- |
| `Train_and_Evaluate.py` | Start a training run, monitor progress, and view live metrics and logs. | This is the default landing workflow and the main way to create new models. |
| `Inference.py` | Evaluate or score a dataset using a previously trained or externally imported model directory. | Separates post-training model usage from training-time configuration. |
| `History.py` | Browse stored runs, inspect metrics, and view generated visualizations. | Gives operators a persistent audit trail without leaving the UI. |

## Why The Pages Are Thin

### Streamlit is used as an interface layer, not as the application core

The page files import controllers, database helpers, and state helpers instead of re-implementing the actual workflows. That design lets the app use Streamlit for what it is good at, which is interaction and rendering, while keeping business logic somewhere more stable.

### Background threads prevent the UI from blocking

Training and inference are launched inside daemon threads. Progress and logs are pushed through a queue, then read back by the page during reruns. This is why the user can keep seeing status updates instead of freezing the browser tab while the model is working.

### Global state survives reruns

`utils/global_state.py` stores flags, queues, log messages, metrics, and status strings in `st.session_state`. That is the mechanism that makes the progress display feel continuous even though Streamlit keeps re-executing the page file.

## Design Notes Per Page

### `Train_and_Evaluate.py`

- Verifies that the default encoder and Llama weights are present under `models/`.
- Reads dataset options from `datasets/`.
- Surfaces hardware warnings using `system_spec.py` before starting a run.
- Starts `TrainingController` in a background thread.

This page is deliberately conservative: it checks for missing models and low-resource hardware before the expensive job begins because failures later in the pipeline are more costly.

### `Inference.py`

- Lets the user choose a model from run history or import an external model folder.
- Reuses the same queue-driven status pattern as training.
- Starts `InferenceController` with the selected dataset and model path.

The split between internal history and external import exists because operational teams often want both: quick reuse of local runs and the ability to test a model copied in from somewhere else.

### `History.py`

- Pulls rows from SQLite through `DatabaseManager`.
- Shows run-level metrics and hyperparameters.
- Displays visualizations from the saved execution directory.

This page exists because experiment management is part of the product, not an afterthought. The application does not just produce a model; it also preserves the evidence needed to compare runs later.

## Relationship To Other Folders

- `pages/` depends on `engine/` for execution.
- `pages/` depends on `utils/` for persistence, UI helpers, and shared state.
- `pages/` does not depend on `prepareData/` directly because raw log construction is considered an offline activity.

If `engine/` is the application machinery, `pages/` is the operator console built on top of it.