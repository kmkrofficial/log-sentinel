# Utils

The `utils/` directory contains the shared infrastructure that the rest of LogSentinel depends on. These modules are deliberately not tied to one page or one controller. They exist to keep cross-cutting concerns isolated from the training and inference flow.

This folder is the reason the rest of the codebase can stay relatively focused: controllers can orchestrate runs, pages can render UI, and the helper modules here take care of persistence, plotting, model loading, text normalization, and session state.

## Module Guide

| File | Responsibility |
| --- | --- |
| `data_loader.py` | Normalizes raw log text with pattern replacement and defines a simple in-memory dataset abstraction. |
| `database_manager.py` | Owns SQLite schema creation plus run creation, updates, and lookup queries. |
| `global_state.py` | Wraps `st.session_state` so training and inference status survive Streamlit reruns. |
| `helpers.py` | Lightweight utility functions for sequence flattening and time formatting. |
| `log_visualizer.py` | Generates confusion matrices, ROC curves, PR curves, loss plots, and resource charts. |
| `model_loader.py` | Loads the quantized Llama backbone and tokenizer with the repository's preferred inference and training settings. |
| `resource_monitor.py` | Samples CPU, RAM, GPU utilization, VRAM, power, and clocks during a run. |
| `ui_helpers.py` | Shared Streamlit rendering helpers and dataset-discovery logic. |

## Why These Concerns Live Here

### Persistence is infrastructure, not workflow

`database_manager.py` does not decide when a run should exist or what training means. It only offers the persistence contract. That keeps `engine/` free to focus on orchestration while the database layer stays simple and replaceable.

### Model loading is centralized for consistency

`model_loader.py` is one of the most important utilities in the project. It ensures the Llama backbone is always loaded with the same quantization and tokenizer rules. Without that centralization, training and inference could quietly diverge.

### Monitoring and visualization are first-class features

The project does not treat metrics and resource usage as debugging extras. `resource_monitor.py` and `log_visualizer.py` exist because comparing runs requires both model quality and hardware behavior. That is especially important for a workstation-oriented application that cares about VRAM constraints.

### Streamlit state is intentionally wrapped

Using `st.session_state` directly everywhere would make the page layer noisy and repetitive. `global_state.py` creates one place to manage status flags, queues, logs, and progress values, which makes the UI behavior easier to reason about.

### Text normalization is shared because it changes model behavior

`data_loader.py` replaces volatile patterns such as IPs, paths, and booleans with placeholders. That preprocessing step is not cosmetic; it shapes the token distribution that both the encoder and downstream classifier see. Keeping it in a shared module guarantees the same normalization logic can be reused across workflows.

## How `utils/` Supports The Architecture

- `pages/` uses `global_state.py`, `ui_helpers.py`, and `database_manager.py` to stay small.
- `engine/` uses `database_manager.py`, `resource_monitor.py`, `log_visualizer.py`, `helpers.py`, and `data_loader.py` to avoid reimplementing support code.
- `logsentinel_model.py` depends on `model_loader.py` so the backbone setup is consistent everywhere.

In effect, `utils/` is the stability layer of the repository. It absorbs the operational details so the domain logic can stay readable.