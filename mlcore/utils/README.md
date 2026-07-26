# Utils

`mlcore/utils/` contains the reusable support modules that the ML runtime depends on. These files do not own job orchestration, HTTP behavior, or frontend state. They exist to centralize lower-level concerns such as text normalization, model loading, telemetry capture, and small helper transforms so that the controllers and model code stay readable.

## File Guide

| File | What it does | Why it exists |
| --- | --- | --- |
| `data_loader.py` | Normalizes raw log text with placeholder replacement and defines an in-memory dataset helper. | Keeps sequence preprocessing consistent across workflows. |
| `helpers.py` | Provides utilities such as `merge_data()`, `format_time()`, and ETA calculation. | Prevents orchestration code from being cluttered with generic helper logic. |
| `model_loader.py` | Loads the quantized Llama backbone and tokenizer with the repository's preferred settings. | Centralizes a fragile and hardware-sensitive model-loading path. |
| `resource_monitor.py` | Samples CPU, RAM, GPU utilization, VRAM, power, and clocks in a background thread. | Makes runtime telemetry a first-class artifact rather than an afterthought. |
| `__init__.py` | Package marker. | Allows the folder to be imported as a normal Python package. |

## Why These Concerns Live Here

### Text normalization is shared because it changes model behavior

`data_loader.py` replaces volatile structures such as IPs, file paths, and literal values with placeholders. That is not just cleanup. It changes the token distribution seen by the sentence encoder and ultimately the classifier, so the rule set needs one canonical home.

### Model loading needs one source of truth

Quantized Llama loading is sensitive to tokenizer behavior, attention implementation, device mapping, and train-mode settings. `model_loader.py` exists so training and inference do not silently diverge.

### Resource monitoring belongs beside the ML runtime

The backend stores summary telemetry in SQLite, but the measurements themselves are produced here because they are tightly coupled to the lifetime of a training or inference process.

## What Is No Longer Here

The older Streamlit layout stored UI state and database helpers in a `utils/` folder. In the current monorepo:

- persistence moved to `backend/utils/`
- UI state and rendering live in `frontend/src/`
- ML-specific helpers remain here in `mlcore/utils/`

That split is intentional and reflects the current separation between API, runtime, and browser concerns.