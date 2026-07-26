# ML Core

`mlcore/` is the headless machine-learning package at the center of LogSentinel. It owns dataset staging, model assembly, training, inference, telemetry capture, offline data preparation, and the optional command-line interface. The backend imports it to serve jobs over HTTP, and the CLI uses it directly for local or scripted workflows.

The design goal is straightforward: the ML runtime should not care whether it is being driven by a browser, an API, or a terminal command. That is why `mlcore/` contains no FastAPI code and no frontend code.

## Package Guide

| Path | Significance | Why it exists |
| --- | --- | --- |
| `cli.py` | Headless entrypoint for `train` and `inference` subcommands. | Makes the runtime usable without the backend or frontend. |
| `config.py` | Central ML paths, default models, and dataset-specific hyperparameter overrides. | Prevents path and hyperparameter logic from fragmenting across controllers. |
| `download_models.py` | Downloads the default encoder and Llama backbone into the configured model cache. | Keeps model provisioning explicit and separate from runtime requests. |
| `setup_catalog.py` | Defines the fixed Zenodo archive metadata, raw-file contracts, preparation strategies, and default model specs. | Makes setup deterministic and prevents user-supplied download URLs from reaching the backend. |
| `setup_manager.py` | Handles resumable downloads, MD5 validation, safe ZIP/TAR extraction, raw-file discovery, model snapshots, setup state, and asset locks. | Gives GUI, CLI, and PowerShell setup paths one shared implementation. |
| `logsentinel_model.py` | Defines the hybrid model that projects sentence-encoder embeddings into Llama space and classifies the result. | This file captures the repository's main modeling idea. |
| `system_spec.py` | Reports CPU, RAM, GPU, CUDA, and driver information. | Useful for debugging, reproducibility, and benchmark context. |
| `engine/` | Training and inference orchestration layer. | Owns long-running workflows and their callback/reporting contract. |
| `utils/` | Shared ML-side support modules. | Centralizes reusable helpers such as model loading and resource monitoring. |
| `prepareData/` | Offline parsing and dataset-construction scripts. | Makes raw-log preprocessing explicit and reproducible instead of burying it inside runtime code. |

## Runtime Paths

`mlcore.config` is the current source of truth for runtime filesystem locations. At execution time it resolves datasets, models, execution outputs, and temporary caches from the repository root.

This keeps `datasets/`, `models/`, `executions/`, and `logsentinel_data/` shared across the API, CLI, and ML controllers. The package code remains under `mlcore/`, while the mutable operational data stays at the repository boundary.

## Why `mlcore/` Is A Separate Package

- The backend needs to focus on HTTP, persistence, and job lifecycle, not on tensor operations.
- The frontend should never need direct access to model code or heavyweight Python dependencies.
- A package boundary makes it easier to run training or inference from the CLI, tests, or future schedulers.
- Separating the ML runtime keeps model-specific dependencies and hardware assumptions away from the rest of the monorepo.

## Key Design Decisions

### HDF5 staging for embeddings

The controllers embed raw log sequences into HDF5 before downstream training or inference. This is a scale decision: large log corpora can exceed RAM if dense vectors stay resident in memory, and HDF5 gives the training loop random access without a full in-memory copy.

### Two-phase training

Training first adapts the projector and classifier, then performs broader LoRA-based fine-tuning. That split matches the model architecture: the sentence-encoder output has to be aligned into the Llama hidden space before low-rank backbone adaptation becomes effective.

### Quantized Llama plus LoRA

`logsentinel_model.py` uses a quantized Llama backbone and LoRA adapters to keep memory pressure practical on a single workstation. The tradeoff is additional complexity around model loading and compatibility, but it enables much larger backbones than full fine-tuning would allow on commodity GPUs.

### Dataset-specific hyperparameter overrides

The repository does not pretend all log datasets behave the same. `config.py` contains dataset-specific overrides for sequence length, embedding chunk size, learning rates, and early stopping because BGL, HDFS, Liberty, and Thunderbird have materially different scale and class-balance characteristics.

### Provisioned raw assets versus prepared splits

Setup keeps extracted source assets under `datasets/<id>/raw/`; preparation writes the controller contract (`train.csv`, `validation.csv`, `test.csv`) at `datasets/<id>/`. Replacing managed raw assets invalidates those generated splits, which prevents a stale prepared dataset from being trained after its source changes.

## Important Interfaces

- Controllers return structured dictionaries containing run status, metric summaries, resource summaries, and the execution directory.
- Controllers also emit callback payloads with fields such as `log`, `status`, `progress`, `metrics`, `validation_metrics`, `run_id`, and `execution_dir`.
- Prepared datasets are expected to expose `Content` and `Label`, with log lines joined inside `Content` by ` ;-; `.

## Subdirectory Docs

- [mlcore/engine/README.md](engine/README.md)
- [mlcore/utils/README.md](utils/README.md)
- [mlcore/prepareData/README.md](prepareData/README.md)