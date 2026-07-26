# Scripts

The `scripts/` folder contains the bootstrap and process-management helpers used to prepare and run the monorepo on a local workstation. The asset-provisioning core is cross-platform Python; PowerShell remains a Windows convenience wrapper for virtual-environment and process management.

## File Guide

| File | What it does | Why it exists |
| --- | --- | --- |
| `bootstrap.py` | Cross-platform setup CLI for dependency installation, status audits, checksum-backed Zenodo dataset provisioning, and model provisioning through server-side `HF_TOKEN`. | Keeps GUI, CLI, and PowerShell setup behavior on one implementation path. |
| `common.ps1` | Defines shared paths, runtime-state storage, process detection, log locations, and helper functions used by the other scripts. | Centralizes all process-management behavior so the command wrappers stay small. |
| `setup.ps1` | Creates or reuses `.venv`, verifies Python 3.11, installs dependencies, optionally installs a local Flash Attention wheel, and delegates model/dataset provisioning to `bootstrap.py`. | Standardizes Windows workstation setup without duplicating asset logic. |
| `start.ps1` | Starts the backend, the frontend, or both and records process metadata in `.runtime/process-state.json`. | Gives the repository a repeatable launch path instead of ad hoc terminal commands. |
| `stop.ps1` | Stops managed backend and/or frontend processes using the stored state file. | Keeps local process cleanup explicit and consistent. |
| `restart.ps1` | Stops then relaunches the selected targets. | Speeds up iterative development when either side needs to be bounced. |

## Why These Scripts Exist

- The project uses both Python and Node tooling, so local setup is multi-step.
- The ML environment has GPU-specific extras such as Flash Attention and a gated Llama model that requires `HF_TOKEN` on the backend process.
- Dataset archives need size and checksum validation before extraction because Thunderbird is roughly 2 GB compressed.
- The backend and frontend are long-running dev servers that benefit from a lightweight process ledger.
- The team workflow appears to be Windows-heavy, so PowerShell is the lowest-friction automation layer in this repository.

## Runtime State

`common.ps1` manages a `.runtime/` directory under `scripts/` that stores:

- process metadata for managed backend/frontend instances
- stdout and stderr log files for launched services

That state file is why `stop.ps1` and `restart.ps1` can operate on processes started earlier without the user manually tracking PIDs.

## Bootstrap Examples

Inspect all asset and runtime state without changing files:

```bash
python scripts/bootstrap.py --check
```

Download, checksum-verify, and safely extract all requested datasets:

```bash
python scripts/bootstrap.py --datasets all
```

Provision the default encoder and gated Llama model. Set `HF_TOKEN` in the backend environment first; the token is never accepted or stored by the browser.

```bash
export HF_TOKEN="..."
python scripts/bootstrap.py --models
```

Install application-level Python and frontend dependencies, then provision all assets:

```bash
python scripts/bootstrap.py --install-python-deps --install-frontend-deps --all
```

`--force` replaces managed raw assets or model snapshots. Replacing raw assets invalidates generated train/validation/test CSV files so they cannot become stale relative to their source archive.

## Important Operational Detail

The backend launches ML jobs inside the backend process itself. That means stopping the backend also kills any active training or inference jobs running through the API. `common.ps1` surfaces this explicitly when it starts the backend, and the docs should treat that as an intentional limitation of the current lightweight orchestration model.