# Changelog

All notable repository-level documentation changes are recorded here.

## 2026-07-25

### Added

- Added the complete checksum-backed setup workflow for BGL, HDFS_v1, Hadoop, and Thunderbird archives.
- Added the cross-platform `scripts/bootstrap.py` entrypoint, setup inventory/provisioning API, and persistent setup-job state in the browser.
- Added the gated Run Model workflow and Results/current-activity interface.
- Added catalog-aware BGL, HDFS_v1, Thunderbird, and Hadoop-blocked preparation behavior with regression coverage.

### Fixed

- Repaired the root README conflict and updated public setup, workflow, API, and validation guidance.
- Prevented stale prepared splits after forced raw-asset replacement.
- Throttled HDFS preparation progress reporting to real batching/session milestones.

## 2026-07-09

### Added

- Added package-level README files for `backend/`, `mlcore/`, and `scripts/`.
- Added legacy-note README files for the retired root-level `engine/` and `utils/` folders.
- Added a training and inference pre-check phase with a structured backend report and browser-side requirement panel.
- Added the checksum-backed setup catalog, shared provisioning manager, cross-platform bootstrap CLI, and setup API.
- Added the unified Run Model stepper and Results/current-activity UI.
- Added API-safe HDFS_v1 and Thunderbird preparation plus a Hadoop auto-detect-or-block policy.

### Changed

- Rewrote the root README around the current backend + mlcore + frontend monorepo architecture.
- Replaced stale frontend template documentation with repository-specific frontend docs.
- Reworked the `mlcore` folder documentation to describe the current API-driven runtime instead of the retired Streamlit layout.
- Replaced the stale root `pages/` documentation with a legacy note that points to the active UI.

### Fixed

- Resolved the root README merge conflict.
- Resolved the merge conflict markers in `backend/requirements.txt` so setup instructions point to a valid dependency manifest.
- Corrected `mlcore.config` runtime paths to use the repository-level dataset, model, execution, and cache directories, allowing the API to discover `datasets/BGL`.
- Added the BGL `POST /api/data-prep` background job flow, which streams `BGL.log` into `train.csv`, `validation.csv`, and `test.csv` without loading the raw corpus into memory.
- Switched the model loader from the obsolete FlashAttention 2 selector to FlashAttention 3 and updated the optional setup-wheel discovery accordingly.
- Blocked training and inference launches when required pre-checks fail, while returning the full report through the API for remediation in the web app.
- Replaced the BGL-only preparation gate with catalog-aware setup and preparation eligibility checks.