# Prepare Data

`mlcore/prepareData/` contains the corpus-construction code that turns verified raw log corpora into the sequence-level CSV format consumed by the controllers in `mlcore/engine/`. `pipeline.py` is the API-safe dispatcher used by the Run Model workflow; the older scripts remain useful provenance references for their original preparation recipes.

If `mlcore/engine/` is the runtime execution layer, `prepareData/` is the corpus-construction layer that makes the runtime possible.

## Output Contract

Most scripts here produce CSV files with the contract expected by the current model runtime:

- `Content`: one sequence of log lines joined by ` ;-; `
- `Label`: `0` for normal and `1` for anomalous
- split files such as `train.csv`, `validation.csv`, and `test.csv`

The current training controller still creates its own 90/10 train-validation split from `train.csv`, but the explicit validation files produced here remain useful for inspection, alternate experiments, and dataset auditing.

## File Guide

| File | What it does | Why it exists |
| --- | --- | --- |
| `pipeline.py` | Provides API-safe BGL windows, HDFS_v1 BlockId sessions, Thunderbird sliced/oversampled windows, and Hadoop label-source detection. | Promotes verified preparation recipes into background jobs while blocking datasets that lack defensible supervised labels. |
| `helper.py` | Shared parsing helpers: log-format regex construction, chunked structuring, and fixed-window utilities. | Prevents the dataset-specific scripts from duplicating low-level parsing code. |
| `sliding_window.py` | Generic sliding-window pipeline for datasets whose dominant grouping signal is chronology. | Acts as the baseline recipe for sequence-first corpora such as BGL. |
| `sliding_window_liberty.py` | Liberty-specific preprocessing with chronological slicing, anomaly-ratio control, and a balanced training source. | Liberty needs custom rebalancing and a multi-step workflow rather than a generic window pass. |
| `sliding_window_thunderbird.py` | Thunderbird-specific sliced parser plus anomaly oversampling. | The raw corpus is large enough that line slicing and dataset-specific balancing need to be explicit. |
| `session_window.py` | HDFS sessionization by `BlockId` with labels derived from `anomaly_label.csv`. | HDFS is session-first rather than purely chronological, so it needs a different grouping strategy. |
| `session_window_sampler.py` | Sampled HDFS variant for quicker local runs and experimentation. | Makes it easier to debug preprocessing or training behavior without rebuilding the full corpus. |
| `process_apache_logs.py` | Apache parser that builds fixed windows and emits normal-only sessions. | Useful for baseline, smoke-test, or alternate log-format experiments. |

## Why The Scripts Look Dataset-Specific

These files intentionally preserve concrete preparation recipes, including:

- absolute or environment-specific paths
- fixed line ranges into huge raw logs
- dataset-specific window sizes and step sizes
- custom balancing or oversampling rules

That is not accidental. In anomaly-detection work, preprocessing is part of the experiment definition. A generic abstraction would hide important assumptions that directly affect model quality.

## Why There Are Multiple Preparation Styles

### Sequence-first datasets

BGL, Liberty, and Thunderbird are modeled as fixed-size chronological windows because the order of messages is the main grouping signal.

### Session-first datasets

HDFS_v1 groups by `BlockId` because the semantic session boundary is stronger than a simple sliding time window. The API implementation stages block-message associations in temporary SQLite storage rather than holding uncontrolled raw corpus state in RAM.

### Custom balancing pipelines

Liberty and Thunderbird use more aggressive balancing logic because their raw anomaly distributions are not good default training inputs.

Thunderbird retains the repository&apos;s documented default: lines 160,000,000 through 170,000,000, 100-line windows, and 10x anomaly oversampling. The API exposes these as constrained advanced settings instead of silently changing the experiment definition.

## Relationship To The Rest Of The Monorepo

- `backend/api/routes/data_prep.py` invokes `pipeline.py` for BGL, HDFS_v1, Thunderbird, and conditionally Hadoop preparation jobs.
- The frontend selects a dataset and displays the background job state, but does not implement parsing rules itself.
- The controllers in `mlcore/engine/` assume the resulting CSV schema already exists.

Hadoop remains intentionally guarded: it can be provisioned and inspected, but supervised preparation proceeds only when an approved source with `Content` and `Label` columns is discovered. No labels are fabricated from raw Hadoop logs.