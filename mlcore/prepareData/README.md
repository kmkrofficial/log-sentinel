# Prepare Data

The `prepareData/` directory contains the offline scripts that turn raw log corpora into model-ready CSV files. These scripts are not part of the Streamlit runtime. They exist because real log datasets differ in structure, labels, and scale, and those differences are important enough that the preparation logic should stay explicit.

In practice, this folder captures the research and operations layer of the project: how raw BGL, Liberty, Thunderbird, HDFS, or Apache logs were parsed, sessionized, windowed, balanced, and finally written into the format consumed by the runtime.

## Output Contract

Most scripts here emit CSV files with the same application-facing contract:

- `Content`: one sequence of log lines joined by ` ;-; `
- `Label`: `0` for normal, `1` for anomalous
- dataset files such as `train.csv`, `validation.csv`, and `test.csv`

One important implementation detail: the current training runtime still creates its own validation split from `train.csv`, even though several scripts here also produce `validation.csv`. That extra file is still useful for manual inspection, alternate experiments, and future runtime changes.

## Script Guide

| File | Role |
| --- | --- |
| `helper.py` | Shared parsing and windowing utilities: log-format regex generation, chunked structuring, and fixed-size window construction. |
| `sliding_window.py` | Generic chunked sliding-window pipeline for sequential datasets such as BGL, Liberty, or Thunderbird. |
| `sliding_window_liberty.py` | Liberty-specific preparation recipe with chronological slicing, anomaly-ratio control, and a balanced training source. |
| `sliding_window_thunderbird.py` | Thunderbird-specific sliced parser and oversampling workflow for a very large raw corpus. |
| `session_window.py` | HDFS sessionization pipeline that groups log lines by block ID and labels each session from `anomaly_label.csv`. |
| `session_window_sampler.py` | Sampled HDFS variant for smaller local runs and faster debugging. |
| `process_apache_logs.py` | Apache parser that creates fixed windows from raw logs and emits normal-only sessions for baseline or smoke-test use cases. |

## Why There Are Multiple Styles Of Script

### Some datasets are sequence-first

BGL, Liberty, and Thunderbird are handled with fixed-size sliding windows because the temporal order of messages is the primary grouping signal. The model sees each window as one sequence.

### Some datasets are session-first

HDFS is grouped by `BlockId`, which is a stronger semantic unit than a fixed chronological window. That is why `session_window.py` exists instead of forcing HDFS through the same logic as the other datasets.

### Some datasets need custom resampling

`sliding_window_liberty.py` and `sliding_window_thunderbird.py` go beyond simple parsing. They introduce dataset-specific balancing and oversampling rules because the raw anomaly distribution is not always suitable for stable training.

## Why These Scripts Are Not A Generic CLI

Several scripts contain absolute paths, fixed line ranges, or dataset-specific constants. That is not accidental. These files preserve the exact preparation recipes used during experimentation on very large corpora, where changing the slice or balancing rule changes the dataset itself.

Treat them as reproducible templates rather than as polished end-user commands. When adapting them to a new environment, the expected edits are usually:

- source and destination paths
- raw log format strings
- line-range slicing for huge files
- window size and step size
- balancing or oversampling factors

## Why This Folder Matters To The Rest Of The Project

The runtime in `engine/` assumes the data has already been normalized into a sequence-level CSV format. That assumption only works because `prepareData/` makes the dataset creation step explicit and repeatable. Without this folder, the model pipeline would hide critical labeling and sessionization choices that directly affect training quality.