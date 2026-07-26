import csv
import hashlib
import os
import re
import shutil
import sqlite3
from collections import deque
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from mlcore.config import DATA_DIR
from mlcore.setup_catalog import get_dataset_spec
from mlcore.setup_manager import locate_raw_file, record_preparation_result, resource_lock


ProgressCallback = Callable[[dict[str, Any]], None]

_CONTENT_SEPARATOR = " ;-; "
_WINDOW_SIZE = 100
_STEP_SIZE = 100
_PROGRESS_INTERVAL = 100_000
_TRAIN_RATIO = 0.8
_VALIDATION_RATIO = 0.1
_THUNDERBIRD_DEFAULTS = {
    "start_line": 160_000_000,
    "end_line": 170_000_000,
    "window_size": 100,
    "step_size": 100,
    "oversampling_factor": 10,
}


class PreparationBlockedError(ValueError):
    """Raised when source data exists but cannot safely become supervised data."""


def _emit(callback: ProgressCallback | None, **payload: Any) -> None:
    if callback is not None:
        callback(payload)


def _dataset_dir(dataset_id: str) -> Path:
    return DATA_DIR / dataset_id


def _parse_bgl_line(raw_line: str) -> tuple[str, int] | None:
    fields = raw_line.rstrip("\r\n").split(maxsplit=9)
    if len(fields) != 10:
        return None
    return fields[9], int(fields[0] != "-")


def _parse_thunderbird_line(raw_line: str) -> tuple[str, int] | None:
    fields = raw_line.rstrip("\r\n").split(maxsplit=8)
    if len(fields) != 9:
        return None
    return fields[8], int(fields[0] != "-")


def _stable_split(value: str) -> str:
    bucket = int(hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest(), 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _normalize_label(value: Any) -> int | None:
    normalized = str(value).strip().casefold()
    if normalized in {"0", "normal", "-", "false", "benign"}:
        return 0
    if normalized in {"1", "anomaly", "anomalous", "abnormal", "true", "error", "alert"}:
        return 1
    return None


class _SplitWriter:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.temp_path = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
        self.file = self.temp_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["Content", "Label"])
        self.sequence_count = 0
        self.label_counts = {"normal": 0, "anomaly": 0}

    def write(self, content: str, label: int) -> None:
        self.writer.writerow([content, label])
        self.sequence_count += 1
        self.label_counts["anomaly" if label else "normal"] += 1

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()

    def commit(self) -> None:
        os.replace(self.temp_path, self.output_path)

    def discard(self) -> None:
        self.temp_path.unlink(missing_ok=True)


class _SplitWriters:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.writers = {split_name: _SplitWriter(dataset_dir / f"{split_name}.csv") for split_name in ("train", "validation", "test")}

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()

    def commit(self) -> None:
        self.close()
        for writer in self.writers.values():
            writer.commit()

    def discard(self) -> None:
        self.close()
        for writer in self.writers.values():
            writer.discard()

    def summary(self) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
        return (
            {split_name: writer.sequence_count for split_name, writer in self.writers.items()},
            {split_name: writer.label_counts for split_name, writer in self.writers.items()},
        )


def _count_lines(source_path: Path, callback: ProgressCallback | None, dataset_name: str) -> int:
    line_count = 0
    with source_path.open("r", encoding="latin-1") as source_file:
        for line_count, _ in enumerate(source_file, start=1):
            if line_count % _PROGRESS_INTERVAL == 0:
                _emit(
                    callback,
                    status=f"Counting {dataset_name} log lines",
                    progress=0.02,
                    log=f"Counted {line_count:,} {dataset_name} log lines.",
                )
    return line_count


def _finalize_result(
    dataset_id: str,
    source_path: Path,
    writers: _SplitWriters,
    callback: ProgressCallback | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sequence_counts, label_counts = writers.summary()
    result = {
        "status": "prepared",
        "dataset_name": dataset_id,
        "source_path": str(source_path),
        "sequence_counts": sequence_counts,
        "label_counts": label_counts,
        "output_paths": {split_name: str(_dataset_dir(dataset_id) / f"{split_name}.csv") for split_name in sequence_counts},
        **(extra or {}),
    }
    writers.commit()
    record_preparation_result(dataset_id, result)
    _emit(
        callback,
        status=f"{dataset_id} preparation complete",
        progress=0.97,
        metrics=sequence_counts,
        log=(
            "Created "
            + ", ".join(f"{split_name}.csv ({count:,} sequences)" for split_name, count in sequence_counts.items())
            + "."
        ),
    )
    return result


def _prepare_bgl_dataset(callback: ProgressCallback | None) -> dict[str, Any]:
    dataset_id = "BGL"
    dataset_dir = _dataset_dir(dataset_id)
    source_path = locate_raw_file(dataset_id, "BGL.log")
    if source_path is None:
        raise FileNotFoundError("BGL source log was not found in datasets/BGL/raw or datasets/BGL.")

    _emit(callback, status="Counting BGL log lines", progress=0.0, log="Counting BGL source rows.")
    total_lines = _count_lines(source_path, callback, dataset_id)
    if total_lines < _WINDOW_SIZE:
        raise ValueError(f"BGL source has {total_lines} lines; at least {_WINDOW_SIZE} are required to create a sequence.")

    train_boundary = int(total_lines * _TRAIN_RATIO)
    validation_boundary = int(total_lines * (_TRAIN_RATIO + _VALIDATION_RATIO))
    writers = _SplitWriters(dataset_dir)
    skipped_lines = 0
    buffers = {split_name: deque() for split_name in writers.writers}

    _emit(
        callback,
        status="Preparing BGL sequences",
        progress=0.05,
        log="Creating chronological 80/10/10 splits with 100-line windows.",
    )

    try:
        with source_path.open("r", encoding="latin-1") as source_file:
            for line_index, raw_line in enumerate(source_file):
                if line_index % _PROGRESS_INTERVAL == 0:
                    _emit(
                        callback,
                        status="Preparing BGL sequences",
                        progress=0.05 + (0.9 * (line_index / total_lines)),
                        log=f"Processed {line_index:,} of {total_lines:,} BGL log lines.",
                    )

                parsed_line = _parse_bgl_line(raw_line)
                if parsed_line is None:
                    skipped_lines += 1
                    continue

                if line_index < train_boundary:
                    split_name = "train"
                elif line_index < validation_boundary:
                    split_name = "validation"
                else:
                    split_name = "test"

                buffer = buffers[split_name]
                buffer.append(parsed_line)
                if len(buffer) < _WINDOW_SIZE:
                    continue

                writers.writers[split_name].write(
                    _CONTENT_SEPARATOR.join(content for content, _ in buffer),
                    max(label for _, label in buffer),
                )
                for _ in range(min(_STEP_SIZE, len(buffer))):
                    buffer.popleft()

        return _finalize_result(
            dataset_id,
            source_path,
            writers,
            callback,
            {"total_source_lines": total_lines, "skipped_source_lines": skipped_lines, "strategy": "bgl_window"},
        )
    except Exception:
        writers.discard()
        raise


def _load_hdfs_labels(label_path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with label_path.open("r", encoding="utf-8", newline="") as label_file:
        for row in csv.DictReader(label_file):
            block_id = row.get("BlockId")
            label = _normalize_label(row.get("Label"))
            if block_id and label is not None:
                labels[block_id] = label
    if not labels:
        raise PreparationBlockedError(f"No usable HDFS labels were found in {label_path.name}.")
    return labels


def _hdfs_message_content(raw_line: str) -> str:
    line = raw_line.rstrip("\r\n")
    return line.split(": ", 1)[1] if ": " in line else line


def _prepare_hdfs_v1_dataset(callback: ProgressCallback | None) -> dict[str, Any]:
    dataset_id = "HDFS_v1"
    dataset_dir = _dataset_dir(dataset_id)
    source_path = locate_raw_file(dataset_id, "HDFS.log")
    label_path = locate_raw_file(dataset_id, "anomaly_label.csv")
    if source_path is None or label_path is None:
        raise FileNotFoundError("HDFS_v1 requires HDFS.log and anomaly_label.csv in the extracted raw assets.")

    labels = _load_hdfs_labels(label_path)
    database_path = dataset_dir / f".hdfs-sessions-{uuid4().hex}.sqlite"
    source_size = source_path.stat().st_size
    writers = _SplitWriters(dataset_dir)
    inserted_messages = 0
    skipped_lines = 0
    last_reported_messages = 0
    connection: sqlite3.Connection | None = None

    _emit(callback, status="Indexing HDFS_v1 sessions", progress=0.02, log="Grouping HDFS messages by BlockId in temporary storage.")
    try:
        connection = sqlite3.connect(database_path)
        with connection:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("CREATE TABLE messages (block_id TEXT NOT NULL, sequence_index INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL)")
            message_batch: list[tuple[str, str]] = []

            with source_path.open("r", encoding="latin-1") as source_file:
                while raw_line := source_file.readline():
                    block_ids = set(re.findall(r"blk_-?\d+", raw_line))
                    if not block_ids:
                        skipped_lines += 1
                        continue
                    content = _hdfs_message_content(raw_line)
                    message_batch.extend((block_id, content) for block_id in block_ids)
                    if len(message_batch) >= 10_000:
                        connection.executemany("INSERT INTO messages (block_id, content) VALUES (?, ?)", message_batch)
                        inserted_messages += len(message_batch)
                        message_batch.clear()
                    if inserted_messages - last_reported_messages >= _PROGRESS_INTERVAL:
                        _emit(
                            callback,
                            status="Indexing HDFS_v1 sessions",
                            progress=0.02 + (0.48 * min(1.0, source_file.tell() / source_size)),
                            log=f"Indexed {inserted_messages:,} HDFS block-message associations.",
                        )
                        last_reported_messages = inserted_messages

            if message_batch:
                connection.executemany("INSERT INTO messages (block_id, content) VALUES (?, ?)", message_batch)
                inserted_messages += len(message_batch)
            connection.execute("CREATE INDEX messages_block_index ON messages(block_id, sequence_index)")

            _emit(callback, status="Writing HDFS_v1 splits", progress=0.55, log="Writing deterministic 80/10/10 block-session splits.")
            cursor = connection.execute("SELECT block_id, content FROM messages ORDER BY block_id, sequence_index")
            current_block_id: str | None = None
            current_messages: list[str] = []
            emitted_sessions = 0
            last_reported_sessions = 0

            def flush_session() -> None:
                nonlocal emitted_sessions, current_messages
                if current_block_id is None or not current_messages:
                    return
                label = labels.get(current_block_id)
                if label is None:
                    return
                split_name = _stable_split(current_block_id)
                writers.writers[split_name].write(_CONTENT_SEPARATOR.join(current_messages), label)
                emitted_sessions += 1

            for block_id, content in cursor:
                if current_block_id is not None and block_id != current_block_id:
                    flush_session()
                    current_messages = []
                current_block_id = block_id
                current_messages.append(content)
                if emitted_sessions - last_reported_sessions >= 10_000:
                    _emit(
                        callback,
                        status="Writing HDFS_v1 splits",
                        progress=0.55 + (0.37 * min(1.0, emitted_sessions / max(1, len(labels)))),
                        log=f"Wrote {emitted_sessions:,} labeled HDFS sessions.",
                    )
                    last_reported_sessions = emitted_sessions
            flush_session()

        return _finalize_result(
            dataset_id,
            source_path,
            writers,
            callback,
            {
                "strategy": "hdfs_block",
                "label_source": str(label_path),
                "labelled_blocks": len(labels),
                "indexed_messages": inserted_messages,
                "skipped_source_lines": skipped_lines,
            },
        )
    except Exception:
        writers.discard()
        raise
    finally:
        if connection is not None:
            connection.close()
        database_path.unlink(missing_ok=True)


def _thunderbird_options(options: dict[str, Any]) -> dict[str, int]:
    merged = {**_THUNDERBIRD_DEFAULTS, **{key: value for key, value in options.items() if value is not None}}
    normalized = {key: int(value) for key, value in merged.items()}
    if normalized["start_line"] < 0 or normalized["end_line"] <= normalized["start_line"]:
        raise ValueError("Thunderbird end_line must be greater than a non-negative start_line.")
    if normalized["end_line"] - normalized["start_line"] > 20_000_000:
        raise ValueError("Thunderbird preparation is limited to a 20,000,000-line slice per job.")
    if not 1 <= normalized["window_size"] <= 1_024 or not 1 <= normalized["step_size"] <= normalized["window_size"]:
        raise ValueError("Thunderbird window_size must be 1-1024 and step_size must be 1-window_size.")
    if not 1 <= normalized["oversampling_factor"] <= 20:
        raise ValueError("Thunderbird oversampling_factor must be between 1 and 20.")
    return normalized


def _prepare_thunderbird_dataset(callback: ProgressCallback | None, options: dict[str, Any]) -> dict[str, Any]:
    dataset_id = "Thunderbird"
    dataset_dir = _dataset_dir(dataset_id)
    source_path = locate_raw_file(dataset_id, "Thunderbird.log")
    if source_path is None:
        raise FileNotFoundError("Thunderbird.log was not found in the extracted raw assets.")

    config = _thunderbird_options(options)
    start_line = config["start_line"]
    end_line = config["end_line"]
    writers = _SplitWriters(dataset_dir)
    window: deque[tuple[str, int]] = deque()
    skipped_lines = 0
    sequence_index = 0

    _emit(
        callback,
        status="Preparing Thunderbird sequences",
        progress=0.02,
        log=(
            f"Processing lines {start_line:,}-{end_line:,} with {config['window_size']}-line windows "
            f"and {config['oversampling_factor']}x anomaly oversampling."
        ),
    )

    try:
        with source_path.open("r", encoding="latin-1") as source_file:
            for _ in range(start_line):
                if not source_file.readline():
                    raise ValueError(f"Thunderbird source ended before requested start_line {start_line:,}.")

            for line_index in range(start_line, end_line):
                raw_line = source_file.readline()
                if not raw_line:
                    break
                if (line_index - start_line) % _PROGRESS_INTERVAL == 0:
                    _emit(
                        callback,
                        status="Preparing Thunderbird sequences",
                        progress=0.02 + (0.9 * ((line_index - start_line) / (end_line - start_line))),
                        log=f"Processed {line_index - start_line:,} of {end_line - start_line:,} Thunderbird lines.",
                    )

                parsed_line = _parse_thunderbird_line(raw_line)
                if parsed_line is None:
                    skipped_lines += 1
                    continue
                window.append(parsed_line)
                if len(window) < config["window_size"]:
                    continue

                label = max(item_label for _, item_label in window)
                content = _CONTENT_SEPARATOR.join(message for message, _ in window)
                copies = config["oversampling_factor"] if label else 1
                for replica in range(copies):
                    split_name = _stable_split(f"{sequence_index}:{replica}")
                    writers.writers[split_name].write(content, label)
                sequence_index += 1
                for _ in range(min(config["step_size"], len(window))):
                    window.popleft()

        return _finalize_result(
            dataset_id,
            source_path,
            writers,
            callback,
            {"strategy": "thunderbird_window", "options": config, "skipped_source_lines": skipped_lines},
        )
    except Exception:
        writers.discard()
        raise


def _hadoop_labeled_csv_source(dataset_dir: Path) -> Path | None:
    raw_dir = dataset_dir / "raw"
    search_root = raw_dir if raw_dir.is_dir() else dataset_dir
    for candidate in sorted(search_root.rglob("*.csv")):
        try:
            with candidate.open("r", encoding="utf-8", newline="") as candidate_file:
                headers = {header.casefold() for header in next(csv.reader(candidate_file), [])}
        except (OSError, UnicodeDecodeError, csv.Error):
            continue
        if {"content", "label"}.issubset(headers):
            return candidate
    return None


def _prepare_hadoop_dataset(callback: ProgressCallback | None) -> dict[str, Any]:
    dataset_id = "Hadoop"
    dataset_dir = _dataset_dir(dataset_id)
    source_path = _hadoop_labeled_csv_source(dataset_dir)
    if source_path is None:
        raise PreparationBlockedError(
            "Hadoop raw assets were found, but no approved CSV with Content and Label columns was detected. "
            "Provide a compatible label source before supervised preparation."
        )

    writers = _SplitWriters(dataset_dir)
    skipped_rows = 0
    _emit(callback, status="Preparing Hadoop sequences", progress=0.05, log=f"Using labeled source {source_path.name}.")
    try:
        with source_path.open("r", encoding="utf-8", newline="") as source_file:
            reader = csv.DictReader(source_file)
            header_map = {header.casefold(): header for header in reader.fieldnames or []}
            content_key = header_map["content"]
            label_key = header_map["label"]
            for row_index, row in enumerate(reader):
                content = str(row.get(content_key, "")).strip()
                label = _normalize_label(row.get(label_key))
                if not content or label is None:
                    skipped_rows += 1
                    continue
                writers.writers[_stable_split(str(row_index))].write(content, label)
                if row_index and row_index % _PROGRESS_INTERVAL == 0:
                    _emit(callback, status="Preparing Hadoop sequences", progress=0.5, log=f"Processed {row_index:,} labeled Hadoop rows.")

        return _finalize_result(
            dataset_id,
            source_path,
            writers,
            callback,
            {"strategy": "hadoop_detect", "skipped_rows": skipped_rows},
        )
    except Exception:
        writers.discard()
        raise


def get_preparation_status(dataset_name: str) -> dict[str, Any]:
    spec = get_dataset_spec(dataset_name)
    try:
        if spec.preparation_strategy == "bgl_window":
            source = locate_raw_file(spec.dataset_id, "BGL.log")
            if source is None:
                raise FileNotFoundError("BGL.log is missing.")
        elif spec.preparation_strategy == "hdfs_block":
            if locate_raw_file(spec.dataset_id, "HDFS.log") is None or locate_raw_file(spec.dataset_id, "anomaly_label.csv") is None:
                raise FileNotFoundError("HDFS.log or anomaly_label.csv is missing.")
        elif spec.preparation_strategy == "thunderbird_window":
            if locate_raw_file(spec.dataset_id, "Thunderbird.log") is None:
                raise FileNotFoundError("Thunderbird.log is missing.")
        elif spec.preparation_strategy == "hadoop_detect" and _hadoop_labeled_csv_source(_dataset_dir(spec.dataset_id)) is None:
            raise PreparationBlockedError("No approved Hadoop CSV with Content and Label columns was detected.")
        return {"dataset_name": spec.dataset_id, "eligible": True, "strategy": spec.preparation_strategy, "blocker": None}
    except PreparationBlockedError as error:
        return {"dataset_name": spec.dataset_id, "eligible": False, "strategy": spec.preparation_strategy, "blocker": str(error)}
    except FileNotFoundError as error:
        return {"dataset_name": spec.dataset_id, "eligible": False, "strategy": spec.preparation_strategy, "blocker": str(error)}


def validate_dataset_preparation(dataset_name: str, options: dict[str, Any] | None = None) -> str:
    normalized_name = dataset_name.strip()
    status = get_preparation_status(normalized_name)
    if not status["eligible"]:
        raise PreparationBlockedError(status["blocker"] or f"{normalized_name} is not ready for preparation.")
    if normalized_name == "Thunderbird":
        _thunderbird_options(options or {})
    return normalized_name


def prepare_dataset(
    dataset_name: str,
    callback: ProgressCallback | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_name = validate_dataset_preparation(dataset_name, options)
    with resource_lock(f"dataset:{normalized_name}"):
        if normalized_name == "BGL":
            return _prepare_bgl_dataset(callback)
        if normalized_name == "HDFS_v1":
            return _prepare_hdfs_v1_dataset(callback)
        if normalized_name == "Thunderbird":
            return _prepare_thunderbird_dataset(callback, options or {})
        if normalized_name == "Hadoop":
            return _prepare_hadoop_dataset(callback)
    raise ValueError(f"No preparation strategy is configured for {normalized_name}.")