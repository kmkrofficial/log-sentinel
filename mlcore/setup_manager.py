"""Shared dataset and model provisioning for the CLI and FastAPI control plane."""

import fnmatch
import hashlib
import http.client
import json
import os
import shutil
import tarfile
import tempfile
import threading
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

from huggingface_hub import snapshot_download

from mlcore.config import DATA_CACHE_DIR, DATA_DIR, MODELS_DIR
from mlcore.setup_catalog import DATASET_SPECS, MODEL_SPECS, DatasetSpec, ModelSpec, get_dataset_spec, get_model_spec


ProgressCallback = Callable[[dict[str, Any]], None]
SETUP_STATE_VERSION = 1
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
PARALLEL_DOWNLOAD_THRESHOLD_BYTES = 64 * 1024 * 1024
PARALLEL_DOWNLOAD_MIN_RANGE_BYTES = 32 * 1024 * 1024
DEFAULT_DOWNLOAD_CONNECTIONS = 8
MAX_DOWNLOAD_CONNECTIONS = 16
DOWNLOAD_RANGE_MAX_ATTEMPTS = 4
DOWNLOADS_DIR = DATA_CACHE_DIR / "downloads"
SETUP_STATE_PATH = DATA_CACHE_DIR / "setup-state.json"

_LOCK_GUARD = threading.Lock()
_RESOURCE_LOCKS: dict[str, threading.Lock] = {}


class SetupError(RuntimeError):
    """Raised when a provisioned asset cannot be safely made ready."""


class ResourceBusyError(SetupError):
    """Raised when another setup or preparation job owns the same resource."""


class RetryableDownloadError(SetupError):
    """Raised when a download range can safely resume from its partial bytes."""


def _emit(callback: ProgressCallback | None, **payload: Any) -> None:
    if callback is not None:
        callback(payload)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_state() -> dict[str, Any]:
    return {"version": SETUP_STATE_VERSION, "datasets": {}, "models": {}}


def load_setup_state() -> dict[str, Any]:
    if not SETUP_STATE_PATH.is_file():
        return _default_state()

    try:
        with SETUP_STATE_PATH.open("r", encoding="utf-8") as state_file:
            state = json.load(state_file)
    except (OSError, json.JSONDecodeError):
        return _default_state()

    if not isinstance(state, dict):
        return _default_state()

    state.setdefault("version", SETUP_STATE_VERSION)
    state.setdefault("datasets", {})
    state.setdefault("models", {})
    return state


def save_setup_state(state: dict[str, Any]) -> None:
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(prefix=".setup-state-", suffix=".json", dir=DATA_CACHE_DIR)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as state_file:
            json.dump(state, state_file, indent=2, sort_keys=True)
            state_file.write("\n")
        os.replace(temporary_path, SETUP_STATE_PATH)
    except Exception:
        Path(temporary_path).unlink(missing_ok=True)
        raise


@contextmanager
def resource_lock(resource_name: str):
    with _LOCK_GUARD:
        lock = _RESOURCE_LOCKS.setdefault(resource_name, threading.Lock())

    if not lock.acquire(blocking=False):
        raise ResourceBusyError(f"Another job is already modifying {resource_name}.")

    try:
        yield
    finally:
        lock.release()


def _archive_path(spec: DatasetSpec) -> Path:
    return DOWNLOADS_DIR / spec.archive_filename


def _dataset_dir(spec: DatasetSpec) -> Path:
    return DATA_DIR / spec.dataset_id


def _download_connection_count(remaining_bytes: int) -> int:
    configured_connections = os.getenv("LOGSENTINEL_DOWNLOAD_CONNECTIONS", str(DEFAULT_DOWNLOAD_CONNECTIONS))
    try:
        requested_connections = int(configured_connections)
    except ValueError:
        requested_connections = DEFAULT_DOWNLOAD_CONNECTIONS

    requested_connections = max(1, min(MAX_DOWNLOAD_CONNECTIONS, requested_connections))
    required_connections = max(1, (remaining_bytes + PARALLEL_DOWNLOAD_MIN_RANGE_BYTES - 1) // PARALLEL_DOWNLOAD_MIN_RANGE_BYTES)
    return min(requested_connections, required_connections)


def _raw_dir(spec: DatasetSpec) -> Path:
    return _dataset_dir(spec) / "raw"


def _checksum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as archive_file:
        for block in iter(lambda: archive_file.read(DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksum(path: Path, expected_size: int, expected_md5: str) -> None:
    if not path.is_file():
        raise SetupError(f"Archive was not found: {path}.")

    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise SetupError(f"Archive size mismatch for {path.name}: expected {expected_size:,} bytes, found {actual_size:,} bytes.")

    actual_md5 = _checksum(path)
    if actual_md5.lower() != expected_md5.lower():
        raise SetupError(f"Archive checksum mismatch for {path.name}: expected {expected_md5}, found {actual_md5}.")


def _safe_destination(root: Path, relative_name: str) -> Path:
    destination = (root / relative_name).resolve()
    resolved_root = root.resolve()
    if destination != resolved_root and resolved_root not in destination.parents:
        raise SetupError(f"Archive entry escapes the extraction directory: {relative_name}.")
    return destination


def safe_extract_zip(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                _safe_destination(destination, member.filename).mkdir(parents=True, exist_ok=True)
                continue

            mode = member.external_attr >> 16
            if mode and (mode & 0o170000) == 0o120000:
                raise SetupError(f"Refusing symbolic link in archive: {member.filename}.")

            target = _safe_destination(destination, member.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source_file, target.open("wb") as target_file:
                shutil.copyfileobj(source_file, target_file)


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk() or member.isdev():
                raise SetupError(f"Refusing unsafe archive entry: {member.name}.")

            target = _safe_destination(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue

            source_file = archive.extractfile(member)
            if source_file is None:
                raise SetupError(f"Could not extract archive entry: {member.name}.")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source_file, target.open("wb") as target_file:
                shutil.copyfileobj(source_file, target_file)


def safe_extract_archive(spec: DatasetSpec, archive_path: Path, destination: Path) -> None:
    if spec.archive_format == "zip":
        safe_extract_zip(archive_path, destination)
        return
    if spec.archive_format == "tar.gz":
        safe_extract_tar(archive_path, destination)
        return
    raise SetupError(f"Unsupported archive format for {spec.dataset_id}: {spec.archive_format}.")


def _matches_pattern(path: Path, pattern: str) -> bool:
    return fnmatch.fnmatch(path.name.casefold(), pattern.casefold())


def discover_raw_files(root: Path, spec: DatasetSpec) -> dict[str, str]:
    if not root.is_dir():
        raise SetupError(f"Raw dataset directory was not found: {root}.")

    files = sorted(path for path in root.rglob("*") if path.is_file())
    discovered: dict[str, str] = {}

    for required_name in spec.required_raw_filenames:
        matched_path = next((path for path in files if path.name.casefold() == required_name.casefold()), None)
        if matched_path is None:
            raise SetupError(f"{spec.dataset_id} archive is missing required raw file {required_name}.")
        discovered[required_name] = str(matched_path.relative_to(root))

    for path in files:
        if any(_matches_pattern(path, pattern) for pattern in spec.raw_file_patterns):
            relative_path = str(path.relative_to(root))
            discovered.setdefault(path.name, relative_path)

    if not discovered:
        patterns = ", ".join(spec.raw_file_patterns)
        raise SetupError(f"No raw files matching {patterns} were found in the {spec.dataset_id} archive.")

    return dict(sorted(discovered.items()))


def locate_raw_file(dataset_id: str, filename: str) -> Path | None:
    spec = get_dataset_spec(dataset_id)
    candidate_roots = (_raw_dir(spec), _dataset_dir(spec))

    for candidate_root in candidate_roots:
        if not candidate_root.is_dir():
            continue
        direct_path = candidate_root / filename
        if direct_path.is_file():
            return direct_path
        matched_path = next((path for path in candidate_root.rglob("*") if path.is_file() and path.name.casefold() == filename.casefold()), None)
        if matched_path is not None:
            return matched_path
    return None


def _archive_is_verified(spec: DatasetSpec, recorded_archive: dict[str, Any] | None = None) -> bool:
    archive_path = _archive_path(spec)
    if not archive_path.is_file() or archive_path.stat().st_size != spec.expected_size:
        return False

    if recorded_archive and (
        recorded_archive.get("filename") == spec.archive_filename
        and recorded_archive.get("md5") == spec.md5
        and recorded_archive.get("expected_size") == spec.expected_size
        and recorded_archive.get("verified_at")
    ):
        return True

    try:
        verify_checksum(archive_path, spec.expected_size, spec.md5)
    except SetupError:
        return False
    return True


def _raw_is_ready(spec: DatasetSpec) -> tuple[bool, dict[str, str], str | None]:
    for root, layout in ((_raw_dir(spec), "managed"), (_dataset_dir(spec), "legacy")):
        try:
            raw_files = discover_raw_files(root, spec)
        except SetupError:
            continue
        return True, raw_files, layout
    return False, {}, None


def _prepared_outputs(spec: DatasetSpec) -> dict[str, Any]:
    dataset_dir = _dataset_dir(spec)
    outputs: dict[str, Any] = {}
    for split_name in ("train", "validation", "test"):
        output_path = dataset_dir / f"{split_name}.csv"
        outputs[split_name] = {
            "path": str(output_path),
            "ready": output_path.is_file() and output_path.stat().st_size > 0,
            "size_bytes": output_path.stat().st_size if output_path.is_file() else 0,
        }
    return outputs


def _hadoop_label_candidates(raw_files: dict[str, str]) -> list[str]:
    return sorted(
        relative_path
        for filename, relative_path in raw_files.items()
        if "label" in filename.casefold() or "anomaly" in filename.casefold()
    )


def _model_asset_status(spec: ModelSpec) -> dict[str, Any]:
    model_dir = MODELS_DIR / spec.folder_name
    config_path = model_dir / "config.json"
    weights = [model_dir / filename for filename in ("model.safetensors", "pytorch_model.bin", "model.bin")]
    weight_path = next((path for path in weights if path.is_file()), None)
    return {
        "key": spec.key,
        "model_id": spec.model_id,
        "path": str(model_dir),
        "requires_hf_token": spec.requires_hf_token,
        "ready": config_path.is_file() and weight_path is not None,
        "config_ready": config_path.is_file(),
        "weight_file": weight_path.name if weight_path else None,
    }


def scan_setup_state() -> dict[str, Any]:
    state = load_setup_state()
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    datasets: list[dict[str, Any]] = []
    for spec in DATASET_SPECS.values():
        raw_ready, raw_files, raw_layout = _raw_is_ready(spec)
        dataset_state = state["datasets"].get(spec.dataset_id, {})
        label_candidates = _hadoop_label_candidates(raw_files) if spec.preparation_strategy == "hadoop_detect" else []
        preparation_blocker = None
        if raw_ready and spec.preparation_strategy == "hadoop_detect" and not label_candidates:
            preparation_blocker = "No compatible Hadoop anomaly-label source was detected. Download is complete, but supervised preparation is blocked."

        datasets.append(
            {
                "id": spec.dataset_id,
                "display_name": spec.display_name,
                "preparation_strategy": spec.preparation_strategy,
                "archive": {
                    "filename": spec.archive_filename,
                    "format": spec.archive_format,
                    "expected_size": spec.expected_size,
                    "md5": spec.md5,
                    "cached": _archive_path(spec).is_file(),
                    "verified": _archive_is_verified(spec, dataset_state.get("archive")),
                },
                "raw": {
                    "ready": raw_ready,
                    "layout": raw_layout,
                    "path": str(_raw_dir(spec) if raw_layout == "managed" else _dataset_dir(spec)),
                    "files": raw_files,
                },
                "prepared": _prepared_outputs(spec),
                "preparation": dataset_state.get("preparation", {}),
                "label_candidates": label_candidates,
                "preparation_blocker": preparation_blocker,
                "last_error": dataset_state.get("last_error"),
            }
        )

    models = [_model_asset_status(spec) for spec in MODEL_SPECS.values()]
    disk_usage = shutil.disk_usage(DATA_CACHE_DIR)
    return {
        "checked_at": _now(),
        "hf_token_configured": bool(os.getenv("HF_TOKEN")),
        "downloads_path": str(DOWNLOADS_DIR),
        "datasets": datasets,
        "models": models,
        "storage": {
            "path": str(DATA_CACHE_DIR),
            "free_bytes": disk_usage.free,
            "total_bytes": disk_usage.total,
        },
    }


def download_with_resume(
    url: str,
    destination: Path,
    expected_size: int,
    expected_md5: str | None = None,
    callback: ProgressCallback | None = None,
    progress_start: float = 0.0,
    progress_span: float = 1.0,
    label: str = "asset",
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_name(f".{destination.name}.part")
    existing_size = partial_path.stat().st_size if partial_path.is_file() else 0

    if existing_size > expected_size:
        partial_path.unlink()
        existing_size = 0
    elif existing_size == expected_size and existing_size:
        if expected_md5 and _checksum(partial_path).lower() == expected_md5.lower():
            os.replace(partial_path, destination)
            _emit(callback, status=f"Downloading {label}", progress=progress_start + progress_span, log=f"Resumed completed verified {label} from a partial download.")
            return destination
        partial_path.unlink()
        existing_size = 0

    remaining_bytes = expected_size - existing_size
    connection_count = _download_connection_count(remaining_bytes)
    if expected_size >= PARALLEL_DOWNLOAD_THRESHOLD_BYTES and connection_count > 1:
        return _download_with_parallel_ranges(
            url=url,
            destination=destination,
            partial_path=partial_path,
            existing_size=existing_size,
            expected_size=expected_size,
            connection_count=connection_count,
            callback=callback,
            progress_start=progress_start,
            progress_span=progress_span,
            label=label,
        )

    return _download_single_stream(
        url=url,
        destination=destination,
        partial_path=partial_path,
        existing_size=existing_size,
        expected_size=expected_size,
        callback=callback,
        progress_start=progress_start,
        progress_span=progress_span,
        label=label,
    )


def _emit_download_progress(
    callback: ProgressCallback | None,
    label: str,
    expected_size: int,
    downloaded_size: int,
    progress_start: float,
    progress_span: float,
) -> None:
    percent = min(100, int((downloaded_size / expected_size) * 100)) if expected_size else 0
    _emit(
        callback,
        status=f"Downloading {label}",
        progress=progress_start + (progress_span * (percent / 100)),
        log=f"Downloading {label}: {downloaded_size / 1024**2:.1f} MiB of {expected_size / 1024**2:.1f} MiB ({percent}%).",
    )


def _open_download_response(url: str, start: int | None = None, end: int | None = None):
    headers = {"Accept-Encoding": "identity"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}" if end is not None else f"bytes={start}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        return urllib.request.urlopen(request, timeout=60)
    except urllib.error.URLError as error:
        raise RetryableDownloadError(f"Failed to download asset: {error.reason}.") from error


def _download_single_stream(
    *,
    url: str,
    destination: Path,
    partial_path: Path,
    existing_size: int,
    expected_size: int,
    callback: ProgressCallback | None,
    progress_start: float,
    progress_span: float,
    label: str,
) -> Path:
    response = _open_download_response(url, start=existing_size if existing_size else None)
    with response:
        response_status = response.getcode()
        if existing_size and response_status != 206:
            existing_size = 0
            mode = "wb"
        else:
            mode = "ab" if existing_size else "wb"

        downloaded_size = existing_size
        last_reported_percent = -1
        with partial_path.open(mode) as destination_file:
            while True:
                block = response.read(DOWNLOAD_CHUNK_SIZE)
                if not block:
                    break
                destination_file.write(block)
                downloaded_size += len(block)

                percent = min(100, int((downloaded_size / expected_size) * 100)) if expected_size else 0
                if percent >= last_reported_percent + 5:
                    last_reported_percent = percent
                    _emit_download_progress(callback, label, expected_size, downloaded_size, progress_start, progress_span)

    if partial_path.stat().st_size != expected_size:
        raise SetupError(
            f"Incomplete download for {label}: expected {expected_size:,} bytes, found {partial_path.stat().st_size:,} bytes."
        )

    os.replace(partial_path, destination)
    return destination


def _range_segments(start: int, end: int, connection_count: int) -> list[tuple[int, int]]:
    remaining_bytes = end - start + 1
    base_size, remainder = divmod(remaining_bytes, connection_count)
    segments: list[tuple[int, int]] = []
    cursor = start
    for index in range(connection_count):
        segment_size = base_size + (1 if index < remainder else 0)
        segment_end = cursor + segment_size - 1
        segments.append((cursor, segment_end))
        cursor = segment_end + 1
    return segments


def _download_range_segment(
    url: str,
    start: int,
    end: int,
    segment_path: Path,
    progress: dict[str, int],
    progress_lock: threading.Lock,
    callback: ProgressCallback | None,
    expected_size: int,
    progress_start: float,
    progress_span: float,
    label: str,
) -> None:
    expected_segment_size = end - start + 1
    retry_count = 0

    while True:
        completed_size = segment_path.stat().st_size if segment_path.is_file() else 0
        if completed_size > expected_segment_size:
            segment_path.unlink()
            completed_size = 0
        if completed_size == expected_segment_size:
            return

        range_start = start + completed_size
        try:
            response = _open_download_response(url, start=range_start, end=end)
            with response:
                if response.getcode() != 206:
                    raise SetupError(f"Server did not honor a range request for {label} bytes {range_start}-{end}.")
                content_range = response.headers.get("Content-Range", "")
                expected_content_range = f"bytes {range_start}-{end}/{expected_size}"
                if content_range != expected_content_range:
                    raise SetupError(f"Server returned an unexpected range for {label}: {content_range or 'missing Content-Range header'}.")

                last_reported_percent = -1
                with segment_path.open("ab" if completed_size else "wb") as segment_file:
                    while True:
                        block = response.read(DOWNLOAD_CHUNK_SIZE)
                        if not block:
                            break
                        segment_file.write(block)
                        with progress_lock:
                            progress["downloaded"] += len(block)
                            percent = min(100, int((progress["downloaded"] / expected_size) * 100))
                            should_report = percent >= progress["last_reported_percent"] + 5 and percent != last_reported_percent
                            if should_report:
                                progress["last_reported_percent"] = percent
                                last_reported_percent = percent
                                total_downloaded = progress["downloaded"]
                            else:
                                total_downloaded = 0
                        if total_downloaded:
                            _emit_download_progress(callback, label, expected_size, total_downloaded, progress_start, progress_span)

            completed_size = segment_path.stat().st_size
            if completed_size == expected_segment_size:
                return
            if completed_size > expected_segment_size:
                raise SetupError(
                    f"Range download for {label} bytes {start}-{end} exceeded its expected size of {expected_segment_size:,} bytes."
                )
            raise RetryableDownloadError(
                f"Range download for {label} bytes {start}-{end} ended early at {completed_size:,} of {expected_segment_size:,} bytes."
            )
        except (RetryableDownloadError, TimeoutError, ConnectionError, http.client.IncompleteRead, OSError) as error:
            retry_count += 1
            if retry_count >= DOWNLOAD_RANGE_MAX_ATTEMPTS:
                raise SetupError(
                    f"Could not download {label} bytes {start}-{end} after {DOWNLOAD_RANGE_MAX_ATTEMPTS} attempts: {error}"
                ) from error
            _emit(
                callback,
                status=f"Downloading {label}",
                progress=progress_start + (progress_span * (progress["downloaded"] / expected_size)),
                log=f"Transient download error for {label} bytes {start}-{end}; resuming range ({retry_count}/{DOWNLOAD_RANGE_MAX_ATTEMPTS - 1}).",
            )


def _download_with_parallel_ranges(
    *,
    url: str,
    destination: Path,
    partial_path: Path,
    existing_size: int,
    expected_size: int,
    connection_count: int,
    callback: ProgressCallback | None,
    progress_start: float,
    progress_span: float,
    label: str,
) -> Path:
    segment_dir = destination.with_name(f".{destination.name}.ranges")
    segment_dir.mkdir(parents=True, exist_ok=True)
    segments = _range_segments(existing_size, expected_size - 1, connection_count)
    segment_paths = [(start, end, segment_dir / f"{start:020d}-{end:020d}.part") for start, end in segments]
    completed_segment_bytes = 0
    for start, end, segment_path in segment_paths:
        if not segment_path.is_file():
            continue
        expected_segment_size = end - start + 1
        if segment_path.stat().st_size > expected_segment_size:
            segment_path.unlink()
            continue
        completed_segment_bytes += segment_path.stat().st_size
    progress = {"downloaded": existing_size + completed_segment_bytes, "last_reported_percent": -1}
    progress_lock = threading.Lock()

    _emit(
        callback,
        status=f"Downloading {label}",
        progress=progress_start + (progress_span * (progress["downloaded"] / expected_size)),
        log=f"Downloading {label} with {connection_count} parallel range connections.",
    )

    try:
        with ThreadPoolExecutor(max_workers=connection_count, thread_name_prefix="logsentinel-download") as executor:
            futures = [
                executor.submit(
                    _download_range_segment,
                    url,
                    start,
                    end,
                    segment_path,
                    progress,
                    progress_lock,
                    callback,
                    expected_size,
                    progress_start,
                    progress_span,
                    label,
                )
                for start, end, segment_path in segment_paths
            ]
            for future in as_completed(futures):
                future.result()

        merged_path = destination.with_name(f".{destination.name}.{uuid4().hex}.merge")
        try:
            with merged_path.open("wb") as merged_file:
                if existing_size:
                    with partial_path.open("rb") as prefix_file:
                        shutil.copyfileobj(prefix_file, merged_file, DOWNLOAD_CHUNK_SIZE)
                for _, _, segment_path in segment_paths:
                    with segment_path.open("rb") as segment_file:
                        shutil.copyfileobj(segment_file, merged_file, DOWNLOAD_CHUNK_SIZE)

            if merged_path.stat().st_size != expected_size:
                raise SetupError(
                    f"Merged parallel download for {label} is incomplete: expected {expected_size:,} bytes, found {merged_path.stat().st_size:,} bytes."
                )
            os.replace(merged_path, destination)
        except Exception:
            merged_path.unlink(missing_ok=True)
            raise
    finally:
        if destination.is_file() and destination.stat().st_size == expected_size:
            partial_path.unlink(missing_ok=True)
            shutil.rmtree(segment_dir, ignore_errors=True)

    _emit_download_progress(callback, label, expected_size, expected_size, progress_start, progress_span)
    return destination


def _promote_raw_directory(extracted_dir: Path, target_raw_dir: Path) -> None:
    target_raw_dir.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = target_raw_dir.with_name(f".{target_raw_dir.name}.backup-{uuid4().hex}")

    if target_raw_dir.exists():
        target_raw_dir.rename(backup_dir)

    try:
        extracted_dir.rename(target_raw_dir)
    except Exception:
        if backup_dir.exists() and not target_raw_dir.exists():
            backup_dir.rename(target_raw_dir)
        raise
    finally:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)


def _invalidate_prepared_outputs(spec: DatasetSpec) -> None:
    for split_name in ("train", "validation", "test"):
        (_dataset_dir(spec) / f"{split_name}.csv").unlink(missing_ok=True)

    state = load_setup_state()
    dataset_state = state["datasets"].setdefault(spec.dataset_id, {})
    dataset_state.pop("preparation", None)
    save_setup_state(state)


def provision_dataset(dataset_id: str, force: bool = False, callback: ProgressCallback | None = None) -> dict[str, Any]:
    spec = get_dataset_spec(dataset_id)
    with resource_lock(f"dataset:{spec.dataset_id}"):
        raw_ready, raw_files, raw_layout = _raw_is_ready(spec)
        if raw_ready and not force:
            _emit(callback, status=f"Dataset {spec.dataset_id} ready", progress=1.0, log=f"Reusing verified {raw_layout} raw files for {spec.dataset_id}.")
            return {"dataset_id": spec.dataset_id, "status": "skipped", "raw_files": raw_files, "layout": raw_layout}

        archive_path = _archive_path(spec)
        if _archive_is_verified(spec) and not force:
            _emit(callback, status=f"Verifying {spec.dataset_id} archive", progress=0.1, log=f"Using cached verified archive {archive_path.name}.")
        else:
            if archive_path.exists():
                archive_path.unlink()
            _emit(callback, status=f"Downloading {spec.dataset_id}", progress=0.0, log=f"Downloading {spec.archive_filename} from Zenodo.")
            download_with_resume(
                spec.download_url,
                archive_path,
                spec.expected_size,
                expected_md5=spec.md5,
                callback=callback,
                progress_start=0.0,
                progress_span=0.65,
                label=spec.archive_filename,
            )
            _emit(callback, status=f"Verifying {spec.dataset_id} archive", progress=0.68, log=f"Verifying MD5 for {spec.archive_filename}.")
            verify_checksum(archive_path, spec.expected_size, spec.md5)

        dataset_dir = _dataset_dir(spec)
        extracted_dir = dataset_dir.parent / f".{spec.dataset_id}.extract-{uuid4().hex}"
        try:
            _emit(callback, status=f"Extracting {spec.dataset_id}", progress=0.72, log=f"Safely extracting {spec.archive_filename}.")
            safe_extract_archive(spec, archive_path, extracted_dir)
            raw_files = discover_raw_files(extracted_dir, spec)
            _emit(callback, status=f"Finalizing {spec.dataset_id}", progress=0.92, log=f"Discovered {len(raw_files)} raw files for {spec.dataset_id}.")
            _promote_raw_directory(extracted_dir, _raw_dir(spec))
            _invalidate_prepared_outputs(spec)
        except Exception:
            if extracted_dir.exists():
                shutil.rmtree(extracted_dir)
            raise

        state = load_setup_state()
        state["datasets"][spec.dataset_id] = {
            "archive": {
                "filename": spec.archive_filename,
                "md5": spec.md5,
                "expected_size": spec.expected_size,
                "verified_at": _now(),
            },
            "raw_files": raw_files,
            "extracted_at": _now(),
            "last_error": None,
        }
        save_setup_state(state)
        _emit(callback, status=f"Dataset {spec.dataset_id} ready", progress=1.0, log=f"{spec.dataset_id} raw assets are ready.")
        return {"dataset_id": spec.dataset_id, "status": "provisioned", "raw_files": raw_files, "layout": "managed"}


def provision_model(model_key: str, force: bool = False, callback: ProgressCallback | None = None) -> dict[str, Any]:
    spec = get_model_spec(model_key)
    with resource_lock(f"model:{spec.key}"):
        current_status = _model_asset_status(spec)
        if current_status["ready"] and not force:
            _emit(callback, status=f"Model {spec.key} ready", progress=1.0, log=f"Reusing local {spec.model_id} model assets.")
            return {"model_key": spec.key, "status": "skipped", "path": current_status["path"]}

        token = os.getenv("HF_TOKEN")
        if spec.requires_hf_token and not token:
            raise SetupError(f"{spec.model_id} is gated. Configure HF_TOKEN on the backend before provisioning it.")

        target_dir = MODELS_DIR / spec.folder_name
        _emit(callback, status=f"Downloading model {spec.key}", progress=0.05, log=f"Downloading {spec.model_id} from Hugging Face.")
        try:
            snapshot_download(
                repo_id=spec.model_id,
                local_dir=str(target_dir),
                token=token,
                force_download=force,
            )
        except Exception as error:
            raise SetupError(f"Failed to download {spec.model_id}: {error}") from error

        current_status = _model_asset_status(spec)
        if not current_status["ready"]:
            raise SetupError(f"Model download for {spec.model_id} completed without the required config and weight files.")

        state = load_setup_state()
        state["models"][spec.key] = {
            "model_id": spec.model_id,
            "path": current_status["path"],
            "verified_at": _now(),
            "last_error": None,
        }
        save_setup_state(state)
        _emit(callback, status=f"Model {spec.key} ready", progress=1.0, log=f"{spec.model_id} is ready.")
        return {"model_key": spec.key, "status": "provisioned", "path": current_status["path"]}


def provision_assets(
    dataset_ids: Iterable[str] = (),
    model_keys: Iterable[str] = (),
    force: bool = False,
    callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    requested_datasets = list(dict.fromkeys(dataset_ids))
    requested_models = list(dict.fromkeys(model_keys))
    for dataset_id in requested_datasets:
        get_dataset_spec(dataset_id)
    for model_key in requested_models:
        get_model_spec(model_key)

    targets = [("dataset", dataset_id) for dataset_id in requested_datasets] + [("model", model_key) for model_key in requested_models]
    if not targets:
        raise SetupError("Select at least one dataset or model to provision.")

    results: dict[str, Any] = {"datasets": [], "models": []}
    total_targets = len(targets)
    for index, (target_type, target_id) in enumerate(targets):
        progress_start = index / total_targets
        progress_span = 1 / total_targets

        def scoped_callback(payload: dict[str, Any], start: float = progress_start, span: float = progress_span) -> None:
            local_progress = payload.get("progress")
            if local_progress is not None:
                payload = {**payload, "progress": start + (span * float(local_progress))}
            _emit(callback, **payload)

        if target_type == "dataset":
            results["datasets"].append(provision_dataset(target_id, force=force, callback=scoped_callback))
        else:
            results["models"].append(provision_model(target_id, force=force, callback=scoped_callback))

    return results


def record_preparation_result(dataset_id: str, result: dict[str, Any]) -> None:
    spec = get_dataset_spec(dataset_id)
    state = load_setup_state()
    dataset_state = state["datasets"].setdefault(spec.dataset_id, {})
    dataset_state["preparation"] = {"updated_at": _now(), **result}
    dataset_state["last_error"] = None
    save_setup_state(state)


def record_setup_error(target_type: str, target_id: str, message: str) -> None:
    state = load_setup_state()
    group = "datasets" if target_type == "dataset" else "models"
    target_state = state[group].setdefault(target_id, {})
    target_state["last_error"] = message
    save_setup_state(state)