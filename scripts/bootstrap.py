#!/usr/bin/env python3
"""Cross-platform LogSentinel bootstrap for dependencies and provisioned assets."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mlcore.setup_catalog import DATASET_SPECS, MODEL_SPECS
from mlcore.setup_manager import SetupError, provision_assets, scan_setup_state


def _parse_dataset_ids(values: list[str] | None) -> list[str]:
    if not values:
        return []
    if "all" in values:
        return list(DATASET_SPECS)

    invalid = sorted(set(values) - set(DATASET_SPECS))
    if invalid:
        raise SetupError(f"Unsupported dataset IDs: {', '.join(invalid)}.")
    return list(dict.fromkeys(values))


def _install_python_dependencies(python_executable: str) -> None:
    requirements_path = REPOSITORY_ROOT / "backend" / "requirements.txt"
    print(f"Installing Python dependencies from {requirements_path}.")
    subprocess.run([python_executable, "-m", "pip", "install", "-r", str(requirements_path)], check=True)


def _install_frontend_dependencies() -> None:
    npm_path = shutil.which("npm")
    if npm_path is None:
        raise SetupError("npm was not found. Install Node.js before requesting frontend dependency setup.")

    frontend_dir = REPOSITORY_ROOT / "frontend"
    print(f"Installing frontend dependencies in {frontend_dir}.")
    subprocess.run([npm_path, "ci"], cwd=frontend_dir, check=True)


def _progress(payload: dict) -> None:
    status = payload.get("status", "Provisioning")
    progress = payload.get("progress")
    message = payload.get("log")
    prefix = f"[{progress * 100:5.1f}%] " if isinstance(progress, (int, float)) else ""
    print(f"{prefix}{status}{f': {message}' if message else ''}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap LogSentinel dependencies, datasets, and models.")
    parser.add_argument("--check", action="store_true", help="Print current setup status as JSON without changing files.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[*DATASET_SPECS, "all"],
        help="Dataset archives to provision. Use 'all' for every supported dataset.",
    )
    parser.add_argument("--models", action="store_true", help="Provision the default encoder and gated Llama model.")
    parser.add_argument("--all", action="store_true", help="Provision every supported dataset and both default models.")
    parser.add_argument("--force", action="store_true", help="Replace verified managed raw assets or model snapshots.")
    parser.add_argument("--install-python-deps", action="store_true", help="Install backend Python dependencies into the active interpreter.")
    parser.add_argument("--install-frontend-deps", action="store_true", help="Run npm ci in the frontend package.")
    parser.add_argument("--python-executable", default=sys.executable, help="Python interpreter used with --install-python-deps.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_ids = _parse_dataset_ids(args.datasets)
    model_keys = list(MODEL_SPECS) if args.models else []

    if args.all:
        dataset_ids = list(DATASET_SPECS)
        model_keys = list(MODEL_SPECS)

    if args.install_python_deps:
        _install_python_dependencies(args.python_executable)
    if args.install_frontend_deps:
        _install_frontend_dependencies()

    if dataset_ids or model_keys:
        result = provision_assets(dataset_ids=dataset_ids, model_keys=model_keys, force=args.force, callback=_progress)
        print(json.dumps(result, indent=2))

    if args.check or not any((dataset_ids, model_keys, args.install_python_deps, args.install_frontend_deps)):
        print(json.dumps(scan_setup_state(), indent=2))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SetupError, subprocess.CalledProcessError) as error:
        print(f"Bootstrap failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error