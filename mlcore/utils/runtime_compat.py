"""Compatibility checks for optional CUDA acceleration."""

import importlib.metadata
import importlib.util
import os
import platform
import re
import shutil
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


_BITSANDBYTES_BINARY_PATTERN = re.compile(r"libbitsandbytes_cuda(?P<tag>\d+)\.so$")


@dataclass(frozen=True)
class BitsAndBytesCudaBinary:
    """A bitsandbytes binary selected for the active CUDA runtime."""

    tag: str
    path: Path
    source: str


@dataclass(frozen=True)
class TorchCompileReadiness:
    """Whether optional torch.compile acceleration can safely be enabled."""

    ready: bool
    detail: str


def _cuda_version_parts(cuda_version: str | None) -> tuple[int, int] | None:
    if not cuda_version:
        return None

    match = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", str(cuda_version))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _bitsandbytes_package_dir() -> Path | None:
    spec = importlib.util.find_spec("bitsandbytes")
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).parent


def _configured_cuda_tag() -> str | None:
    configured = os.getenv("BNB_CUDA_VERSION", "")
    tag = re.sub(r"[^0-9]", "", configured)
    return tag or None


def resolve_bitsandbytes_cuda_binary(
    cuda_version: str | None,
    package_dir: Path | None = None,
) -> BitsAndBytesCudaBinary | None:
    """Find an exact or same-major compatible bitsandbytes CUDA binary."""
    package_dir = package_dir or _bitsandbytes_package_dir()
    if package_dir is None:
        return None

    configured_tag = _configured_cuda_tag()
    if configured_tag:
        configured_path = package_dir / f"libbitsandbytes_cuda{configured_tag}.so"
        if configured_path.is_file():
            return BitsAndBytesCudaBinary(configured_tag, configured_path, "configured")
        return None

    version_parts = _cuda_version_parts(cuda_version)
    if version_parts is None:
        return None

    cuda_major, cuda_minor = version_parts
    exact_tag = f"{cuda_major}{cuda_minor}"
    exact_path = package_dir / f"libbitsandbytes_cuda{exact_tag}.so"
    if exact_path.is_file():
        return BitsAndBytesCudaBinary(exact_tag, exact_path, "exact")

    compatible_binaries: list[BitsAndBytesCudaBinary] = []
    major_prefix = str(cuda_major)
    for binary_path in package_dir.glob("libbitsandbytes_cuda*.so"):
        match = _BITSANDBYTES_BINARY_PATTERN.fullmatch(binary_path.name)
        if match is None:
            continue
        tag = match.group("tag")
        if not tag.startswith(major_prefix):
            continue
        minor_text = tag.removeprefix(major_prefix)
        if not minor_text.isdigit():
            continue
        binary_minor = int(minor_text)
        if binary_minor <= cuda_minor:
            compatible_binaries.append(BitsAndBytesCudaBinary(tag, binary_path, "compatible"))

    return max(compatible_binaries, key=lambda binary: int(binary.tag.removeprefix(major_prefix)), default=None)


def configure_bitsandbytes_cuda(cuda_version: str | None = None) -> BitsAndBytesCudaBinary | None:
    """Configure bitsandbytes before it imports its native CUDA library."""
    if cuda_version is None:
        try:
            import torch
        except Exception:
            return None
        cuda_version = torch.version.cuda

    binary = resolve_bitsandbytes_cuda_binary(cuda_version)
    if binary is not None and binary.source == "compatible":
        os.environ.setdefault("BNB_CUDA_VERSION", binary.tag)
    return binary


def bitsandbytes_version() -> str | None:
    try:
        return importlib.metadata.version("bitsandbytes")
    except importlib.metadata.PackageNotFoundError:
        return None


def torch_compile_readiness() -> TorchCompileReadiness:
    """Report whether the optional Linux torch.compile path is usable."""
    if platform.system() != "Linux":
        return TorchCompileReadiness(False, "torch.compile acceleration is only enabled by this application on Linux.")

    if importlib.util.find_spec("triton") is None:
        return TorchCompileReadiness(False, "Triton is not installed.")

    include_path = sysconfig.get_path("include")
    python_header = Path(include_path) / "Python.h" if include_path else None
    if python_header is None or not python_header.is_file():
        return TorchCompileReadiness(
            False,
            f"Python development header was not found at {python_header}. torch.compile acceleration will be skipped.",
        )

    compiler_path = shutil.which("gcc")
    if compiler_path is None:
        return TorchCompileReadiness(False, "gcc was not found. torch.compile acceleration will be skipped.")

    return TorchCompileReadiness(True, f"Found Python.h and gcc at {compiler_path} for torch.compile acceleration.")


def maybe_compile_model(model: Any, log: Callable[[str], None], mode: str = "max-autotune") -> Any:
    """Apply torch.compile only when the local optional toolchain is ready."""
    readiness = torch_compile_readiness()
    if not readiness.ready:
        log(f"Skipping optional torch.compile acceleration: {readiness.detail}")
        return model

    try:
        import torch

        log("Enabling optional torch.compile() acceleration.")
        return torch.compile(model, mode=mode)
    except Exception as error:
        log(f"Skipping optional torch.compile acceleration after setup failed: {error}")
        return model