# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Environment diagnostics for the ``opendde-mlx doctor`` command."""

from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys
from importlib import metadata

from opendde.config.data import default_root_dir
from opendde.version import __version__

_OPTIONAL_MODULES = ("rdkit", "biotite", "Bio", "sklearn", "scipy", "pandas", "torch")


def distribution_version(module_name: str) -> str | None:
    try:
        return metadata.version(module_name)
    except metadata.PackageNotFoundError:
        return None


def module_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception:
        return False
    return True


def _chip_name() -> str:
    try:
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return platform.processor() or "unknown"


def _memory_gb() -> float:
    try:
        return (
            int(
                subprocess.run(
                    ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
                ).stdout
            )
            / 1e9
        )
    except Exception:
        return float("nan")


def format_doctor_report() -> str:
    lines = [
        f"OpenDDE-MLX {__version__}",
        f"Python {sys.version.split()[0]} ({sys.executable})",
        f"Platform: {platform.platform()}",
        f"Chip: {_chip_name()}",
        f"Unified memory: {_memory_gb():.1f} GB",
    ]
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        lines.append("WARNING: OpenDDE-MLX supports Apple Silicon macOS only.")
    try:
        import mlx.core as mx

        lines.append(
            f"MLX {mx.__version__}: default device {mx.default_device()}, metal={mx.metal.is_available()}"
        )
    except Exception as exc:  # pragma: no cover - MLX missing
        lines.append(f"MLX: unavailable ({exc})")
    for module_name in _OPTIONAL_MODULES:
        version = distribution_version(module_name if module_name != "Bio" else "biopython")
        status = "ok" if module_available(module_name) else "missing"
        note = " (only needed to convert .pt checkpoints)" if module_name == "torch" else ""
        lines.append(f"{module_name}: {status} {version or ''}{note}")
    root_dir = os.environ.get("OPENDDE_ROOT_DIR", default_root_dir())
    lines.append(f"OPENDDE_ROOT_DIR: {root_dir}")
    checkpoint_dir = os.path.join(root_dir, "checkpoint")
    if os.path.isdir(checkpoint_dir):
        lines.append("Checkpoints: " + ", ".join(sorted(os.listdir(checkpoint_dir))))
    else:
        lines.append("Checkpoints: none found (run `opendde-mlx pred` to download opendde.pt)")
    return "\n".join(lines)
