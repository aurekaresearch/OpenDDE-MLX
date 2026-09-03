# SPDX-License-Identifier: Apache-2.0
"""Helpers to load PyTorch reference dumps into MLX modules."""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from opendde.model.checkpoint import load_weights, remap_key

PARITY_DIR = Path(os.environ.get("OPENDDE_PARITY_DIR", "/tmp/parity"))


def load_case(name: str) -> dict[str, np.ndarray]:
    path = PARITY_DIR / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"reference dump {path} missing; run tests/parity/reference.py")
    return dict(np.load(path))


def load_module(module: nn.Module, arrays: dict[str, np.ndarray], prefix: str) -> nn.Module:
    """Load ``{prefix}.param:{torch_key}`` entries into ``module``."""
    head = f"{prefix}.param:"
    weights = {
        remap_key(k[len(head) :]): mx.array(v) for k, v in arrays.items() if k.startswith(head)
    }
    load_weights(module, weights, strict=True)
    return module


def assert_close(
    actual: mx.array, expected: np.ndarray, atol: float = 1e-4, rtol: float = 1e-4
) -> None:
    actual_np = np.asarray(actual.astype(mx.float32))
    assert actual_np.shape == expected.shape, f"shape {actual_np.shape} vs {expected.shape}"
    diff = np.abs(actual_np - expected)
    tol = atol + rtol * np.abs(expected)
    worst = float((diff - tol).max())
    assert np.all(diff <= tol), f"max abs diff {diff.max():.3e} (worst excess {worst:.3e})"
