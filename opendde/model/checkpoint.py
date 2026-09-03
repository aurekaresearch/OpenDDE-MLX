# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Checkpoint conversion and loading.

Released OpenDDE checkpoints are PyTorch ``.pt`` files. They are converted
once to ``.safetensors`` (same parameter names, with a few ``nn.Sequential``
indices renumbered) and then loaded directly with ``mx.load``.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

logger = logging.getLogger(__name__)

# torch ``nn.Sequential`` slots (activations occupy indices) -> MLX list index
_SEQUENTIAL_REMAP = {
    "small_mlp": {"1": "0", "3": "1", "5": "2"},
    "single_split_mlp": {"0": "0", "1": "1", "3": "2"},
}
_SEQUENTIAL_PATTERN = re.compile(r"(^|\.)(small_mlp|single_split_mlp)\.(\d+)\.")


def remap_key(torch_key: str) -> str:
    """Map a PyTorch state-dict key to the MLX parameter path."""
    key = torch_key[len("module.") :] if torch_key.startswith("module.") else torch_key

    def _sub(match: re.Match[str]) -> str:
        prefix, name, index = match.group(1), match.group(2), match.group(3)
        return f"{prefix}{name}.{_SEQUENTIAL_REMAP[name][index]}."

    return _SEQUENTIAL_PATTERN.sub(_sub, key)


def convert_torch_checkpoint(pt_path: str, safetensors_path: str) -> None:
    """Convert a released ``.pt`` checkpoint into ``.safetensors`` (needs torch)."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Converting .pt checkpoints requires PyTorch: `uv pip install torch`. "
            "Alternatively download the converted .safetensors checkpoint."
        ) from exc
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    arrays = {
        remap_key(key): mx.array(value.detach().to(torch.float32).numpy())
        for key, value in state_dict.items()
    }
    logger.info("Converting %d tensors from %s to %s", len(arrays), pt_path, safetensors_path)
    # mx.save_safetensors appends the extension when it is missing
    tmp_path = safetensors_path[: -len(".safetensors")] + ".tmp.safetensors"
    mx.save_safetensors(tmp_path, arrays)
    os.replace(tmp_path, safetensors_path)


def ensure_safetensors(checkpoint_path: str) -> str:
    """Return a ``.safetensors`` path, converting a ``.pt`` checkpoint on first use."""
    if checkpoint_path.endswith(".safetensors"):
        return checkpoint_path
    safetensors_path = os.path.splitext(checkpoint_path)[0] + ".safetensors"
    if not os.path.exists(safetensors_path):
        logger.info("Converting %s to %s (one-time)", checkpoint_path, safetensors_path)
        convert_torch_checkpoint(checkpoint_path, safetensors_path)
    return safetensors_path


def load_checkpoint(model: nn.Module, checkpoint_path: str, strict: bool = True) -> dict[str, Any]:
    """Load ``.safetensors`` (or ``.pt``, converted on the fly) weights into ``model``."""
    weights = mx.load(ensure_safetensors(checkpoint_path))
    return load_weights(model, weights, strict=strict)


def load_weights(
    model: nn.Module, weights: dict[str, mx.array], strict: bool = True
) -> dict[str, Any]:
    """Load a flat ``{name: array}`` mapping, reporting missing/unexpected keys."""
    model_params = dict(tree_flatten(model.parameters()))
    missing = sorted(set(model_params) - set(weights))
    unexpected = sorted(set(weights) - set(model_params))
    if strict and (missing or unexpected):
        raise ValueError(
            f"Checkpoint mismatch: {len(missing)} missing, {len(unexpected)} unexpected keys.\n"
            f"missing (first 10): {missing[:10]}\nunexpected (first 10): {unexpected[:10]}"
        )
    matched = []
    for name, value in weights.items():
        if name not in model_params:
            continue
        expected_shape = model_params[name].shape
        if tuple(value.shape) != tuple(expected_shape):
            raise ValueError(
                f"Shape mismatch for {name}: checkpoint {value.shape} vs model {expected_shape}"
            )
        matched.append((name, value.astype(model_params[name].dtype)))
    model.load_weights(matched, strict=False)
    mx.eval(model.parameters())
    return {"missing": missing, "unexpected": unexpected, "loaded": len(matched)}


def count_parameters(model: nn.Module) -> int:
    return sum(v.size for _, v in tree_flatten(model.parameters()))
