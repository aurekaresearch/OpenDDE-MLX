# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Valid-first random subsampling of MSA rows (uses the global numpy RNG)."""

from typing import Optional

import mlx.core as mx
import numpy as np


def _collapse_msa_row_mask(row_mask: np.ndarray) -> np.ndarray:
    """Collapse optional batch dims and keep one boolean per MSA row."""
    if row_mask.ndim == 1:
        return row_mask
    return row_mask.reshape(-1, row_mask.shape[-1]).any(axis=0)


def subsample_msa_feature_dict_valid_first(
    feat_dict: dict[str, mx.array],
    dim_dict: dict[str, int],
    num_msa: int = 1024,
    msa_mask: Optional[mx.array] = None,
    gap_token: Optional[int] = None,
) -> dict[str, mx.array]:
    """Subsample MSA rows with AF3/OpenFold3-style valid-first priority.

    Rows with at least one valid token are shuffled ahead of fully padded/all-gap
    rows, then truncated to ``num_msa``. Each call re-samples the order.
    """
    msa = np.asarray(feat_dict["msa"])
    msa_len = msa.shape[dim_dict["msa"]]
    num_msa = max(0, min(num_msa, msa_len))

    indices = np.zeros(0, dtype=np.int64)
    if num_msa > 0:
        row_valid = None
        if msa_mask is not None:
            row_valid = _collapse_msa_row_mask(np.asarray(msa_mask).astype(bool).any(axis=-1))
        # The stored msa_mask is usually all-ones, so fall back to the tokens themselves.
        if gap_token is not None and (row_valid is None or row_valid.all()):
            row_valid = _collapse_msa_row_mask((msa != gap_token).any(axis=-1))
        if row_valid is None:
            row_valid = np.ones(msa_len, dtype=bool)

        valid_idx = np.flatnonzero(row_valid)
        invalid_idx = np.flatnonzero(~row_valid)
        selected = []
        take_valid = min(valid_idx.size, num_msa)
        if take_valid > 0:
            selected.append(valid_idx[np.random.permutation(valid_idx.size)][:take_valid])
        take_invalid = num_msa - take_valid
        if take_invalid > 0 and invalid_idx.size > 0:
            selected.append(invalid_idx[np.random.permutation(invalid_idx.size)][:take_invalid])
        if selected:
            indices = np.concatenate(selected)

    idx = mx.array(indices)
    return {name: mx.take(feat_dict[name], idx, axis=dim) for name, dim in dim_dict.items()}
