# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Small array helpers shared by the MLX model modules."""

from typing import Any, Optional, Sequence

import mlx.core as mx
import numpy as np
from scipy.spatial.transform import Rotation


def permute_final_dims(x: mx.array, inds: Sequence[int]) -> mx.array:
    """Permute the last ``len(inds)`` dims of ``x`` (OpenFold convention)."""
    zero_index = -1 * len(inds)
    first_inds = list(range(len(x.shape[:zero_index])))
    return x.transpose(*(first_inds + [zero_index + i for i in inds]))


def flatten_final_dims(x: mx.array, num_dims: int) -> mx.array:
    return x.reshape(*x.shape[:-num_dims], -1)


def expand_at_dim(x: mx.array, dim: int, n: int) -> mx.array:
    """Insert a new axis at ``dim`` and broadcast it to size ``n``."""
    x = mx.expand_dims(x, dim)
    if dim < 0:
        dim = x.ndim + dim
    shape = list(x.shape)
    shape[dim] = n
    return mx.broadcast_to(x, shape)


def pad_at_dim(x: mx.array, dim: int, pad_length: Sequence[int], value: float = 0) -> mx.array:
    """Pad ``x`` along ``dim`` with ``pad_length[0]`` before and ``[1]`` after."""
    if dim < 0:
        dim = x.ndim + dim
    if tuple(pad_length) == (0, 0):
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[dim] = (int(pad_length[0]), int(pad_length[1]))
    return mx.pad(x, pad_width, constant_values=value)


def reshape_at_dim(x: mx.array, dim: int, target_shape: Sequence[int]) -> mx.array:
    if dim < 0:
        dim = x.ndim + dim
    return x.reshape(*x.shape[:dim], *target_shape, *x.shape[dim + 1 :])


def move_final_dim_to_dim(x: mx.array, dim: int) -> mx.array:
    n_dim = x.ndim
    if dim < 0:
        dim = n_dim + dim
    if dim >= n_dim - 1:
        return x
    order = list(range(dim)) + [n_dim - 1] + list(range(dim, n_dim - 1))
    return x.transpose(*order)


def unfold(x: mx.array, dim: int, size: int, step: int) -> mx.array:
    """Sliding windows along ``dim`` (like ``Tensor.unfold``), window on a new last axis."""
    if dim < 0:
        dim = x.ndim + dim
    n = x.shape[dim]
    n_windows = (n - size) // step + 1
    idx = mx.arange(n_windows)[:, None] * step + mx.arange(size)[None, :]
    out = mx.take(x, idx.reshape(-1), axis=dim)
    out = reshape_at_dim(out, dim, (n_windows, size))
    # move the window axis (dim + 1) to the end
    order = [i for i in range(out.ndim) if i != dim + 1] + [dim + 1]
    return out.transpose(*order)


def broadcast_token_to_atom(x_token: mx.array, atom_to_token_idx: mx.array) -> mx.array:
    """Gather token rows for every atom: ``[..., N_token, d] -> [..., N_atom, d]``."""
    return mx.take(x_token, atom_to_token_idx, axis=-2)


def aggregate_atom_to_token(
    x_atom: mx.array,
    atom_to_token_idx: mx.array,
    n_token: Optional[int] = None,
    reduce: str = "mean",
) -> mx.array:
    """Scatter-reduce atom rows into token rows along the second last axis."""
    if n_token is None:
        n_token = int(atom_to_token_idx.max().item()) + 1
    out = mx.zeros((*x_atom.shape[:-2], n_token, x_atom.shape[-1]), dtype=x_atom.dtype)
    out = out.at[..., atom_to_token_idx, :].add(x_atom)
    if reduce == "sum":
        return out
    if reduce == "mean":
        count = (
            mx.zeros((n_token,), dtype=x_atom.dtype)
            .at[atom_to_token_idx]
            .add(mx.ones(atom_to_token_idx.shape, dtype=x_atom.dtype))
        )
        return out / mx.maximum(count, 1)[:, None]
    raise ValueError(f"Unsupported reduce: {reduce}")


def scatter_sum(src: mx.array, index: mx.array, dim_size: int) -> mx.array:
    """Sum ``src[..., i]`` into ``out[..., index[i]]`` along the last axis."""
    out = mx.zeros((*src.shape[:-1], dim_size), dtype=src.dtype)
    return out.at[..., index].add(src)


def one_hot(x: mx.array, lower_bins: mx.array, upper_bins: mx.array) -> mx.array:
    """Binned one-hot of ``x`` from open intervals ``(lower, upper)``."""
    x = x[..., None]
    return ((x > lower_bins) & (x < upper_bins)).astype(mx.float32)


def distogram_bin_tops(min_bin: float, max_bin: float, no_bins: int) -> mx.array:
    """Inclusive top edge of each distogram bin (``no_bins - 1`` breaks + inf)."""
    breaks = mx.array(np.linspace(min_bin, max_bin, no_bins - 1, dtype=np.float32))
    return mx.concatenate([breaks, mx.array([float("inf")])])


def cdist(a: mx.array, b: Optional[mx.array] = None) -> mx.array:
    """Pairwise Euclidean distances between the rows of ``a`` and ``b``."""
    if b is None:
        b = a
    diff = a[..., :, None, :] - b[..., None, :, :]
    return mx.sqrt(mx.maximum((diff * diff).sum(-1), 0.0))


def rot_vec_mul(r: mx.array, t: mx.array) -> mx.array:
    """Apply rotation matrices ``r [..., 3, 3]`` to vectors ``t [..., 3]``."""
    return (r * t[..., None, :]).sum(-1)


def uniform_random_rotation(n: int, rng: np.random.Generator) -> mx.array:
    return mx.array(Rotation.random(num=n, random_state=rng).as_matrix().astype(np.float32))


def centre_random_augmentation(
    x: mx.array,
    rng: np.random.Generator,
    N_sample: int = 1,
    s_trans: float = 1.0,
) -> mx.array:
    """Algorithm 19: centre coordinates, then apply a random rigid transform.

    Args:
        x: coordinates ``[..., N_atom, 3]``
    Returns:
        ``[..., N_sample, N_atom, 3]``
    """
    x = x - x.mean(axis=-2, keepdims=True)
    x = expand_at_dim(x, dim=-3, n=N_sample)
    batch_shape = x.shape[:-3]
    n_augment = int(np.prod(batch_shape)) * N_sample
    rot = uniform_random_rotation(n_augment, rng).reshape(*batch_shape, N_sample, 3, 3)
    trans = mx.array(
        (s_trans * rng.standard_normal((*batch_shape, N_sample, 3))).astype(np.float32)
    )
    return rot_vec_mul(rot[..., None, :, :], x) + trans[..., None, :]


def simple_merge_dict_list(dict_list: list[dict]) -> dict:
    merged: dict[Any, list[np.ndarray]] = {}
    for d in dict_list:
        for k, v in d.items():
            if isinstance(v, (float, int)):
                v = np.array([v])
            elif isinstance(v, mx.array):
                v = np.asarray(v).reshape(-1)
            merged.setdefault(k, []).append(np.asarray(v))
    return {k: np.concatenate(v) for k, v in merged.items()}
