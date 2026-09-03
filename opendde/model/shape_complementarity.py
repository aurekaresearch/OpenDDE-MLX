# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Geometric shape-complementarity fields between chains.

Every token gets an outward surface normal from the gradient of a Gaussian
density of its own chain's atoms; facing, roughly anti-parallel token pairs
across chains at a ~4 A gap score highly. Rows are processed in chunks of
``pair_chunk_size`` tokens to bound the ``[N_token, N_atom]`` intermediates.
"""

from typing import Any, Optional, TypedDict

import mlx.core as mx
import numpy as np

from opendde.data.tokenizer import STRUCTURAL_TOKEN_ROLES
from opendde.model.utils import aggregate_atom_to_token, scatter_sum

_STRUCTURAL_SHAPE_COMP_REQUIRED = (
    "parent_residue_idx",
    "structural_token_index",
    "atom_to_structural_token_idx",
    "structural_distogram_rep_atom_mask",
    "subtoken_role_id",
    "asym_id",
    "token_index",
)
_STRUCTURAL_SHAPE_COMP_ALIASES = (
    ("token_index", "structural_token_index", True),
    ("atom_to_token_idx", "atom_to_structural_token_idx", True),
    ("atom_to_tokatom_idx", "atom_to_structural_tokatom_idx", True),
    ("distogram_rep_atom_mask", "structural_distogram_rep_atom_mask", False),
    ("pae_rep_atom_mask", "structural_pae_rep_atom_mask", False),
)
_PARENT_INDEXED_TOKEN_FEATURES = ("asym_id", "residue_index", "entity_id", "sym_id")


class _ShapeCompTokenFeatures(TypedDict):
    atom_to_token_idx: mx.array
    rep_atom_mask: mx.array
    token_asym_id: mx.array
    token_role_id: mx.array
    is_structural: bool
    is_protein_token: mx.array


def _to_bool_mask(mask: Optional[Any], n_atom: int) -> mx.array:
    if mask is None:
        return mx.ones((n_atom,), dtype=mx.bool_)
    return mx.array(np.asarray(mask)).astype(mx.bool_)


def _first_item(x: Any) -> bool:
    return bool(np.asarray(x).reshape(-1)[0])


def get_shape_comp_atom_mask(
    feat_dict: dict[str, Any], label_dict: Optional[dict[str, Any]] = None
) -> mx.array:
    """Atoms contributing to the density: ``ref_mask`` at inference, supervision masks otherwise."""
    if "atom_to_token_idx" not in feat_dict:
        raise KeyError("shape complementarity requires atom_to_token_idx")
    n_atom = feat_dict["atom_to_token_idx"].shape[0]
    if label_dict is None:
        return _to_bool_mask(feat_dict.get("ref_mask"), n_atom)
    atom_mask = _to_bool_mask(label_dict.get("coordinate_mask"), n_atom)
    if "is_known_chain_condition_case" in feat_dict and _first_item(
        feat_dict["is_known_chain_condition_case"]
    ):
        return atom_mask
    if "atom_supervision_mask" in label_dict:
        return _to_bool_mask(label_dict["atom_supervision_mask"], n_atom)
    return atom_mask


def _select_rep_atom_mask(
    feat_dict: dict[str, Any], atom_to_token_idx: mx.array, n_token: int, is_structural: bool
) -> mx.array:
    """First candidate rep-atom mask that selects exactly one atom per active token."""
    candidate_keys = (
        ("structural_distogram_rep_atom_mask", "distogram_rep_atom_mask")
        if is_structural
        else ("distogram_rep_atom_mask", "structural_distogram_rep_atom_mask")
    )
    atom_to_token = np.asarray(atom_to_token_idx)
    checked = []
    for key in candidate_keys:
        if key not in feat_dict:
            continue
        rep_atom_mask = np.asarray(feat_dict[key]).astype(bool)
        checked.append(key)
        if rep_atom_mask.ndim != 1 or rep_atom_mask.shape[0] != atom_to_token.shape[0]:
            continue
        rep_token_idx = atom_to_token[np.flatnonzero(rep_atom_mask)]
        if rep_token_idx.shape[0] == n_token and np.array_equal(
            np.sort(rep_token_idx), np.arange(n_token)
        ):
            return mx.array(rep_atom_mask)
    raise ValueError(
        "Could not resolve representative atom mask for shape complementarity "
        f"with n_token={n_token}; checked={checked}"
    )


def _resolve_residue_protein_token_mask(
    feat_dict: dict[str, Any], atom_to_token_idx: mx.array, n_token: int
) -> mx.array:
    """Residue-level tokens that are multi-atom protein residues."""
    if "is_protein_token" in feat_dict and feat_dict["is_protein_token"].shape[0] == n_token:
        return feat_dict["is_protein_token"].astype(mx.bool_)

    def _token_any(name: str) -> mx.array:
        flag = mx.array(np.asarray(feat_dict[name])).astype(mx.float32)
        return scatter_sum(flag, atom_to_token_idx, n_token) > 0.5

    atom_count = scatter_sum(mx.ones(atom_to_token_idx.shape), atom_to_token_idx, n_token)
    return (
        _token_any("is_protein")
        & ~_token_any("is_ligand")
        & ~_token_any("is_dna")
        & ~_token_any("is_rna")
        & (atom_count > 1)
    )


def shape_comp_pred_uses_structural_tokens(
    feat_dict: dict[str, Any], pred_dict: dict[str, mx.array]
) -> bool:
    marker = pred_dict.get("shape_comp_uses_structural_tokens")
    if marker is not None:
        return _first_item(marker)
    structural_token_index = feat_dict.get("structural_token_index")
    if structural_token_index is None:
        return False
    pred_n_token = int(pred_dict["shape_comp_token_pred"].shape[-1])
    residue_n_token = int(feat_dict["token_index"].shape[-1])
    structural_n_token = int(structural_token_index.shape[-1])
    return pred_n_token == structural_n_token and pred_n_token != residue_n_token


def structural_shape_comp_feature_dict(feat_dict: dict[str, Any]) -> dict[str, Any]:
    """Re-key structural-token features under the residue-token names."""
    missing = [name for name in _STRUCTURAL_SHAPE_COMP_REQUIRED if name not in feat_dict]
    if missing:
        raise KeyError(
            "Structural-token shape-complementarity prediction requires feature(s): "
            + ", ".join(missing)
        )
    parent = feat_dict["parent_residue_idx"].astype(mx.int32)
    structural_token_index = feat_dict["structural_token_index"]
    residue_n_token = int(feat_dict["token_index"].shape[-1])
    if parent.ndim != 1 or structural_token_index.ndim != 1:
        raise ValueError("Structural-token shape-complementarity features must be unbatched")
    if parent.shape[0] != structural_token_index.shape[0]:
        raise ValueError(
            "parent_residue_idx must match structural token count: "
            f"{parent.shape} vs {structural_token_index.shape}"
        )
    if parent.size and (parent.min().item() < 0 or parent.max().item() >= residue_n_token):
        raise ValueError(
            f"parent_residue_idx points outside residue-token range 0..{residue_n_token - 1}"
        )

    structural = dict(feat_dict)
    for target_key, source_key, cast_int in _STRUCTURAL_SHAPE_COMP_ALIASES:
        if source_key in feat_dict:
            value = feat_dict[source_key]
            structural[target_key] = value.astype(mx.int32) if cast_int else value
    for token_feature in _PARENT_INDEXED_TOKEN_FEATURES:
        if token_feature in feat_dict:
            structural[token_feature] = mx.take(feat_dict[token_feature], parent, axis=-1)
    return structural


def resolve_shape_comp_feature_dict_for_pred(
    feat_dict: dict[str, Any], pred_dict: dict[str, mx.array]
) -> dict[str, Any]:
    if shape_comp_pred_uses_structural_tokens(feat_dict, pred_dict):
        return structural_shape_comp_feature_dict(feat_dict)
    return feat_dict


def resolve_shape_comp_token_features(
    feat_dict: dict[str, Any], n_token: Optional[int] = None
) -> _ShapeCompTokenFeatures:
    """Token-space features needed by the fields; works for residue and structural tokens."""
    if n_token is None:
        n_token = int(feat_dict["token_index"].shape[-1])
    atom_to_token_idx = feat_dict["atom_to_token_idx"].astype(mx.int32)
    if atom_to_token_idx.size == 0:
        raise ValueError("shape complementarity requires at least one atom")
    max_token_idx = int(atom_to_token_idx.max().item())
    if max_token_idx + 1 != n_token:
        raise ValueError(
            f"atom_to_token_idx does not match active token space: "
            f"max+1={max_token_idx + 1}, n_token={n_token}"
        )
    token_asym_id = feat_dict["asym_id"].astype(mx.int32)
    if token_asym_id.shape[0] != n_token:
        raise ValueError(f"asym_id does not match active token space: {token_asym_id.shape}")

    token_role_id = feat_dict.get("subtoken_role_id")
    is_structural = (
        token_role_id is not None and token_role_id.ndim == 1 and token_role_id.shape[0] == n_token
    )
    rep_atom_mask = _select_rep_atom_mask(feat_dict, atom_to_token_idx, n_token, is_structural)
    if is_structural:
        token_role_id = token_role_id.astype(mx.int32)
        if (
            "structural_is_protein_token" in feat_dict
            and feat_dict["structural_is_protein_token"].shape[0] == n_token
        ):
            is_protein_token = feat_dict["structural_is_protein_token"].astype(mx.bool_)
        else:
            is_protein_token = (token_role_id == STRUCTURAL_TOKEN_ROLES["protein_bb"]) | (
                token_role_id == STRUCTURAL_TOKEN_ROLES["protein_sc"]
            )
    else:
        token_role_id = mx.full((n_token,), -1, dtype=mx.int32)
        is_protein_token = _resolve_residue_protein_token_mask(
            feat_dict, atom_to_token_idx, n_token
        )
    return {
        "atom_to_token_idx": atom_to_token_idx,
        "rep_atom_mask": rep_atom_mask,
        "token_asym_id": token_asym_id,
        "token_role_id": token_role_id,
        "is_structural": is_structural,
        "is_protein_token": is_protein_token,
    }


def _sorted_rep_atom_indices(
    atom_to_token_idx: mx.array, rep_atom_mask: mx.array, n_token: int
) -> mx.array:
    """Representative atom index of token ``0..n_token-1`` in token order."""
    rep_atom_idx = np.flatnonzero(np.asarray(rep_atom_mask))
    rep_token_idx = np.asarray(atom_to_token_idx)[rep_atom_idx]
    order = np.argsort(rep_token_idx, kind="stable")
    if not np.array_equal(rep_token_idx[order], np.arange(n_token)):
        raise ValueError("Representative atoms do not cover the active token space exactly")
    return mx.array(rep_atom_idx[order])


def _masked_softmax(logits: mx.array, mask: mx.array, axis: int, eps: float) -> mx.array:
    masked_logits = mx.where(mask, logits, mx.finfo(logits.dtype).min)
    max_logits = masked_logits.max(axis=axis, keepdims=True)
    max_logits = mx.where(mask.any(axis=axis, keepdims=True), max_logits, 0.0)
    exp_logits = mx.where(mask, mx.exp(masked_logits - max_logits), 0.0)
    denom = mx.maximum(exp_logits.sum(axis=axis, keepdims=True), eps)
    return mx.where(mask, exp_logits / denom, 0.0)


def _masked_topk_mean(values: np.ndarray, mask: np.ndarray, topk: int) -> tuple[float, float]:
    """Mean and top-``k`` mean of ``values[mask]`` (zeros when nothing is valid)."""
    valid = values[mask]
    if valid.size == 0:
        return 0.0, 0.0
    top = np.sort(valid)[-min(topk, valid.size) :]
    return float(valid.mean()), float(top.mean())


def summarize_shape_comp_pair(
    pair_score: mx.array, pair_mask: mx.array, topk: int = 32
) -> tuple[mx.array, mx.array, mx.array]:
    """Masked mean, masked top-k mean and valid fraction of ``pair_score [..., N, N]``."""
    prefix_shape = tuple(pair_score.shape[:-2])
    flat_score = np.asarray(pair_score).reshape(-1, pair_score.shape[-2] * pair_score.shape[-1])
    flat_mask = np.asarray(pair_mask).reshape(flat_score.shape).astype(bool)
    stats = [_masked_topk_mean(s, m, topk) for s, m in zip(flat_score, flat_mask)]
    pair_mean = mx.array(np.array([s[0] for s in stats], dtype=np.float32)).reshape(prefix_shape)
    pair_topk_mean = mx.array(np.array([s[1] for s in stats], dtype=np.float32)).reshape(
        prefix_shape
    )
    valid_pair_frac = pair_mask.astype(pair_score.dtype).mean(axis=(-1, -2))
    return pair_mean, pair_topk_mean, valid_pair_frac


def build_shape_comp_pred_outputs(
    shape_comp: dict[str, mx.array], keep_pair_map: bool, pair_summary_topk: int = 32
) -> dict[str, mx.array]:
    """Rename field outputs to the ``*_pred`` keys stored in prediction dicts."""
    summary_keys = (
        "shape_comp_pair_mean",
        "shape_comp_pair_topk_mean",
        "shape_comp_valid_pair_frac",
    )
    if all(k in shape_comp for k in summary_keys):
        pair_mean, pair_topk_mean, valid_pair_frac = (shape_comp[k] for k in summary_keys)
    else:
        pair_mean, pair_topk_mean, valid_pair_frac = summarize_shape_comp_pair(
            shape_comp["shape_comp_pair"], shape_comp["shape_comp_pair_mask"], pair_summary_topk
        )
    outputs = {
        "shape_comp_token_pred": shape_comp["shape_comp_token"],
        "shape_comp_global_pred": shape_comp["shape_comp_global"],
        "shape_comp_token_mask": shape_comp["shape_comp_token_mask"],
        "shape_comp_pair_mean_pred": pair_mean,
        "shape_comp_pair_topk_mean_pred": pair_topk_mean,
        "shape_comp_valid_pair_frac_pred": valid_pair_frac,
    }
    if keep_pair_map:
        if "shape_comp_pair" not in shape_comp or "shape_comp_pair_mask" not in shape_comp:
            raise KeyError("keep_pair_map=True requires shape_comp_pair and mask")
        outputs["shape_comp_pair_pred"] = shape_comp["shape_comp_pair"]
        outputs["shape_comp_pair_mask"] = shape_comp["shape_comp_pair_mask"]
    return outputs


def compute_shape_complementarity_fields(
    coordinate: mx.array,
    feat_dict: dict[str, Any],
    atom_mask: Optional[mx.array] = None,
    density_sigma: float = 1.5,
    interface_cutoff: float = 12.0,
    gap_mean: float = 4.0,
    gap_scale: float = 2.0,
    clash_distance: float = 2.0,
    clash_scale: float = 0.5,
    pool_temperature: float = 25.0,
    normal_strength_min: float = 1e-3,
    pair_chunk_size: Optional[int] = 128,
    return_pair_map: bool = True,
    eps: float = 1e-6,
    **_: Any,
) -> dict[str, mx.array]:
    """Per-token, per-pair and global shape-complementarity scores.

    Args:
        coordinate: ``[..., N_atom, 3]``
        feat_dict: token features (see ``resolve_shape_comp_token_features``)
        atom_mask: ``[N_atom]`` atoms contributing to the density (default all)
    Returns:
        ``shape_comp_token [..., N_token]``, ``shape_comp_token_mask``, ``shape_comp_global [...]``,
        ``shape_comp_pair_mean``, ``shape_comp_pair_topk_mean``, ``shape_comp_valid_pair_frac``,
        ``normal_strength [..., N_token]`` and, if ``return_pair_map``, ``shape_comp_pair`` /
        ``shape_comp_pair_mask [..., N_token, N_token]``.
    """
    if coordinate.ndim < 2 or coordinate.shape[-1] != 3:
        raise ValueError(f"coordinate must have shape [..., N_atom, 3]; got {coordinate.shape}")
    n_token = int(feat_dict["token_index"].shape[-1])
    resolved = resolve_shape_comp_token_features(feat_dict, n_token=n_token)
    atom_to_token_idx = resolved["atom_to_token_idx"]
    token_asym_id = resolved["token_asym_id"]
    is_protein_token = resolved["is_protein_token"].astype(mx.bool_)

    coord = coordinate.astype(mx.float32)
    prefix_shape = tuple(coord.shape[:-2])
    prefix_ones = (1,) * len(prefix_shape)
    atom_mask = _to_bool_mask(atom_mask, coord.shape[-2])
    atom_mask_float = atom_mask.astype(mx.float32)

    rep_atom_indices = _sorted_rep_atom_indices(
        atom_to_token_idx, resolved["rep_atom_mask"], n_token
    )
    rep_center = mx.take(coord, rep_atom_indices, axis=-2)
    rep_valid = atom_mask[rep_atom_indices]
    if resolved["is_structural"]:
        supervised_count = scatter_sum(atom_mask_float, atom_to_token_idx, n_token)
        supervised_center = (
            aggregate_atom_to_token(
                coord * atom_mask_float[:, None], atom_to_token_idx, n_token, reduce="sum"
            )
            / mx.maximum(supervised_count, 1.0)[:, None]
        )
        protein_sc = resolved["token_role_id"] == STRUCTURAL_TOKEN_ROLES["protein_sc"]
        token_center = mx.where(protein_sc[:, None], supervised_center, rep_center)
        center_valid = mx.where(protein_sc, supervised_count > 0, rep_valid)
    else:
        token_center, center_valid = rep_center, rep_valid

    chunk_size = (
        n_token if not pair_chunk_size or pair_chunk_size <= 0 else min(pair_chunk_size, n_token)
    )
    atom_asym_id = token_asym_id[atom_to_token_idx]
    same_chain_atom_mask = (token_asym_id[:, None] == atom_asym_id[None, :]) & atom_mask[None, :]

    # surface normal = gradient of the same-chain Gaussian density at the token centre
    gradient_chunks = []
    for start in range(0, n_token, chunk_size):
        end = min(start + chunk_size, n_token)
        delta = token_center[..., start:end, None, :] - coord[..., None, :, :]
        weight = mx.exp(-(delta * delta).sum(-1) / (2.0 * density_sigma * density_sigma))
        weight = weight * same_chain_atom_mask[start:end].astype(mx.float32)
        gradient_chunks.append(
            (weight[..., None] * delta).sum(-2) / (density_sigma * density_sigma)
        )
    token_gradient = mx.concatenate(gradient_chunks, axis=-2)
    normal_strength = mx.sqrt((token_gradient * token_gradient).sum(-1))
    token_normal = token_gradient / mx.maximum(normal_strength, eps)[..., None]
    token_valid = (center_valid & is_protein_token).reshape(prefix_ones + (n_token,)) & (
        normal_strength > normal_strength_min
    )

    flat_prefix = int(np.prod(prefix_shape)) if prefix_shape else 1
    topk = 32
    topk_values = np.full((flat_prefix, topk), -np.inf, dtype=np.float32)
    pair_sum = mx.zeros(prefix_shape)
    pair_count = mx.zeros(prefix_shape)
    token_score_chunks, token_mask_chunks, pair_score_chunks, pair_mask_chunks = [], [], [], []
    for start in range(0, n_token, chunk_size):
        end = min(start + chunk_size, n_token)
        chunk_len = end - start
        delta = token_center[..., None, :, :] - token_center[..., start:end, None, :]
        dist = mx.sqrt((delta * delta).sum(-1))
        unit = delta / mx.maximum(dist, eps)[..., None]
        normal_i = token_normal[..., start:end, None, :]
        normal_j = token_normal[..., None, :, :]
        facing = mx.maximum((normal_i * unit).sum(-1), 0.0) * mx.maximum(
            (normal_j * -unit).sum(-1), 0.0
        )
        opposite = 0.5 * (1.0 - (normal_i * normal_j).sum(-1))
        gap = mx.exp(-(((dist - gap_mean) / gap_scale) ** 2))
        anti_clash = 1.0 - mx.sigmoid((clash_distance - dist) / clash_scale)
        cross_chain = token_asym_id[start:end, None] != token_asym_id[None, :]
        pair_mask = (
            token_valid[..., start:end, None]
            & token_valid[..., None, :]
            & cross_chain.reshape(prefix_ones + (chunk_len, n_token))
            & (dist <= interface_cutoff)
        )
        pair_score = mx.where(pair_mask, facing * opposite * gap * anti_clash, 0.0)
        partner_weight = _masked_softmax(-(dist * dist) / pool_temperature, pair_mask, -1, eps)
        token_mask = pair_mask.any(axis=-1)
        token_score_chunks.append(mx.where(token_mask, (partner_weight * pair_score).sum(-1), 0.0))
        token_mask_chunks.append(token_mask)

        mask_float = pair_mask.astype(mx.float32)
        pair_sum = pair_sum + (pair_score * mask_float).sum(axis=(-2, -1))
        pair_count = pair_count + mask_float.sum(axis=(-2, -1))
        flat_score = np.asarray(pair_score).reshape(flat_prefix, -1)
        flat_mask = np.asarray(pair_mask).reshape(flat_prefix, -1)
        for b in range(flat_prefix):
            valid = flat_score[b][flat_mask[b]]
            if valid.size:
                combined = np.concatenate([topk_values[b], valid])
                topk_values[b] = np.sort(combined)[-topk:]
        if return_pair_map:
            pair_score_chunks.append(pair_score)
            pair_mask_chunks.append(pair_mask)

    token_score = mx.concatenate(token_score_chunks, axis=-1)
    token_mask = mx.concatenate(token_mask_chunks, axis=-1)
    global_denom = mx.maximum(token_mask.astype(mx.float32).sum(-1), 1.0)
    global_score = mx.where(token_mask.any(axis=-1), token_score.sum(-1) / global_denom, 0.0)

    pair_mean = mx.where(pair_count > 0, pair_sum / mx.maximum(pair_count, 1.0), 0.0)
    topk_finite = np.isfinite(topk_values)
    pair_topk_mean = np.where(
        topk_finite.any(-1),
        np.where(topk_finite, topk_values, 0.0).sum(-1) / np.maximum(topk_finite.sum(-1), 1),
        0.0,
    ).astype(np.float32)
    outputs = {
        "shape_comp_token": token_score,
        "shape_comp_token_mask": token_mask,
        "shape_comp_global": global_score,
        "shape_comp_pair_mean": pair_mean,
        "shape_comp_pair_topk_mean": mx.array(pair_topk_mean).reshape(prefix_shape),
        "shape_comp_valid_pair_frac": pair_count / float(max(n_token * n_token, 1)),
        "normal_strength": normal_strength,
    }
    if return_pair_map:
        outputs["shape_comp_pair"] = mx.concatenate(pair_score_chunks, axis=-2)
        outputs["shape_comp_pair_mask"] = mx.concatenate(pair_mask_chunks, axis=-2)
    return outputs
