# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Per-sample confidence summaries (pLDDT, PDE, pTM/ipTM, clashes, ranking score).

Softmax and bin-weighted reductions run in MLX; the small per-chain loops run
on NumPy arrays, so every summary value is a NumPy scalar or array.
"""

from typing import TYPE_CHECKING, Any, Optional, Union

import mlx.core as mx
import numpy as np

from opendde.metrics.clash import Clash
from opendde.model.utils import distogram_bin_tops

if TYPE_CHECKING:  # pragma: no cover
    from opendde.config.schema import BinConfig, OpenDDEConfig

ArrayLike = Union[mx.array, np.ndarray]


def _as_mx(x: ArrayLike) -> mx.array:
    return x if isinstance(x, mx.array) else mx.array(np.asarray(x))


def _as_np(x: ArrayLike) -> np.ndarray:
    return np.asarray(x)


def _remap_contiguous(asym_id: ArrayLike) -> np.ndarray:
    """Relabel chain ids to ``0..N_chain-1`` (chains may have been filtered out)."""
    return np.unique(_as_np(asym_id).astype(np.int64), return_inverse=True)[1]


def get_bin_params(cfg: "BinConfig") -> dict[str, Any]:
    return {"min_bin": cfg.min_bin, "max_bin": cfg.max_bin, "no_bins": cfg.no_bins}


def get_bin_centers(min_bin: float, max_bin: float, no_bins: int) -> mx.array:
    """Centre of each of the ``no_bins`` equal-width bins in ``[min_bin, max_bin]``."""
    bin_width = (max_bin - min_bin) / no_bins
    boundaries = np.linspace(min_bin, max_bin - bin_width, no_bins, dtype=np.float32)
    return mx.array(boundaries + np.float32(0.5 * bin_width))


def compute_contact_prob(
    distogram_logits: mx.array, min_bin: float, max_bin: float, no_bins: int, thres: float = 8.0
) -> mx.array:
    """Probability mass of distogram bins whose top edge is ``<= thres`` ``[..., N, N]``."""
    prob = mx.softmax(distogram_logits, axis=-1)
    in_contact = distogram_bin_tops(min_bin, max_bin, no_bins) <= thres
    return (prob * in_contact.astype(prob.dtype)).sum(-1)


def logits_to_score(
    logits: mx.array, min_bin: float, max_bin: float, no_bins: int, return_prob: bool = False
) -> Union[mx.array, tuple[mx.array, mx.array]]:
    """Expected bin centre under ``softmax(logits)``; optionally also the probabilities."""
    prob = mx.softmax(logits, axis=-1)
    score = prob @ get_bin_centers(min_bin, max_bin, no_bins)
    return (score, prob) if return_prob else score


def calculate_normalization(N: int) -> float:
    """TM-score normalisation constant."""
    return 1.24 * (max(N, 19) - 15) ** (1 / 3) - 1.8


def _token_pair_tm(
    pae_prob: mx.array,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    token_idx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """TM-style per-pair score ``[..., N_d, N_d]`` restricted to ``token_idx`` tokens."""
    if token_idx is not None:
        idx = mx.array(token_idx)
        pae_prob = mx.take(mx.take(pae_prob, idx, axis=-3), idx, axis=-2)
    ptm_norm = calculate_normalization(pae_prob.shape[-2])
    per_bin_weight = 1.0 / (1.0 + (get_bin_centers(min_bin, max_bin, no_bins) / ptm_norm) ** 2)
    return _as_np(pae_prob @ per_bin_weight)


def _select_tokens(
    has_frame: ArrayLike, token_mask: Optional[ArrayLike]
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    has_frame = _as_np(has_frame).astype(bool)
    if token_mask is None:
        return has_frame, None
    token_idx = np.flatnonzero(_as_np(token_mask))
    return has_frame[token_idx], token_idx


def calculate_ptm(
    pae_prob: mx.array,
    has_frame: ArrayLike,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    token_mask: Optional[ArrayLike] = None,
) -> np.ndarray:
    """pTM ``[...]`` from ``pae_prob [..., N_token, N_token, N_bins]``."""
    has_frame, token_idx = _select_tokens(has_frame, token_mask)
    if not has_frame.any():
        return np.zeros(pae_prob.shape[:-3], dtype=np.float32)
    token_pair_tm = _token_pair_tm(pae_prob, min_bin, max_bin, no_bins, token_idx)
    return token_pair_tm.mean(-1)[..., has_frame].max(-1)


def calculate_iptm(
    pae_prob: mx.array,
    has_frame: ArrayLike,
    asym_id: ArrayLike,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    token_mask: Optional[ArrayLike] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """ipTM ``[...]``: pTM restricted to cross-chain token pairs."""
    has_frame, token_idx = _select_tokens(has_frame, token_mask)
    if not has_frame.any():
        return np.zeros(pae_prob.shape[:-3], dtype=np.float32)
    asym_id = _as_np(asym_id)
    if token_idx is not None:
        asym_id = asym_id[token_idx]
    token_pair_tm = _token_pair_tm(pae_prob, min_bin, max_bin, no_bins, token_idx)
    is_diff_chain = (asym_id[None, :] != asym_id[:, None]).astype(np.float32)
    iptm = (token_pair_tm * is_diff_chain).sum(-1) / (eps + is_diff_chain.sum(-1))
    return iptm[..., has_frame].max(-1)


def calculate_chain_based_ptm(
    pae_prob: mx.array,
    has_frame: ArrayLike,
    asym_id: ArrayLike,
    token_is_ligand: ArrayLike,
    min_bin: float,
    max_bin: float,
    no_bins: int,
) -> dict[str, np.ndarray]:
    """``chain_ptm``, ``chain_iptm``, ``chain_pair_iptm`` and ``chain_pair_iptm_global``."""
    has_frame = _as_np(has_frame).astype(bool)
    asym_id = _remap_contiguous(asym_id)
    token_is_ligand = _as_np(token_is_ligand).astype(bool)
    n_chain = int(asym_id.max()) + 1
    chain_masks = [asym_id == aid for aid in range(n_chain)]
    chain_is_ligand = [token_is_ligand[mask].sum() >= mask.sum() // 2 for mask in chain_masks]
    bins = (min_bin, max_bin, no_bins)
    batch_shape = tuple(pae_prob.shape[:-3])

    chain_pair_iptm = np.zeros(batch_shape + (n_chain, n_chain), dtype=np.float32)
    for aid_1 in range(n_chain):
        for aid_2 in range(aid_1 + 1, n_chain):
            pair_mask = chain_masks[aid_1] | chain_masks[aid_2]
            value = calculate_iptm(pae_prob, has_frame, asym_id, *bins, token_mask=pair_mask)
            chain_pair_iptm[..., aid_1, aid_2] = chain_pair_iptm[..., aid_2, aid_1] = value

    chain_ptm = np.zeros(batch_shape + (n_chain,), dtype=np.float32)
    for aid, mask in enumerate(chain_masks):
        chain_ptm[..., aid] = calculate_ptm(pae_prob, has_frame, *bins, token_mask=mask)

    chain_has_frame = [(mask & has_frame).any() for mask in chain_masks]
    chain_iptm = np.zeros(batch_shape + (n_chain,), dtype=np.float32)
    for aid in range(n_chain):
        pairs = [
            (i, j)
            for i in range(n_chain)
            for j in range(n_chain)
            if (i == aid or j == aid) and i != j and chain_has_frame[i]
        ]
        if pairs:
            vals = np.stack([chain_pair_iptm[..., i, j] for i, j in pairs], axis=-1)
            chain_iptm[..., aid] = vals.mean(-1)

    chain_pair_iptm_global = np.zeros(batch_shape + (n_chain, n_chain), dtype=np.float32)
    for aid_1 in range(n_chain):
        for aid_2 in range(n_chain):
            if aid_1 == aid_2:
                continue
            if chain_is_ligand[aid_1]:
                value = chain_iptm[..., aid_1]
            elif chain_is_ligand[aid_2]:
                value = chain_iptm[..., aid_2]
            else:
                value = (chain_iptm[..., aid_1] + chain_iptm[..., aid_2]) * 0.5
            chain_pair_iptm_global[..., aid_1, aid_2] = value

    return {
        "chain_ptm": chain_ptm,
        "chain_iptm": chain_iptm,
        "chain_pair_iptm": chain_pair_iptm,
        "chain_pair_iptm_global": chain_pair_iptm_global,
    }


def calculate_chain_based_gpde(
    token_pair_pde: ArrayLike,
    contact_probs: ArrayLike,
    asym_id: ArrayLike,
    eps: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Contact-weighted PDE within each chain (``chain_gpde``) and per chain pair."""
    token_pair_pde = _as_np(token_pair_pde)
    contact_probs = _as_np(contact_probs)
    asym_id = _remap_contiguous(asym_id)
    n_chain = int(asym_id.max()) + 1
    batch_shape = token_pair_pde.shape[:-2]

    def _gpde(mask_1: np.ndarray, mask_2: np.ndarray) -> np.ndarray:
        probs = contact_probs[..., mask_1, :][..., mask_2]
        pde = token_pair_pde[..., mask_1, :][..., mask_2]
        return (pde * probs).sum((-1, -2)) / (probs.sum((-1, -2)) + eps)

    chain_gpde = np.zeros(batch_shape + (n_chain,), dtype=np.float32)
    chain_pair_gpde = np.zeros(batch_shape + (n_chain, n_chain), dtype=np.float32)
    for aid_1 in range(n_chain):
        chain_gpde[..., aid_1] = _gpde(asym_id == aid_1, asym_id == aid_1)
        for aid_2 in range(aid_1 + 1, n_chain):
            value = _gpde(asym_id == aid_1, asym_id == aid_2)
            chain_pair_gpde[..., aid_1, aid_2] = chain_pair_gpde[..., aid_2, aid_1] = value
    return {"chain_gpde": chain_gpde, "chain_pair_gpde": chain_pair_gpde}


def calculate_chain_based_plddt(
    atom_plddt: ArrayLike, asym_id: ArrayLike, atom_to_token_idx: ArrayLike
) -> dict[str, np.ndarray]:
    """Mean atom pLDDT per chain and per chain pair."""
    atom_plddt = _as_np(atom_plddt)
    asym_id = _remap_contiguous(asym_id)
    atom_asym_id = asym_id[_as_np(atom_to_token_idx).astype(np.int64)]
    n_chain = int(asym_id.max()) + 1
    batch_shape = atom_plddt.shape[:-1]

    def _mean_plddt(atom_mask: np.ndarray) -> np.ndarray:
        return atom_plddt[..., atom_mask].mean(-1, dtype=np.float64).astype(atom_plddt.dtype)

    chain_plddt = np.zeros(batch_shape + (n_chain,), dtype=np.float32)
    chain_pair_plddt = np.zeros(batch_shape + (n_chain, n_chain), dtype=np.float32)
    for aid_1 in range(n_chain):
        chain_plddt[..., aid_1] = _mean_plddt(atom_asym_id == aid_1)
        for aid_2 in range(n_chain):
            if aid_1 != aid_2:
                pair_mask = (atom_asym_id == aid_1) | (atom_asym_id == aid_2)
                chain_pair_plddt[..., aid_1, aid_2] = _mean_plddt(pair_mask)
    return {"chain_plddt": chain_plddt, "chain_pair_plddt": chain_pair_plddt}


def calculate_clash(
    pred_coordinate: ArrayLike,
    asym_id: ArrayLike,
    atom_to_token_idx: ArrayLike,
    is_polymer: ArrayLike,
    threshold: float,
) -> np.ndarray:
    """Whether any polymer chain pair clashes (AF3 criterion) ``[N_sample]``."""
    is_polymer = _as_np(is_polymer).astype(np.int64)
    dummy = np.zeros_like(is_polymer)
    clash_dict = Clash(af3_clash_threshold=threshold, compute_vdw_clash=False)(
        pred_coordinate, asym_id, atom_to_token_idx, 1 - is_polymer, is_polymer, dummy, dummy
    )
    n_sample = _as_np(pred_coordinate).shape[0]
    return clash_dict["summary"]["af3_clash"].reshape(n_sample, -1).max(-1)


def break_down_to_per_sample_dict(
    input_dict: dict[str, Any], shared_keys: list[str] = []
) -> list[dict[str, Any]]:
    """Split leading-axis batched entries into one dict per sample."""
    per_sample_keys = [key for key in input_dict if key not in shared_keys]
    assert len(per_sample_keys) > 0
    n_sample = input_dict[per_sample_keys[0]].shape[0]
    for key in per_sample_keys:
        assert input_dict[key].shape[0] == n_sample
    return [
        {key: input_dict[key][i] for key in per_sample_keys}
        | {key: input_dict[key] for key in shared_keys}
        for i in range(n_sample)
    ]


def _compute_full_data_and_summary(
    configs: "OpenDDEConfig",
    pae_logits: mx.array,
    plddt_logits: mx.array,
    pde_logits: mx.array,
    contact_probs: mx.array,
    token_asym_id: np.ndarray,
    token_has_frame: np.ndarray,
    atom_coordinate: np.ndarray,
    atom_to_token_idx: np.ndarray,
    atom_is_polymer: np.ndarray,
    N_recycle: int,
    return_full_data: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summary (and optionally full) confidence data for a batch of samples."""
    n_token = token_asym_id.shape[0]
    atom_is_ligand = 1 - atom_is_polymer.astype(np.int64)
    token_is_ligand = np.bincount(atom_to_token_idx, weights=atom_is_ligand, minlength=n_token) > 0

    atom_plddt = logits_to_score(plddt_logits, **get_bin_params(configs.confidence.plddt))
    token_pair_pde = logits_to_score(pde_logits, **get_bin_params(configs.confidence.pde))
    pae_bins = get_bin_params(configs.confidence.pae)
    pae_score, pae_prob = logits_to_score(pae_logits, **pae_bins, return_prob=True)

    summary: dict[str, Any] = {}
    summary["plddt"] = _as_np(atom_plddt.mean(-1) * 100)
    summary["gpde"] = _as_np(
        (token_pair_pde * contact_probs).sum((-1, -2)) / contact_probs.sum((-1, -2))
    )
    summary["ptm"] = calculate_ptm(pae_prob, token_has_frame, **pae_bins)
    summary["iptm"] = calculate_iptm(pae_prob, token_has_frame, token_asym_id, **pae_bins)
    summary.update(calculate_chain_based_gpde(token_pair_pde, contact_probs, token_asym_id))
    summary.update(
        calculate_chain_based_ptm(
            pae_prob, token_has_frame, token_asym_id, token_is_ligand, **pae_bins
        )
    )
    summary.update(calculate_chain_based_plddt(atom_plddt, token_asym_id, atom_to_token_idx))
    summary["has_clash"] = calculate_clash(
        atom_coordinate,
        token_asym_id,
        atom_to_token_idx,
        atom_is_polymer,
        configs.metrics.clash.af3_clash_threshold,
    )
    summary["num_recycles"] = np.array(N_recycle)
    summary["disorder"] = np.zeros_like(summary["ptm"])
    summary["ranking_score"] = (
        0.8 * summary["iptm"]
        + 0.2 * summary["ptm"]
        + 0.5 * summary["disorder"]
        - 100 * summary["has_clash"]
    ).astype(np.float32)
    summary_by_sample = break_down_to_per_sample_dict(summary, shared_keys=["num_recycles"])
    if not return_full_data:
        return summary_by_sample, [{}]

    full_data = {
        "atom_plddt": _as_np(atom_plddt),
        "token_pair_pde": _as_np(token_pair_pde),
        "contact_probs": _as_np(contact_probs),
        "token_pair_pae": _as_np(pae_score),
        "token_has_frame": token_has_frame,
        "token_asym_id": token_asym_id,
        "atom_to_token_idx": atom_to_token_idx,
        "atom_is_polymer": atom_is_polymer,
        "atom_coordinate": atom_coordinate,
    }
    shared = [
        "contact_probs",
        "token_has_frame",
        "token_asym_id",
        "atom_to_token_idx",
        "atom_is_polymer",
    ]
    return summary_by_sample, break_down_to_per_sample_dict(full_data, shared_keys=shared)


def compute_full_data_and_summary(
    configs: "OpenDDEConfig",
    pae_logits: ArrayLike,
    plddt_logits: ArrayLike,
    pde_logits: ArrayLike,
    contact_probs: ArrayLike,
    token_asym_id: ArrayLike,
    token_has_frame: ArrayLike,
    atom_coordinate: ArrayLike,
    atom_to_token_idx: ArrayLike,
    atom_is_polymer: ArrayLike,
    N_recycle: int,
    return_full_data: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Wrapper of ``_compute_full_data_and_summary`` enumerating the ``N_sample`` samples.

    Args:
        pae_logits, pde_logits: ``[N_sample, N_token, N_token, N_bins]``
        plddt_logits: ``[N_sample, N_atom, N_bins]``
        contact_probs: ``[N_token, N_token]`` or ``[N_sample, N_token, N_token]``
        atom_coordinate: ``[N_sample, N_atom, 3]``
    """
    pae_logits, plddt_logits, pde_logits = map(_as_mx, (pae_logits, plddt_logits, pde_logits))
    contact_probs = _as_mx(contact_probs)
    n_sample = pae_logits.shape[0]
    if contact_probs.ndim == 2:
        contact_probs = mx.broadcast_to(contact_probs[None], (n_sample, *contact_probs.shape))
    assert contact_probs.shape[0] == plddt_logits.shape[0] == pde_logits.shape[0] == n_sample
    token_asym_id = _as_np(token_asym_id)
    token_has_frame = _as_np(token_has_frame)
    atom_coordinate = _as_np(atom_coordinate)
    atom_to_token_idx = _as_np(atom_to_token_idx).astype(np.int64)
    atom_is_polymer = _as_np(atom_is_polymer)

    summary_confidence, full_data = [], []
    for i in range(n_sample):
        summary_i, full_i = _compute_full_data_and_summary(
            configs=configs,
            pae_logits=pae_logits[i : i + 1],
            plddt_logits=plddt_logits[i : i + 1],
            pde_logits=pde_logits[i : i + 1],
            contact_probs=contact_probs[i],
            token_asym_id=token_asym_id,
            token_has_frame=token_has_frame,
            atom_coordinate=atom_coordinate[i : i + 1],
            atom_to_token_idx=atom_to_token_idx,
            atom_is_polymer=atom_is_polymer,
            N_recycle=N_recycle,
            return_full_data=return_full_data,
        )
        summary_confidence.extend(summary_i)
        full_data.extend(full_i)
    return summary_confidence, full_data
