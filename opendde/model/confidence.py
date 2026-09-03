# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Confidence head (AF3 Algorithm 31): pLDDT, PAE, PDE and resolved logits."""

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from opendde.model.pairformer import PairformerStack
from opendde.model.primitives import LayerNorm, LinearNoBias
from opendde.model.utils import broadcast_token_to_atom, cdist, one_hot


class ConfidenceHead(nn.Module):
    """Runs a small Pairformer on the trunk embeddings plus predicted distances.

    Samples are processed one at a time so only one pair tensor is live.
    """

    def __init__(
        self,
        n_blocks: int = 4,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        b_pae: int = 64,
        b_pde: int = 64,
        b_plddt: int = 50,
        b_resolved: int = 2,
        max_atoms_per_token: int = 20,
        distance_bin_start: float = 3.25,
        distance_bin_end: float = 52.0,
        distance_bin_step: float = 1.25,
        hidden_scale_up: bool = False,
    ) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.c_s = c_s
        self.c_z = c_z
        self.c_s_inputs = c_s_inputs
        self.b_pae = b_pae
        self.b_pde = b_pde
        self.b_plddt = b_plddt
        self.b_resolved = b_resolved
        self.max_atoms_per_token = max_atoms_per_token
        self.linear_no_bias_s1 = LinearNoBias(c_s_inputs, c_z)
        self.linear_no_bias_s2 = LinearNoBias(c_s_inputs, c_z)
        lower_bins = np.arange(distance_bin_start, distance_bin_end, distance_bin_step)
        self.lower_bins = mx.array(lower_bins.astype(np.float32))
        self.upper_bins = mx.array(np.append(lower_bins[1:], 1e6).astype(np.float32))
        self.num_bins = len(lower_bins)
        self.linear_no_bias_d = LinearNoBias(self.num_bins, c_z)
        self.linear_no_bias_d_wo_onehot = LinearNoBias(1, c_z)
        self.pairformer_stack = PairformerStack(
            c_z=c_z, c_s=c_s, n_blocks=n_blocks, hidden_scale_up=hidden_scale_up
        )
        self.linear_no_bias_pae = LinearNoBias(c_z, b_pae, initializer="zeros")
        self.linear_no_bias_pde = LinearNoBias(c_z, b_pde, initializer="zeros")
        self.plddt_weight = mx.zeros((max_atoms_per_token, c_s, b_plddt))
        self.resolved_weight = mx.zeros((max_atoms_per_token, c_s, b_resolved))
        self.input_strunk_ln = LayerNorm(c_s)
        self.pae_ln = LayerNorm(c_z)
        self.pde_ln = LayerNorm(c_z)
        self.plddt_ln = LayerNorm(c_s)
        self.resolved_ln = LayerNorm(c_s)

    @staticmethod
    def _select_distogram_rep_atom_mask(feat: dict[str, Any], n_token: int) -> np.ndarray:
        """Prefer the structural representative-atom mask when it covers every token."""
        structural_mask = feat.get("structural_distogram_rep_atom_mask")
        if structural_mask is not None:
            structural_mask = np.asarray(structural_mask).astype(bool)
            if int(structural_mask.sum()) == n_token:
                return structural_mask
        return np.asarray(feat["distogram_rep_atom_mask"]).astype(bool)

    def __call__(
        self,
        feat: dict[str, Any],
        s_inputs: mx.array,
        s_trunk: mx.array,
        z_trunk: mx.array,
        pair_mask: Optional[mx.array],
        x_pred_coords: mx.array,
        chunk_size: Optional[int] = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        Args:
            s_inputs ``[..., N, c_s_inputs]``, s_trunk ``[..., N, c_s]``,
            z_trunk ``[..., N, N, c_z]``,
            x_pred_coords ``[..., N_sample, N_atom, 3]``
        Returns:
            plddt ``[..., N_sample, N_atom, b_plddt]``, pae/pde ``[..., N_sample, N, N, b]``,
            resolved ``[..., N_sample, N_atom, b_resolved]``
        """
        s_trunk = self.input_strunk_ln(mx.clip(s_trunk, -512, 512))
        z_init = (
            self.linear_no_bias_s1(s_inputs)[..., None, :, :]
            + self.linear_no_bias_s2(s_inputs)[..., None, :]
        )
        z_trunk = z_init + z_trunk
        rep_mask = self._select_distogram_rep_atom_mask(feat, s_trunk.shape[-2])
        x_rep = mx.take(x_pred_coords, mx.array(np.flatnonzero(rep_mask)), axis=-2)

        preds = []
        for i in range(x_rep.shape[-3]):
            sample = self.memory_efficient_forward(
                feat, s_trunk, z_trunk, pair_mask, x_rep[..., i, :, :], chunk_size=chunk_size
            )
            mx.eval(sample)
            preds.append(sample)
        plddt, pae, pde, resolved = zip(*preds)
        return (
            mx.stack(plddt, axis=-3),
            mx.stack(pae, axis=-4),
            mx.stack(pde, axis=-4),
            mx.stack(resolved, axis=-3),
        )

    def memory_efficient_forward(
        self,
        feat: dict[str, Any],
        s_trunk: mx.array,
        z_pair: mx.array,
        pair_mask: Optional[mx.array],
        x_pred_rep_coords: mx.array,
        chunk_size: Optional[int] = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Single-sample forward; ``x_pred_rep_coords [..., N, 3]`` are the token rep atoms."""
        distance_pred = cdist(x_pred_rep_coords.astype(mx.float32))
        z_pair = z_pair + self.linear_no_bias_d(
            one_hot(distance_pred, self.lower_bins, self.upper_bins).astype(z_pair.dtype)
        )
        z_pair = z_pair + self.linear_no_bias_d_wo_onehot(
            distance_pred[..., None].astype(z_pair.dtype)
        )
        s_single, z_pair = self.pairformer_stack(
            s_trunk,
            z_pair,
            pair_mask,
            chunk_size=chunk_size,
            extra_attn_bias=feat.get("structural_pair_attn_bias"),
        )
        z_pair = z_pair.astype(mx.float32)
        s_single = s_single.astype(mx.float32)
        pae_pred = self.linear_no_bias_pae(self.pae_ln(z_pair))
        pde_pred = self.linear_no_bias_pde(self.pde_ln(z_pair + z_pair.swapaxes(-2, -3)))
        a = broadcast_token_to_atom(s_single, feat["atom_to_token_idx"])
        atom_to_tokatom_idx = feat["atom_to_tokatom_idx"]
        plddt_pred = mx.einsum(
            "...nc,ncb->...nb", self.plddt_ln(a), self.plddt_weight[atom_to_tokatom_idx]
        )
        resolved_pred = mx.einsum(
            "...nc,ncb->...nb", self.resolved_ln(a), self.resolved_weight[atom_to_tokatom_idx]
        )
        return plddt_pred, pae_pred, pde_pred, resolved_pred
