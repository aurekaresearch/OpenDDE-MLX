# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Expansion of residue-level trunk activations into structural (sub-)tokens."""

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from opendde.data.tokenizer import STRUCTURAL_TOKEN_ROLES
from opendde.model.primitives import LayerNorm, LinearNoBias

_BACKBONE_ROLES = (
    STRUCTURAL_TOKEN_ROLES["protein_bb"],
    STRUCTURAL_TOKEN_ROLES["dna_bb"],
    STRUCTURAL_TOKEN_ROLES["rna_bb"],
)
_SIDECHAIN_ROLES = (STRUCTURAL_TOKEN_ROLES["protein_sc"],)
_BASE_ROLES = (STRUCTURAL_TOKEN_ROLES["dna_base"], STRUCTURAL_TOKEN_ROLES["rna_base"])
# role category (0 backbone, 1 sidechain, 2 base, 3 other) x category -> role pair type
_ROLE_PAIR_TYPE_TABLE = mx.array(
    [[0, 1, 4, 7], [2, 3, 7, 7], [5, 7, 6, 7], [7, 7, 7, 7]], dtype=mx.int32
)


def _isin(x: mx.array, values: tuple[int, ...]) -> mx.array:
    out = x == values[0]
    for v in values[1:]:
        out = out | (x == v)
    return out


def _trunc_normal(shape: tuple[int, ...], std: float) -> mx.array:
    return std * mx.random.truncated_normal(-2.0, 2.0, shape)


class StructuralTokenExpander(nn.Module):
    """Gather parent-residue activations per structural token and add role conditioning.

    Single activations receive a role embedding (plus a small MLP on ``s``);
    pair activations receive a role-pair projection, boolean pair-feature
    embeddings and a scalar attention bias per pair feature.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_s_inputs: int,
        n_roles: int = max(STRUCTURAL_TOKEN_ROLES.values()) + 1,
        init_mode: str = "zero",
        role_init_std: float = 0.02,
        pair_feature_init_std: float = 0.02,
        attention_bias_init: float = 0.1,
        pair_projection_mode: str = "full",
        pair_chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if n_roles < max(STRUCTURAL_TOKEN_ROLES.values()) + 1:
            raise ValueError(f"n_roles={n_roles} is too small for structural token roles")
        if pair_projection_mode not in {"full", "none"}:
            raise ValueError(f"Unsupported pair_projection_mode {pair_projection_mode!r}")
        if init_mode not in {"zero", "scratch"}:
            raise ValueError(f"init_mode must be 'zero' or 'scratch'; got {init_mode!r}")
        self.c_s = c_s
        self.c_z = c_z
        self.c_s_inputs = c_s_inputs
        self.n_roles = n_roles
        self.pair_projection_mode = pair_projection_mode
        self.pair_chunk_size = pair_chunk_size

        self.single_split_mlp = [
            LayerNorm(c_s),
            LinearNoBias(c_s, 2 * c_s),
            LinearNoBias(2 * c_s, c_s, initializer="zeros"),
        ]
        self.single_input_role_embedding = nn.Embedding(n_roles, c_s_inputs)
        self.single_role_embedding = nn.Embedding(n_roles, c_s)
        if pair_projection_mode == "full":
            self.pair_block_proj = [
                LinearNoBias(c_z, c_z, initializer="zeros") for _ in range(n_roles * n_roles)
            ]
        self.same_parent_embedding = nn.Embedding(2, c_z)
        self.same_residue_twin_embedding = nn.Embedding(2, c_z)
        self.prev_bb_chain_embedding = nn.Embedding(2, c_z)
        self.next_bb_chain_embedding = nn.Embedding(2, c_z)
        self.role_pair_type_embedding = nn.Embedding(8, c_z)
        self.attn_bias_same_parent = mx.zeros(())
        self.attn_bias_same_residue_twin = mx.zeros(())
        self.attn_bias_prev_bb_chain = mx.zeros(())
        self.attn_bias_next_bb_chain = mx.zeros(())
        self.attn_bias_role_pair_type = mx.zeros((8,))
        self._init_role_conditioning(
            init_mode, role_init_std, pair_feature_init_std, attention_bias_init
        )

    def _init_role_conditioning(
        self, init_mode: str, role_std: float, pair_std: float, bias_init: float
    ) -> None:
        role_embeddings = [self.single_input_role_embedding, self.single_role_embedding]
        boolean_embeddings = [
            self.same_parent_embedding,
            self.same_residue_twin_embedding,
            self.prev_bb_chain_embedding,
            self.next_bb_chain_embedding,
        ]
        for emb in role_embeddings + boolean_embeddings + [self.role_pair_type_embedding]:
            emb.weight = mx.zeros(emb.weight.shape)
        if init_mode == "zero":
            return
        for emb in role_embeddings:
            emb.weight = _trunc_normal(emb.weight.shape, role_std)
        for emb in boolean_embeddings:
            emb.weight = mx.concatenate(
                [mx.zeros((1, self.c_z)), _trunc_normal((1, self.c_z), pair_std)]
            )
        self.role_pair_type_embedding.weight = _trunc_normal((8, self.c_z), pair_std)
        self.attn_bias_same_parent = mx.array(bias_init)
        self.attn_bias_same_residue_twin = mx.array(bias_init)
        self.attn_bias_prev_bb_chain = mx.array(bias_init)
        self.attn_bias_next_bb_chain = mx.array(bias_init)

    def _pair_project_by_role_full(
        self, z: mx.array, row_role: np.ndarray, col_role: np.ndarray
    ) -> mx.array:
        """Apply ``pair_block_proj[role_i * n_roles + role_j]`` to every ``z[..., i, j, :]``.

        Rows and columns are stably sorted by role so that each (role_i, role_j)
        block is one contiguous matmul; the result is permuted back afterwards.
        """
        row_order = np.argsort(row_role, kind="stable")
        col_order = np.argsort(col_role, kind="stable")
        z_sorted = mx.take(mx.take(z, mx.array(row_order), axis=-3), mx.array(col_order), axis=-2)
        row_bounds = np.searchsorted(row_role[row_order], np.arange(self.n_roles + 1))
        col_bounds = np.searchsorted(col_role[col_order], np.arange(self.n_roles + 1))
        row_blocks = []
        for ri in range(self.n_roles):
            r0, r1 = int(row_bounds[ri]), int(row_bounds[ri + 1])
            if r0 == r1:
                continue
            col_blocks = []
            for rj in range(self.n_roles):
                c0, c1 = int(col_bounds[rj]), int(col_bounds[rj + 1])
                if c0 == c1:
                    continue
                proj = self.pair_block_proj[ri * self.n_roles + rj]
                col_blocks.append(proj(z_sorted[..., r0:r1, c0:c1, :]))
            row_blocks.append(mx.concatenate(col_blocks, axis=-2))
        delta_sorted = mx.concatenate(row_blocks, axis=-3)
        row_inv = mx.array(np.argsort(row_order))
        col_inv = mx.array(np.argsort(col_order))
        return mx.take(mx.take(delta_sorted, row_inv, axis=-3), col_inv, axis=-2)

    @staticmethod
    def _build_structural_pair_context(
        feat: dict[str, Any], role: mx.array, parent: mx.array
    ) -> dict[str, mx.array]:
        n_struct = role.shape[-1]
        polymer_type = feat.get("structural_polymer_type")
        prev_parent = feat.get("prev_parent_residue_idx")
        next_parent = feat.get("next_parent_residue_idx")
        return {
            "parent": parent,
            "asym_id": mx.take(feat["asym_id"], parent, axis=-1),
            "polymer_type": (
                mx.zeros((n_struct,), dtype=mx.int32)
                if polymer_type is None
                else polymer_type.astype(mx.int32)
            ),
            "is_backbone": _isin(role, _BACKBONE_ROLES),
            "is_sidechain": _isin(role, _SIDECHAIN_ROLES),
            "is_base": _isin(role, _BASE_ROLES),
            "prev_parent": (
                mx.full((n_struct,), -1, dtype=mx.int32)
                if prev_parent is None
                else prev_parent.astype(mx.int32)
            ),
            "next_parent": (
                mx.full((n_struct,), -1, dtype=mx.int32)
                if next_parent is None
                else next_parent.astype(mx.int32)
            ),
        }

    @staticmethod
    def _build_structural_pair_features_for_rows(
        ctx: dict[str, mx.array], rows: slice
    ) -> dict[str, mx.array]:
        """Boolean pair features and role-pair type for ``ctx`` rows ``rows`` vs all columns."""
        row = {k: v[rows] for k, v in ctx.items()}
        same_parent_residue = row["parent"][:, None] == ctx["parent"][None, :]
        same_chain = row["asym_id"][:, None] == ctx["asym_id"][None, :]
        same_polymer_type = (row["polymer_type"][:, None] == ctx["polymer_type"][None, :]) & (
            row["polymer_type"][:, None] > 0
        )
        row_bb, col_bb = row["is_backbone"][:, None], ctx["is_backbone"][None, :]
        row_split = (row["is_sidechain"] | row["is_base"])[:, None]
        col_split = (ctx["is_sidechain"] | ctx["is_base"])[None, :]
        same_residue_twin = same_parent_residue & ((row_bb & col_split) | (col_bb & row_split))
        bb_pair = row_bb & col_bb & same_chain
        prev_bb_chain = bb_pair & (row["prev_parent"][:, None] == ctx["parent"][None, :])
        next_bb_chain = bb_pair & (row["next_parent"][:, None] == ctx["parent"][None, :])

        def category(c: dict[str, mx.array]) -> mx.array:
            cat = mx.full(c["parent"].shape, 3, dtype=mx.int32)
            cat = mx.where(c["is_base"], 2, cat)
            cat = mx.where(c["is_sidechain"], 1, cat)
            return mx.where(c["is_backbone"], 0, cat)

        role_pair_type = _ROLE_PAIR_TYPE_TABLE[category(row)[:, None], category(ctx)[None, :]]
        return {
            "same_parent_residue": same_parent_residue,
            "same_residue_twin": same_residue_twin,
            "prev_bb_chain": prev_bb_chain,
            "next_bb_chain": next_bb_chain,
            "role_pair_type": role_pair_type,
            "same_chain": same_chain,
            "same_polymer_type": same_polymer_type,
        }

    def _make_pair_init_bias(self, pf: dict[str, mx.array], dtype: mx.Dtype) -> mx.array:
        bias = (
            self.same_parent_embedding(pf["same_parent_residue"].astype(mx.int32))
            + self.same_residue_twin_embedding(pf["same_residue_twin"].astype(mx.int32))
            + self.prev_bb_chain_embedding(pf["prev_bb_chain"].astype(mx.int32))
            + self.next_bb_chain_embedding(pf["next_bb_chain"].astype(mx.int32))
            + self.role_pair_type_embedding(pf["role_pair_type"])
        )
        return bias.astype(dtype)

    def _make_attention_bias(self, pf: dict[str, mx.array], dtype: mx.Dtype) -> mx.array:
        bias = (
            self.attn_bias_same_parent * pf["same_parent_residue"]
            + self.attn_bias_same_residue_twin * pf["same_residue_twin"]
            + self.attn_bias_prev_bb_chain * pf["prev_bb_chain"]
            + self.attn_bias_next_bb_chain * pf["next_bb_chain"]
            + self.attn_bias_role_pair_type[pf["role_pair_type"]]
        )
        return bias.astype(dtype)

    def __call__(
        self,
        feat: dict[str, Any],
        s_inputs_res: mx.array,
        s_res: mx.array,
        z_res: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, dict[str, mx.array]]:
        """
        Args:
            feat: needs ``parent_residue_idx``, ``subtoken_role_id`` ``[N_struct]``,
                ``residue_index``, ``asym_id`` ``[N_res]``; optional
                ``structural_polymer_type``, ``prev/next_parent_residue_idx`` ``[N_struct]``.
            s_inputs_res: ``[..., N_res, c_s_inputs]``; s_res: ``[..., N_res, c_s]``
            z_res: ``[..., N_res, N_res, c_z]``
        Returns:
            ``s_inputs_struct``, ``s_struct``, ``z_struct`` and pair features including
            ``structural_pair_attn_bias`` ``[N_struct, N_struct]``.
        """
        parent = feat["parent_residue_idx"].astype(mx.int32)
        role = feat["subtoken_role_id"].astype(mx.int32)
        n_struct = role.shape[-1]

        s_inputs_struct = mx.take(s_inputs_res, parent, axis=-2) + self.single_input_role_embedding(
            role
        ).astype(s_inputs_res.dtype)
        s_parent = mx.take(s_res, parent, axis=-2)
        ln, lin_1, lin_2 = self.single_split_mlp
        s_struct = (
            s_parent
            + lin_2(nn.silu(lin_1(ln(s_parent))))
            + self.single_role_embedding(role).astype(s_parent.dtype)
        )

        ctx = self._build_structural_pair_context(feat, role, parent)
        role_np = np.asarray(role)
        z_cols = mx.take(z_res, parent, axis=-2)
        chunk = min(self.pair_chunk_size or n_struct, n_struct)
        z_chunks, feature_chunks = [], []
        for start in range(0, n_struct, chunk):
            rows = slice(start, min(start + chunk, n_struct))
            pf = self._build_structural_pair_features_for_rows(ctx, rows)
            z_chunk = mx.take(z_cols, parent[rows], axis=-3)
            if self.pair_projection_mode == "full":
                z_chunk = z_chunk + self._pair_project_by_role_full(z_chunk, role_np[rows], role_np)
            z_chunks.append(z_chunk + self._make_pair_init_bias(pf, z_chunk.dtype))
            pf["structural_pair_attn_bias"] = self._make_attention_bias(pf, z_chunk.dtype)
            feature_chunks.append(pf)
        z_struct = mx.concatenate(z_chunks, axis=-3)
        pair_features = {
            k: mx.concatenate([c[k] for c in feature_chunks], axis=0) for k in feature_chunks[0]
        }
        pair_features["residue_index"] = mx.take(feat["residue_index"], parent, axis=-1)
        return s_inputs_struct, s_struct, z_struct, pair_features
