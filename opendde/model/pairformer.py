# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Pairformer, MSA module and template embedder (AF3 Algorithms 8, 10, 16, 17)."""

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from opendde.data.constants import STD_RESIDUES_WITH_GAP
from opendde.model.msa_sampling import subsample_msa_feature_dict_valid_first
from opendde.model.primitives import LayerNorm, LinearNoBias, Transition
from opendde.model.transformer import AttentionPairBias
from opendde.model.triangular import (
    OuterProductMean,
    TriangleAttention,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from opendde.model.utils import expand_at_dim


def _one_hot(x: mx.array, num_classes: int) -> mx.array:
    return (x[..., None] == mx.arange(num_classes)).astype(mx.float32)


class PairformerBlock(nn.Module):
    """Algorithm 17 lines 2-8: triangle updates on ``z`` and pair-biased attention on ``s``.

    ``c_s = 0`` builds a pair-only block (used by the MSA module and template embedder).
    """

    def __init__(
        self,
        n_heads: int = 16,
        c_z: int = 128,
        c_s: int = 384,
        c_hidden_mul: int = 128,
        c_hidden_pair_att: int = 32,
        no_heads_pair: int = 4,
        num_intermediate_factor: int = 4,
        hidden_scale_up: bool = False,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.c_s = c_s
        if hidden_scale_up:
            no_heads_pair = c_z // c_hidden_pair_att
            c_hidden_mul = c_z
        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z=c_z, c_hidden=c_hidden_mul)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z=c_z, c_hidden=c_hidden_mul)
        self.tri_att_start = TriangleAttention(c_z, c_hidden_pair_att, no_heads_pair, starting=True)
        self.tri_att_end = TriangleAttention(c_z, c_hidden_pair_att, no_heads_pair, starting=False)
        self.pair_transition = Transition(c_in=c_z, n=num_intermediate_factor)
        if c_s > 0:
            self.attention_pair_bias = AttentionPairBias(
                has_s=False, create_offset_ln_z=True, n_heads=n_heads, c_a=c_s, c_z=c_z
            )
            self.single_transition = Transition(c_in=c_s, n=4)

    def __call__(
        self,
        s: Optional[mx.array],
        z: mx.array,
        pair_mask: Optional[mx.array] = None,
        chunk_size: Optional[int] = None,
        extra_attn_bias: Optional[mx.array] = None,
    ) -> tuple[Optional[mx.array], mx.array]:
        """``s [..., N, c_s]`` (or None), ``z [..., N, N, c_z]``, ``pair_mask [..., N, N]``."""
        z = z + self.tri_mul_out(z, mask=pair_mask, chunk_size=chunk_size)
        z = z + self.tri_mul_in(z, mask=pair_mask, chunk_size=chunk_size)
        z = z + self.tri_att_start(z, mask=pair_mask, chunk_size=chunk_size)
        z = z + self.tri_att_end(z, mask=pair_mask, chunk_size=chunk_size)
        z = z + self.pair_transition(z)
        if self.c_s > 0:
            s = s + self.attention_pair_bias(a=s, s=None, z=z, extra_attn_bias=extra_attn_bias)
            s = s + self.single_transition(s)
        return s, z


class PairformerStack(nn.Module):
    """Algorithm 17: a stack of ``PairformerBlock``."""

    def __init__(
        self,
        n_blocks: int = 48,
        n_heads: int = 16,
        c_z: int = 128,
        c_s: int = 384,
        num_intermediate_factor: int = 4,
        hidden_scale_up: bool = False,
    ) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.blocks = [
            PairformerBlock(
                n_heads=n_heads,
                c_z=c_z,
                c_s=c_s,
                num_intermediate_factor=num_intermediate_factor,
                hidden_scale_up=hidden_scale_up,
            )
            for _ in range(n_blocks)
        ]

    def __call__(
        self,
        s: Optional[mx.array],
        z: mx.array,
        pair_mask: Optional[mx.array] = None,
        chunk_size: Optional[int] = None,
        extra_attn_bias: Optional[mx.array] = None,
    ) -> tuple[Optional[mx.array], mx.array]:
        for block in self.blocks:
            s, z = block(s, z, pair_mask, chunk_size=chunk_size, extra_attn_bias=extra_attn_bias)
            mx.eval([t for t in (s, z) if t is not None])
        return s, z


class MSAPairWeightedAveraging(nn.Module):
    """Algorithm 10: gated averaging of MSA values with pair-derived weights."""

    def __init__(self, c_m: int = 64, c: int = 32, c_z: int = 128, n_heads: int = 8) -> None:
        super().__init__()
        self.c = c
        self.n_heads = n_heads
        self.layernorm_m = LayerNorm(c_m)
        self.linear_no_bias_mv = LinearNoBias(c_m, c * n_heads)
        self.layernorm_z = LayerNorm(c_z)
        self.linear_no_bias_z = LinearNoBias(c_z, n_heads)
        self.linear_no_bias_mg = LinearNoBias(c_m, c * n_heads, initializer="zeros")
        self.linear_no_bias_out = LinearNoBias(c * n_heads, c_m, initializer="zeros")

    def __call__(self, m: mx.array, z: mx.array) -> mx.array:
        """``m [..., N_msa, N, c_m]``, ``z [..., N, N, c_z]`` -> update of ``m``."""
        m = self.layernorm_m(m)
        heads = (*m.shape[:-1], self.n_heads, self.c)
        v = self.linear_no_bias_mv(m).reshape(heads)
        g = mx.sigmoid(self.linear_no_bias_mg(m)).reshape(heads)
        w = mx.softmax(self.linear_no_bias_z(self.layernorm_z(z)), axis=-2)  # [..., i, j, h]
        o = g * mx.einsum("...ijh,...mjhc->...mihc", w, v)
        return self.linear_no_bias_out(o.reshape(*o.shape[:-2], self.n_heads * self.c))


class MSAStack(nn.Module):
    """Algorithm 8 lines 7-8, applied to row chunks of ``msa_chunk_size`` sequences."""

    def __init__(
        self, c_m: int = 64, c_z: int = 128, c: int = 8, msa_chunk_size: Optional[int] = 2048
    ) -> None:
        super().__init__()
        self.msa_pair_weighted_averaging = MSAPairWeightedAveraging(c_m=c_m, c=c, c_z=c_z)
        self.transition_m = Transition(c_in=c_m, n=4)
        self.msa_chunk_size = msa_chunk_size

    def __call__(self, m: mx.array, z: mx.array) -> mx.array:
        num_msa = m.shape[-3]
        chunk = self.msa_chunk_size or max(num_msa, 1)
        outs = []
        for start in range(0, num_msa, chunk):
            m_chunk = m[..., start : start + chunk, :, :]
            m_chunk = m_chunk + self.msa_pair_weighted_averaging(m_chunk, z)
            outs.append(m_chunk + self.transition_m(m_chunk))
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=-3)


class MSABlock(nn.Module):
    """Boltz-style MSA block: MSA stack, then outer product mean, then a pair block."""

    def __init__(
        self,
        c_m: int = 64,
        c_z: int = 128,
        c_hidden: int = 32,
        is_last_block: bool = False,
        msa_chunk_size: Optional[int] = 2048,
        hidden_scale_up: bool = False,
    ) -> None:
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.is_last_block = is_last_block
        self.msa_stack = MSAStack(c_m=c_m, c_z=c_z, msa_chunk_size=msa_chunk_size)
        self.outer_product_mean_msa = OuterProductMean(c_m=c_m, c_z=c_z, c_hidden=c_hidden)
        self.pair_stack = PairformerBlock(c_z=c_z, c_s=0, hidden_scale_up=hidden_scale_up)

    def __call__(
        self,
        m: mx.array,
        z: mx.array,
        pair_mask: Optional[mx.array] = None,
        chunk_size: Optional[int] = None,
    ) -> tuple[Optional[mx.array], mx.array]:
        m = self.msa_stack(m, z)
        z = z + self.outer_product_mean_msa(m, chunk_size=chunk_size)
        _, z = self.pair_stack(None, z, pair_mask, chunk_size=chunk_size)
        return (None if self.is_last_block else m), z


class MSAModule(nn.Module):
    """Algorithm 8 (Boltz ordering): embed a random MSA subset and refine ``z``."""

    def __init__(
        self,
        n_blocks: int = 4,
        c_m: int = 64,
        c_z: int = 128,
        c_s_inputs: int = 449,
        msa_chunk_size: Optional[int] = 2048,
        msa_configs: Optional[dict[str, Any]] = None,
        hidden_scale_up: bool = False,
    ) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.c_m = c_m
        self.c_s_inputs = c_s_inputs
        self.msa_chunk_size = msa_chunk_size
        self.input_feature = {"msa": 32, "has_deletion": 1, "deletion_value": 1}
        if msa_configs is None or "msa_depth" not in msa_configs:
            raise ValueError("MSA config must define msa_depth.")
        self.msa_depth = int(msa_configs["msa_depth"])
        if self.msa_depth <= 0:
            raise ValueError("MSA msa_depth must be positive.")
        self.linear_no_bias_m = LinearNoBias(sum(self.input_feature.values()), c_m)
        self.linear_no_bias_s = LinearNoBias(c_s_inputs, c_m)
        self.blocks = [
            MSABlock(
                c_m=c_m,
                c_z=c_z,
                is_last_block=(i + 1 == n_blocks),
                msa_chunk_size=msa_chunk_size,
                hidden_scale_up=hidden_scale_up,
            )
            for i in range(n_blocks)
        ]

    def _prepare_msa_sample(self, feat: dict[str, Any], s_inputs: mx.array) -> Optional[mx.array]:
        if self.n_blocks < 1 or "msa" not in feat or feat["msa"].ndim < 2:
            return None
        msa_feat = subsample_msa_feature_dict_valid_first(
            feat_dict=feat,
            dim_dict={name: -2 for name in self.input_feature},
            num_msa=self.msa_depth,
            msa_mask=feat.get("msa_mask"),
            gap_token=self.input_feature["msa"] - 1,
        )
        msa_feat["msa"] = _one_hot(msa_feat["msa"], self.input_feature["msa"])
        shape = msa_feat["msa"].shape[:-1]
        msa_sample = mx.concatenate(
            [
                msa_feat[name].reshape(*shape, d).astype(mx.float32)
                for name, d in self.input_feature.items()
            ],
            axis=-1,
        )
        msa_sample = msa_sample.astype(self.linear_no_bias_m.weight.dtype)
        return self.linear_no_bias_m(msa_sample) + self.linear_no_bias_s(s_inputs)

    def __call__(
        self,
        feat: dict[str, Any],
        z: mx.array,
        s_inputs: mx.array,
        pair_mask: Optional[mx.array] = None,
        chunk_size: Optional[int] = None,
    ) -> mx.array:
        """Returns the updated ``z [..., N, N, c_z]`` (unchanged when no MSA is present)."""
        m = self._prepare_msa_sample(feat, s_inputs)
        if m is None:
            return z
        for block in self.blocks:
            m, z = block(m, z, pair_mask, chunk_size=chunk_size)
            mx.eval([t for t in (m, z) if t is not None])
        return z


class TemplateEmbedder(nn.Module):
    """Algorithm 16: average of per-template pair embeddings refined by a pair stack."""

    def __init__(
        self,
        n_blocks: int = 2,
        c: int = 64,
        c_z: int = 128,
        num_intermediate_factor: int = 2,
        hidden_scale_up: bool = False,
    ) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.c = c
        self.c_z = c_z
        self.input_feature1 = {
            "template_distogram": 39,
            "template_backbone_frame_mask": 1,
            "template_unit_vector": 3,
            "template_pseudo_beta_mask": 1,
        }
        self.input_feature2 = {"template_restype_i": 32, "template_restype_j": 32}
        self.distogram = {"max_bin": 50.75, "min_bin": 3.25, "no_bins": 39}
        self.inf = 100000.0
        self.linear_no_bias_z = LinearNoBias(c_z, c)
        self.layernorm_z = LayerNorm(c_z)
        self.linear_no_bias_a = LinearNoBias(
            sum(self.input_feature1.values()) + sum(self.input_feature2.values()), c
        )
        self.pairformer_stack = PairformerStack(
            c_s=0,
            c_z=c,
            n_blocks=n_blocks,
            num_intermediate_factor=num_intermediate_factor,
            hidden_scale_up=hidden_scale_up,
        )
        self.layernorm_v = LayerNorm(c)
        self.linear_no_bias_u = LinearNoBias(c, c_z)

    def __call__(
        self,
        feat: dict[str, Any],
        z: mx.array,
        pair_mask: Optional[mx.array] = None,
        chunk_size: Optional[int] = None,
    ) -> Optional[mx.array]:
        """Returns the template update ``[N, N, c_z]`` or ``None`` when templates are unused."""
        if "template_aatype" not in feat or self.n_blocks < 1:
            return None
        asym_id = feat["asym_id"]
        multichain_mask = (asym_id[:, None] == asym_id[None, :]).astype(z.dtype)
        num_templates = feat["template_aatype"].shape[0]
        if pair_mask is None:
            pair_mask = mx.ones(z.shape[:-1], dtype=z.dtype)
        z = self.layernorm_z(z)
        u = self.single_template_forward(0, feat, z, pair_mask, multichain_mask, chunk_size)
        for template_id in range(1, num_templates):
            u = u + self.single_template_forward(
                template_id, feat, z, pair_mask, multichain_mask, chunk_size
            )
        u = u / (1e-7 + num_templates)
        return self.linear_no_bias_u(nn.relu(u))

    def single_template_forward(
        self,
        template_id: int,
        feat: dict[str, Any],
        z: mx.array,
        pair_mask: mx.array,
        multichain_mask: mx.array,
        chunk_size: Optional[int] = None,
    ) -> mx.array:
        """Embed one template (``z`` already layer-normed) -> ``[N, N, c]``."""
        mask_2d = multichain_mask * pair_mask
        aatype = _one_hot(feat["template_aatype"][template_id], len(STD_RESIDUES_WITH_GAP))
        n = z.shape[0]
        at = mx.concatenate(
            [
                feat["template_distogram"][template_id] * mask_2d[..., None],
                (feat["template_pseudo_beta_mask"][template_id] * mask_2d)[..., None],
                expand_at_dim(aatype, dim=-3, n=n),
                expand_at_dim(aatype, dim=-2, n=n),
                feat["template_unit_vector"][template_id] * mask_2d[..., None],
                (feat["template_backbone_frame_mask"][template_id] * mask_2d)[..., None],
            ],
            axis=-1,
        ).astype(z.dtype)
        v = self.linear_no_bias_z(z) + self.linear_no_bias_a(at.astype(z.dtype))
        _, v = self.pairformer_stack(None, v, pair_mask, chunk_size=chunk_size)
        return self.layernorm_v(v)
