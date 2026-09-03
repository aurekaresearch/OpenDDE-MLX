# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Diffusion transformer and atom attention encoder/decoder (AF3 Algorithms 5, 6, 7, 23, 24, 25)."""

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from opendde.model.primitives import (
    AdaptiveLayerNorm,
    Attention,
    BiasInitLinear,
    FusedWeights,
    LayerNorm,
    LinearNoBias,
    gather_pair_embedding_in_dense_trunk,
    rearrange_qk_to_dense_trunk,
)
from opendde.model.utils import (
    aggregate_atom_to_token,
    broadcast_token_to_atom,
)


class AttentionPairBias(nn.Module):
    """Algorithm 24: attention over ``a`` biased by the pair representation ``z``.

    ``has_s`` selects adaptive layer norm conditioned on ``s`` (diffusion) versus a
    plain layer norm (Pairformer single track).
    """

    def __init__(
        self,
        has_s: bool = True,
        create_offset_ln_z: bool = False,
        n_heads: int = 16,
        c_a: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        biasinit: float = -2.0,
        cross_attention_mode: bool = False,
    ) -> None:
        super().__init__()
        assert c_a % n_heads == 0
        self.n_heads = n_heads
        self.has_s = has_s
        self.cross_attention_mode = cross_attention_mode
        if has_s:
            self.layernorm_a = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
            if cross_attention_mode:
                self.layernorm_kv = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
        else:
            self.layernorm_a = LayerNorm(c_a)
            if cross_attention_mode:
                self.layernorm_kv = LayerNorm(c_a)
        self.attention = Attention(
            c_q=c_a,
            c_k=c_a,
            c_v=c_a,
            c_hidden=c_a // n_heads,
            num_heads=n_heads,
            gating=True,
            q_linear_bias=True,
            zero_init=not has_s,
        )
        self.layernorm_z = LayerNorm(c_z, create_offset=create_offset_ln_z)
        self.linear_nobias_z = LinearNoBias(c_z, n_heads)
        if has_s:
            self.linear_a_last = BiasInitLinear(c_s, c_a, biasinit=biasinit)

    def pair_bias(
        self, z: mx.array, z_is_normalized: bool = False, n_pair_dims: int = 2
    ) -> mx.array:
        """Project ``z [..., N, N, c_z]`` (or a local-window block tensor) to per-head biases.

        With ``z_is_normalized`` the caller already applied the (offset-free)
        LayerNorm once, so only the folded ``linear * ln_weight`` matmul remains.
        Returns ``[..., H, N, N]`` (``n_pair_dims=2``) or ``[..., H, n_trunks, n_q, n_k]`` (3).
        """
        if z_is_normalized:
            bias = z @ (self.linear_nobias_z.weight * self.layernorm_z.weight[None, :]).T
        else:
            bias = self.linear_nobias_z(self.layernorm_z(z))
        return mx.moveaxis(bias, -1, -(n_pair_dims + 1))

    @staticmethod
    def _add_extra_bias(bias: mx.array, extra: Optional[mx.array]) -> mx.array:
        if extra is None:
            return bias
        while extra.ndim < bias.ndim - 1:
            extra = extra[None]
        if extra.ndim == bias.ndim - 1:
            extra = mx.expand_dims(extra, -3)
        return bias + extra.astype(bias.dtype)

    def __call__(
        self,
        a: mx.array,
        s: Optional[mx.array],
        z: mx.array,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        extra_attn_bias: Optional[mx.array] = None,
        z_is_normalized: bool = False,
        pair_bias: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            a: ``[..., N, c_a]``; s: ``[..., N, c_s]`` (when ``has_s``)
            z: ``[..., N, N, c_z]`` or ``[..., n_trunks, n_queries, n_keys, c_z]`` for local attention.
            pair_bias: optional precomputed ``pair_bias(z)`` (shared across diffusion steps).
        """
        a = self.layernorm_a(a, s) if self.has_s else self.layernorm_a(a)
        if self.cross_attention_mode:
            kv = self.layernorm_kv(a, s) if self.has_s else self.layernorm_kv(a)
        else:
            kv = a
        local = bool(n_queries and n_keys)
        n_pair_dims = 3 if local else 2
        bias = (
            pair_bias
            if pair_bias is not None
            else self.pair_bias(z, z_is_normalized, n_pair_dims=n_pair_dims)
        )
        # insert missing sample/batch dims before the head dim so bias broadcasts against a
        while bias.ndim < a.ndim - 2 + 1 + n_pair_dims:
            bias = mx.expand_dims(bias, bias.ndim - (1 + n_pair_dims))
        if local:
            a = self.attention(a, kv, trunked_attn_bias=bias, n_queries=n_queries, n_keys=n_keys)
        else:
            if pair_bias is None:  # a cached bias already includes the extra term
                bias = self._add_extra_bias(bias, extra_attn_bias)
            a = self.attention(a, kv, attn_bias=bias)
        if self.has_s:
            a = mx.sigmoid(self.linear_a_last(s)) * a
        return a


class ConditionedTransitionBlock(nn.Module):
    """Algorithm 25: SwiGLU transition with adaptive layer norm and adaLN-Zero gate."""

    def __init__(self, c_a: int, c_s: int, n: int = 2, biasinit: float = -2.0) -> None:
        super().__init__()
        self.adaln = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
        self.linear_nobias_a1 = LinearNoBias(c_a, n * c_a)
        self.linear_nobias_a2 = LinearNoBias(c_a, n * c_a)
        self.linear_nobias_b = LinearNoBias(n * c_a, c_a)
        self.linear_s = BiasInitLinear(c_s, c_a, biasinit=biasinit)
        self._fused = FusedWeights()

    def __call__(self, a: mx.array, s: mx.array) -> mx.array:
        a = self.adaln(a, s)
        a1, a2 = mx.split(
            a @ self._fused(self.linear_nobias_a1, self.linear_nobias_a2).T, 2, axis=-1
        )
        return mx.sigmoid(self.linear_s(s)) * self.linear_nobias_b(nn.silu(a1) * a2)


class DiffusionTransformerBlock(nn.Module):
    """Algorithm 23 (one block): pair-biased attention + conditioned transition."""

    def __init__(
        self,
        c_a: int,
        c_s: int,
        c_z: int,
        n_heads: int,
        biasinit: float = -2.0,
        cross_attention_mode: bool = False,
    ) -> None:
        super().__init__()
        self.attention_pair_bias = AttentionPairBias(
            has_s=True,
            create_offset_ln_z=False,
            n_heads=n_heads,
            c_a=c_a,
            c_s=c_s,
            c_z=c_z,
            biasinit=biasinit,
            cross_attention_mode=cross_attention_mode,
        )
        self.conditioned_transition_block = ConditionedTransitionBlock(
            n=2, c_a=c_a, c_s=c_s, biasinit=biasinit
        )

    def __call__(
        self,
        a: mx.array,
        s: mx.array,
        z: mx.array,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        extra_attn_bias: Optional[mx.array] = None,
        z_is_normalized: bool = False,
        pair_bias: Optional[mx.array] = None,
    ) -> mx.array:
        a = a + self.attention_pair_bias(
            a=a,
            s=s,
            z=z,
            n_queries=n_queries,
            n_keys=n_keys,
            extra_attn_bias=extra_attn_bias,
            z_is_normalized=z_is_normalized,
            pair_bias=pair_bias,
        )
        return a + self.conditioned_transition_block(a=a, s=s)


class DiffusionTransformer(nn.Module):
    """Algorithm 23: a stack of ``DiffusionTransformerBlock``."""

    def __init__(
        self,
        c_a: int,
        c_s: int,
        c_z: int,
        n_blocks: int,
        n_heads: int,
        cross_attention_mode: bool = False,
    ) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.blocks = [
            DiffusionTransformerBlock(
                n_heads=n_heads,
                c_a=c_a,
                c_s=c_s,
                c_z=c_z,
                cross_attention_mode=cross_attention_mode,
            )
            for _ in range(n_blocks)
        ]

    def pair_bias_cache(
        self, z: mx.array, z_is_normalized: bool = False, extra_attn_bias: Optional[mx.array] = None
    ) -> list[mx.array]:
        """Precompute every block's ``[H, N, N]`` pair bias once per diffusion rollout."""
        cache = []
        for block in self.blocks:
            apb = block.attention_pair_bias
            cache.append(apb._add_extra_bias(apb.pair_bias(z, z_is_normalized), extra_attn_bias))
        mx.eval(cache)
        return cache

    def __call__(
        self,
        a: mx.array,
        s: mx.array,
        z: mx.array,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        extra_attn_bias: Optional[mx.array] = None,
        z_is_normalized: bool = False,
        pair_bias_cache: Optional[list[mx.array]] = None,
    ) -> mx.array:
        for i, block in enumerate(self.blocks):
            a = block(
                a,
                s,
                z,
                n_queries=n_queries,
                n_keys=n_keys,
                extra_attn_bias=extra_attn_bias,
                z_is_normalized=z_is_normalized,
                pair_bias=None if pair_bias_cache is None else pair_bias_cache[i],
            )
        return a


class AtomTransformer(nn.Module):
    """Algorithm 7: local (windowed) transformer over atoms."""

    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_blocks: int = 3,
        n_heads: int = 4,
        n_queries: int = 32,
        n_keys: int = 128,
    ) -> None:
        super().__init__()
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.diffusion_transformer = DiffusionTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_a=c_atom,
            c_s=c_atom,
            c_z=c_atompair,
            cross_attention_mode=True,
        )

    def __call__(self, q: mx.array, c: mx.array, p: mx.array) -> mx.array:
        """``q, c [..., N_atom, c_atom]``, ``p [..., n_trunks, n_queries, n_keys, c_atompair]``."""
        assert p.shape[-3] == self.n_queries and p.shape[-2] == self.n_keys
        return self.diffusion_transformer(
            a=q, s=c, z=p, n_queries=self.n_queries, n_keys=self.n_keys
        )


class AtomAttentionEncoder(nn.Module):
    """Algorithm 5: embed atoms (and optionally noisy coordinates) into token features."""

    def __init__(
        self,
        has_coords: bool,
        c_token: int,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_s: int = 384,
        c_z: int = 128,
        n_blocks: int = 3,
        n_heads: int = 4,
        n_queries: int = 32,
        n_keys: int = 128,
    ) -> None:
        super().__init__()
        self.has_coords = has_coords
        self.n_queries = n_queries
        self.n_keys = n_keys
        # ref_mask (1) + ref_element (128) + ref_atom_name_chars (4 * 64)
        self.linear_no_bias_ref_pos = LinearNoBias(3, c_atom, precision=True)
        self.linear_no_bias_ref_charge = LinearNoBias(1, c_atom)
        self.linear_no_bias_f = LinearNoBias(1 + 128 + 4 * 64, c_atom)
        self.linear_no_bias_d = LinearNoBias(3, c_atompair, precision=True)
        self.linear_no_bias_invd = LinearNoBias(1, c_atompair)
        self.linear_no_bias_v = LinearNoBias(1, c_atompair)
        if has_coords:
            self.layernorm_s = LayerNorm(c_s, create_offset=False)
            self.linear_no_bias_s = LinearNoBias(c_s, c_atom, initializer="zeros", precision=True)
            self.layernorm_z = LayerNorm(c_z, create_offset=False)
            self.linear_no_bias_z = LinearNoBias(
                c_z, c_atompair, initializer="zeros", precision=True
            )
            self.linear_no_bias_r = LinearNoBias(3, c_atom, precision=True)
        self.linear_no_bias_cl = LinearNoBias(c_atom, c_atompair)
        self.linear_no_bias_cm = LinearNoBias(c_atom, c_atompair)
        self.small_mlp = [
            LinearNoBias(c_atompair, c_atompair),
            LinearNoBias(c_atompair, c_atompair),
            LinearNoBias(c_atompair, c_atompair, initializer="zeros"),
        ]
        self.atom_transformer = AtomTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_queries=n_queries,
            n_keys=n_keys,
        )
        self.linear_no_bias_q = LinearNoBias(c_atom, c_token)

    def _small_mlp(self, p: mx.array) -> mx.array:
        for linear in self.small_mlp:
            p = linear(nn.relu(p))
        return p

    def _add_token_pair_context(
        self, p_lm: mx.array, z: mx.array, atom_to_token_idx: mx.array
    ) -> mx.array:
        """``p_lm + Linear(LayerNorm(z))`` gathered on every atom local window (streamed by windows)."""
        idx_q, idx_k, _ = rearrange_qk_to_dense_trunk(
            atom_to_token_idx,
            atom_to_token_idx,
            -1,
            -1,
            self.n_queries,
            self.n_keys,
            compute_mask=False,
        )
        p_lm = p_lm[..., None, :, :, :, :]  # broadcast slot for N_sample
        chunks = []
        window_chunk = 64
        for start in range(0, idx_q.shape[0], window_chunk):
            z_pair = gather_pair_embedding_in_dense_trunk(
                z, idx_q[start : start + window_chunk], idx_k[start : start + window_chunk]
            )
            z_pair = self.linear_no_bias_z(self.layernorm_z(z_pair))
            if z_pair.ndim == p_lm.ndim - 1:
                z_pair = z_pair[..., None, :, :, :, :]
            chunks.append(p_lm[..., start : start + window_chunk, :, :, :] + z_pair)
        return mx.concatenate(chunks, axis=-4)

    def prepare_cache(
        self,
        ref_pos: mx.array,
        ref_charge: mx.array,
        ref_mask: mx.array,
        ref_element: mx.array,
        ref_atom_name_chars: mx.array,
        atom_to_token_idx: mx.array,
        d_lm: mx.array,
        v_lm: mx.array,
        pad_info: dict[str, Any],
        z: Optional[mx.array] = None,
    ) -> tuple[mx.array, mx.array]:
        """Coordinate-independent atom single (``c_l``) and pair (``p_lm``) embeddings."""
        batch_shape = ref_pos.shape[:-2]
        n_atom = ref_pos.shape[-2]
        dtype = self.linear_no_bias_f.weight.dtype
        c_l = self.linear_no_bias_ref_pos(ref_pos.astype(dtype)) + self.linear_no_bias_ref_charge(
            mx.arcsinh(ref_charge.astype(mx.float32)).reshape(*batch_shape, n_atom, 1).astype(dtype)
        )
        f = mx.concatenate(
            [
                ref_mask.reshape(*batch_shape, n_atom, 1),
                ref_element.reshape(*batch_shape, n_atom, 128),
                ref_atom_name_chars.reshape(*batch_shape, n_atom, 4 * 64),
            ],
            axis=-1,
        ).astype(dtype)
        c_l = (c_l + self.linear_no_bias_f(f)) * ref_mask.reshape(*batch_shape, n_atom, 1).astype(
            dtype
        )

        v_lm = v_lm.astype(dtype)
        mask_trunked = pad_info["mask_trunked"][..., None].astype(dtype)
        p_lm = self.linear_no_bias_d(d_lm.astype(dtype)) * v_lm * mask_trunked
        inv_d = (1 / (1 + (d_lm.astype(mx.float32) ** 2).sum(-1, keepdims=True))).astype(dtype)
        p_lm = p_lm + self.linear_no_bias_invd(inv_d) * v_lm + self.linear_no_bias_v(v_lm)
        if z is not None:
            p_lm = self._add_token_pair_context(p_lm, z, atom_to_token_idx)
        return p_lm, c_l

    def __call__(
        self,
        atom_to_token_idx: mx.array,
        ref_pos: mx.array,
        ref_charge: mx.array,
        ref_mask: mx.array,
        ref_atom_name_chars: mx.array,
        ref_element: mx.array,
        d_lm: mx.array,
        v_lm: mx.array,
        pad_info: dict[str, Any],
        r_l: Optional[mx.array] = None,
        s: Optional[mx.array] = None,
        z: Optional[mx.array] = None,
        p_lm: Optional[mx.array] = None,
        c_l: Optional[mx.array] = None,
        conditioned: bool = False,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        ``conditioned=True`` means ``p_lm``/``c_l`` already went through :meth:`condition`.

        Returns:
            a ``[..., (N_sample), N_token, c_token]``, q_l/c_l ``[..., (N_sample), N_atom, c_atom]``,
            p_lm ``[..., (N_sample), n_trunks, n_queries, n_keys, c_atompair]``
        """
        if p_lm is None or c_l is None:
            p_lm, c_l = self.prepare_cache(
                ref_pos,
                ref_charge,
                ref_mask,
                ref_element,
                ref_atom_name_chars,
                atom_to_token_idx,
                d_lm,
                v_lm,
                pad_info,
                z=z if self.has_coords else None,
            )
        if not conditioned:
            p_lm, c_l = self.condition(p_lm, c_l, s, atom_to_token_idx)
        q_l = c_l if r_l is None else c_l + self.linear_no_bias_r(r_l.astype(c_l.dtype))
        n_token = None if s is None else s.shape[-2]
        q_l = self.atom_transformer(q_l, c_l, p_lm)
        a = aggregate_atom_to_token(
            nn.relu(self.linear_no_bias_q(q_l)), atom_to_token_idx, n_token, "mean"
        )
        return a, q_l, c_l, p_lm

    def condition(
        self,
        p_lm: mx.array,
        c_l: mx.array,
        s: Optional[mx.array],
        atom_to_token_idx: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Add the trunk single embedding to ``c_l`` and the atom single context to ``p_lm``.

        This part of Algorithm 5 does not depend on the noisy coordinates, so a
        diffusion rollout computes it once and reuses it for every step.
        """
        if s is not None:
            c_l = c_l[..., None, :, :] + broadcast_token_to_atom(
                self.linear_no_bias_s(self.layernorm_s(s)), atom_to_token_idx
            )  # [..., N_sample, N_atom, c_atom]
        c_l_q, c_l_k, _ = rearrange_qk_to_dense_trunk(
            c_l, c_l, -2, -2, self.n_queries, self.n_keys, compute_mask=False
        )
        p_lm = (
            p_lm
            + self.linear_no_bias_cl(nn.relu(c_l_q))[..., :, :, None, :]
            + self.linear_no_bias_cm(nn.relu(c_l_k))[..., :, None, :, :]
        )
        return p_lm + self._small_mlp(p_lm), c_l


class AtomAttentionDecoder(nn.Module):
    """Algorithm 6: broadcast token activations back to atoms and predict coordinates."""

    def __init__(
        self,
        n_blocks: int = 3,
        n_heads: int = 4,
        c_token: int = 384,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_queries: int = 32,
        n_keys: int = 128,
    ) -> None:
        super().__init__()
        self.linear_no_bias_a = LinearNoBias(c_token, c_atom)
        self.layernorm_q = LayerNorm(c_atom, create_offset=False)
        self.linear_no_bias_out = LinearNoBias(c_atom, 3, precision=True)
        self.atom_transformer = AtomTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_queries=n_queries,
            n_keys=n_keys,
        )

    def __call__(
        self,
        atom_to_token_idx: mx.array,
        a: mx.array,
        q_skip: mx.array,
        c_skip: mx.array,
        p_skip: mx.array,
    ) -> mx.array:
        q = broadcast_token_to_atom(self.linear_no_bias_a(a), atom_to_token_idx) + q_skip
        q = self.atom_transformer(q, c_skip, p_skip)
        return self.linear_no_bias_out(self.layernorm_q(q))


def update_input_feature_dict(
    feat: dict[str, Any], n_queries: int = 32, n_keys: int = 128
) -> dict[str, Any]:
    """Algorithm 5 lines 1-3: local atom-pair offsets ``d_lm``, same-reference mask ``v_lm``."""
    (pos_q, uid_q), (pos_k, uid_k), pad_info = rearrange_qk_to_dense_trunk(
        [feat["ref_pos"], feat["ref_space_uid"]],
        [feat["ref_pos"], feat["ref_space_uid"]],
        dim_q=[-2, -1],
        dim_k=[-2, -1],
        n_queries=n_queries,
        n_keys=n_keys,
        compute_mask=True,
    )
    feat["d_lm"] = pos_q[..., :, None, :] - pos_k[..., None, :, :]  # [..., n_trunks, n_q, n_k, 3]
    feat["v_lm"] = (uid_q[..., :, None] == uid_k[..., None, :])[..., None].astype(mx.float32)
    feat["pad_info"] = pad_info
    return feat
