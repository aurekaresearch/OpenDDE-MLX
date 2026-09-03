# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Core layers shared by every OpenDDE module (AF3 Algorithms 11/24/25/26).

Parameter names mirror the reference PyTorch implementation so released
checkpoints load without renaming.
"""

import math
from typing import Any, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from opendde.model.utils import (
    flatten_final_dims,
    move_final_dim_to_dim,
    pad_at_dim,
    reshape_at_dim,
    unfold,
)


# Transition hidden activations are computed in row slabs to bound peak memory.
TRANSITION_CHUNK_ROWS = 65_536
# Smallest head dim handled by the fused scaled-dot-product-attention kernel.
FUSED_ATTENTION_HEAD_DIM = 64
# Additive attention masks use magnitudes like 1e9/1e10, which overflow to -inf in
# fp16 and turn a fully masked softmax row into NaN. Clamp to a value that is still
# finite in fp16 and just as saturating in the softmax.
FP16_BIAS_FLOOR = -6.0e4


def schedule(outs: list[mx.array]) -> None:
    """Queue the newest chunk and wait for the previous one (double buffering).

    Bounds peak memory to two chunks in flight while keeping the GPU busy.
    """
    mx.async_eval(outs[-1])
    if len(outs) > 1:
        mx.eval(outs[-2])


class Linear(nn.Module):
    """``y = x @ W.T + b`` with an optional fp32 compute path.

    ``precision=True`` keeps the layer in fp32 even when the rest of the model
    runs in bf16 (used for coordinate and conditioning projections).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        precision: bool = False,
        initializer: str = "default",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.precision = precision
        scale = math.sqrt(1.0 / in_features)
        self.weight = (
            mx.zeros((out_features, in_features))
            if initializer == "zeros"
            else mx.random.truncated_normal(-2 * scale, 2 * scale, (out_features, in_features))
        )
        if bias:
            self.bias = mx.zeros((out_features,))

    def __call__(self, x: mx.array) -> mx.array:
        if self.precision and x.dtype != mx.float32:
            y = (
                mx.addmm(self.bias, x.astype(mx.float32), self.weight.T)
                if "bias" in self
                else x.astype(mx.float32) @ self.weight.T
            )
            return y.astype(x.dtype)
        if "bias" in self:
            return mx.addmm(self.bias, x, self.weight.T)
        return x @ self.weight.T


def LinearNoBias(in_features: int, out_features: int, **kwargs: Any) -> Linear:
    return Linear(in_features, out_features, bias=False, **kwargs)


class BiasInitLinear(Linear):
    """Zero weight, constant bias initialisation (adaLN-Zero gates)."""

    def __init__(self, in_features: int, out_features: int, biasinit: float = 0.0) -> None:
        super().__init__(in_features, out_features, bias=True, initializer="zeros")
        self.bias = mx.full((out_features,), biasinit)


def LayerNorm(
    c_in: int, create_scale: bool = True, create_offset: bool = True, eps: float = 1e-5
) -> nn.LayerNorm:
    return nn.LayerNorm(c_in, eps=eps, affine=create_scale, bias=create_offset)


class FusedWeights:
    """Cache of concatenated projection weights (a plain object, not a parameter).

    Sibling projections of the same input are evaluated as one matmul; the fused
    copy is rebuilt whenever the source weights change (checkpoint load, dtype).
    """

    __slots__ = ("sources", "value")

    def __init__(self) -> None:
        self.sources: tuple[mx.array, ...] = ()
        self.value: Optional[mx.array] = None

    def __call__(self, *linears: "Linear") -> mx.array:
        weights = tuple(linear.weight for linear in linears)
        if self.value is None or any(a is not b for a, b in zip(weights, self.sources)):
            self.sources = weights
            self.value = mx.concatenate(weights)
        return self.value


class AdaptiveLayerNorm(nn.Module):
    """Algorithm 26: LayerNorm of ``a`` modulated by the single embedding ``s``."""

    def __init__(self, c_a: int = 768, c_s: int = 384) -> None:
        super().__init__()
        self.layernorm_a = LayerNorm(c_a, create_scale=False, create_offset=False)
        self.layernorm_s = LayerNorm(c_s, create_offset=False)
        self.linear_s = Linear(c_s, c_a, initializer="zeros")
        self.linear_nobias_s = LinearNoBias(c_s, c_a, initializer="zeros")
        self._fused = FusedWeights()

    def __call__(self, a: mx.array, s: mx.array) -> mx.array:
        a = self.layernorm_a(a)
        scale, shift = mx.split(
            self.layernorm_s(s) @ self._fused(self.linear_s, self.linear_nobias_s).T, 2, axis=-1
        )
        return mx.sigmoid(scale + self.linear_s.bias) * a + shift


class Transition(nn.Module):
    """Algorithm 11: SwiGLU transition ``Linear(silu(a(x)) * b(x))``."""

    def __init__(self, c_in: int, n: int) -> None:
        super().__init__()
        self.layernorm1 = LayerNorm(c_in)
        self.linear_no_bias_a = LinearNoBias(c_in, n * c_in)
        self.linear_no_bias_b = LinearNoBias(c_in, n * c_in)
        self.linear_no_bias = LinearNoBias(n * c_in, c_in, initializer="zeros")
        self._fused = FusedWeights()

    def __call__(self, x: mx.array) -> mx.array:
        flat = x.reshape(-1, x.shape[-1])
        outs = []
        for start in range(0, flat.shape[0], TRANSITION_CHUNK_ROWS):
            y = self.layernorm1(flat[start : start + TRANSITION_CHUNK_ROWS])
            a, b = mx.split(
                y @ self._fused(self.linear_no_bias_a, self.linear_no_bias_b).T, 2, axis=-1
            )
            outs.append(self.linear_no_bias(nn.silu(a) * b))
            if flat.shape[0] > TRANSITION_CHUNK_ROWS:
                schedule(outs)
        return (outs[0] if len(outs) == 1 else mx.concatenate(outs)).reshape(x.shape)


def attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    attn_bias: Optional[mx.array] = None,
) -> mx.array:
    """Scaled dot-product attention with a pre-scaled query and an additive bias.

    Runs in the input dtype; the fused Metal kernel accumulates in fp32.

    Args:
        q: ``[..., H, Q, d]`` (already divided by ``sqrt(d)``)
        k, v: ``[..., H, K, d]``
        attn_bias: ``[..., H or 1, Q, K]`` broadcastable to the leading dims of q.
    Returns:
        ``[..., H, Q, d]``
    """
    lead = q.shape[:-3]
    H, Q, d = q.shape[-3:]
    K = k.shape[-2]
    if k.shape[:-3] != lead:
        k = mx.broadcast_to(k, (*lead, *k.shape[-3:]))
        v = mx.broadcast_to(v, (*lead, *v.shape[-3:]))
    q4, k4, v4 = q.reshape(-1, H, Q, d), k.reshape(-1, H, K, d), v.reshape(-1, H, K, d)
    if d < FUSED_ATTENTION_HEAD_DIM:
        # zero-pad the head dim so the fused Metal kernel is used (scores are unchanged)
        pad = [(0, 0)] * 3 + [(0, FUSED_ATTENTION_HEAD_DIM - d)]
        q4, k4, v4 = (mx.pad(t, pad) for t in (q4, k4, v4))
    mask = None
    if attn_bias is not None:
        b_lead = attn_bias.shape[:-3]
        if any(s != 1 for s in b_lead) and b_lead != lead:
            attn_bias = mx.broadcast_to(attn_bias, (*lead, *attn_bias.shape[-3:]))
        if q.dtype == mx.float16:
            attn_bias = mx.maximum(attn_bias, mx.array(FP16_BIAS_FLOOR, attn_bias.dtype))
        # the fused kernel needs a contiguous mask; a strided view silently falls back
        mask = mx.contiguous(attn_bias.reshape(-1, attn_bias.shape[-3], Q, K).astype(q.dtype))
    out = mx.fast.scaled_dot_product_attention(q4, k4, v4, scale=1.0, mask=mask)[..., :d]
    return out.reshape(*lead, H, Q, d)


def rearrange_qk_to_dense_trunk(
    q: Union[mx.array, list[mx.array]],
    k: Union[mx.array, list[mx.array]],
    dim_q: Union[int, list[int]],
    dim_k: Union[int, list[int]],
    n_queries: int = 32,
    n_keys: int = 128,
    compute_mask: bool = True,
) -> tuple[Any, Any, dict[str, Any]]:
    """Split a sequence into dense local windows for atom attention.

    Queries are cut into ``n_trunks`` blocks of ``n_queries``; every block
    attends to a centred window of ``n_keys`` keys.

    Returns:
        q_trunked ``[..., n_trunks, n_queries, ...]``, k_trunked
        ``[..., n_trunks, n_keys, ...]`` and padding info (``mask_trunked``
        ``[n_trunks, n_queries, n_keys]`` marks valid key positions).
    """
    assert n_keys >= n_queries and n_queries % 2 == 0 and n_keys % 2 == 0

    def _as_list(x, dim):
        is_list = isinstance(x, list)
        xs = x if is_list else [x]
        dims = list(dim) if is_list else [dim]
        dims = [d if d >= 0 else xi.ndim + d for xi, d in zip(xs, dims)]
        return xs, dims, is_list

    qs, dims_q, q_is_list = _as_list(q, dim_q)
    ks, dims_k, k_is_list = _as_list(k, dim_k)
    n = qs[0].shape[dims_q[0]]
    assert n == ks[0].shape[dims_k[0]]
    n_trunks = int(math.ceil(n / n_queries))
    q_pad = n_trunks * n_queries - n
    pad_left = (n_keys - n_queries) // 2
    pad_right = int((n_trunks - 1 / 2) * n_queries + n_keys / 2 - n + 1 / 2)

    q_trunked = [
        reshape_at_dim(pad_at_dim(x, d, (0, q_pad)), d, (n_trunks, n_queries))
        for x, d in zip(qs, dims_q)
    ]
    k_trunked = [
        move_final_dim_to_dim(
            unfold(pad_at_dim(x, d, (pad_left, pad_right)), d, n_keys, n_queries), d + 1
        )
        for x, d in zip(ks, dims_k)
    ]

    mask_trunked = None
    if compute_mask:
        # key j of block b is global position b * n_queries + j - pad_left
        key_pos = mx.arange(n_trunks)[:, None] * n_queries + mx.arange(n_keys)[None, :] - pad_left
        mask_trunked = mx.broadcast_to(
            ((key_pos >= 0) & (key_pos < n))[:, None, :], (n_trunks, n_queries, n_keys)
        )
        query_pos = mx.arange(n_trunks)[:, None] * n_queries + mx.arange(n_queries)[None, :]
        mask_trunked = mask_trunked & (query_pos < n)[:, :, None]

    pad_info = {
        "mask_trunked": mask_trunked,
        "q_pad": q_pad,
        "k_pad_left": pad_left,
        "k_pad_right": pad_right,
    }
    return (
        q_trunked if q_is_list else q_trunked[0],
        k_trunked if k_is_list else k_trunked[0],
        pad_info,
    )


def trunk_attention_bias(
    n: int, n_queries: int, n_keys: int, pad_info: dict[str, Any], inf: float = 1e10
) -> mx.array:
    """Additive local-window mask ``[n_trunks, n_queries, n_keys]`` (``-inf`` outside)."""
    mask = pad_info["mask_trunked"]
    return mx.where(mask, mx.zeros(mask.shape, dtype=mx.float32), mx.array(-inf, dtype=mx.float32))


class Attention(nn.Module):
    """Multi-head attention with gating and an additive pair bias (AF3 style).

    Supports dense attention over all tokens and local windowed attention
    over atoms (``n_queries``/``n_keys``).
    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        num_heads: int,
        gating: bool = True,
        q_linear_bias: bool = True,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.linear_q = Linear(c_q, c_hidden * num_heads, bias=q_linear_bias)
        self.linear_k = LinearNoBias(c_k, c_hidden * num_heads)
        self.linear_v = LinearNoBias(c_v, c_hidden * num_heads)
        self.linear_o = LinearNoBias(
            c_hidden * num_heads, c_q, initializer="zeros" if zero_init else "default"
        )
        if gating:
            self.linear_g = LinearNoBias(c_q, c_hidden * num_heads, initializer="zeros")
        self._fused = FusedWeights()

    def _projections(
        self, q_x: mx.array, kv_x: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, Optional[mx.array]]:
        """Fused q/k/v/gate projections: ``[..., H, N, c_hidden]`` each (gate ``[..., N, H*c]``)."""
        gated = "linear_g" in self
        if q_x is kv_x:
            linears = [self.linear_q, self.linear_k, self.linear_v] + (
                [self.linear_g] if gated else []
            )
            parts = mx.split(q_x @ self._fused(*linears).T, len(linears), axis=-1)
        else:
            k, v = mx.split(kv_x @ self._fused(self.linear_k, self.linear_v).T, 2, axis=-1)
            if gated:
                q, g = mx.split(q_x @ self._fused(self.linear_q, self.linear_g).T, 2, axis=-1)
                parts = [q, k, v, g]
            else:
                parts = [self.linear_q(q_x), k, v]
        q, k, v = parts[:3]
        if "bias" in self.linear_q:
            q = q + self.linear_q.bias

        def split_heads(t: mx.array) -> mx.array:
            t = t.reshape(*t.shape[:-1], self.num_heads, self.c_hidden)
            return t.swapaxes(-2, -3)

        q = split_heads(q) / math.sqrt(self.c_hidden)
        return q, split_heads(k), split_heads(v), parts[3] if gated else None

    def _wrap_up(self, o: mx.array, g: Optional[mx.array]) -> mx.array:
        """``o``: ``[..., Q, H, c_hidden]`` -> gated output projection."""
        if g is not None:
            o = o * mx.sigmoid(g).reshape(*g.shape[:-1], self.num_heads, self.c_hidden)
        return self.linear_o(flatten_final_dims(o, 2))

    def __call__(
        self,
        q_x: mx.array,
        kv_x: mx.array,
        attn_bias: Optional[mx.array] = None,
        trunked_attn_bias: Optional[mx.array] = None,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inf: float = 1e10,
    ) -> mx.array:
        """
        Args:
            q_x: ``[..., Q, c_q]``; kv_x: ``[..., K, c_k]``
            attn_bias: ``[..., H, Q, K]`` or ``[..., Q, K]`` for dense attention.
            trunked_attn_bias: ``[..., H, n_trunks, n_queries, n_keys]`` for local attention.
        """
        q, k, v, g = self._projections(q_x, kv_x)
        if attn_bias is not None and attn_bias.ndim != q.ndim:
            attn_bias = mx.expand_dims(attn_bias, -3)

        if n_queries and n_keys:
            o = local_attention(q, k, v, n_queries, n_keys, trunked_attn_bias, inf=inf)
        else:
            o = attention(q, k, v, attn_bias)
        return self._wrap_up(o.swapaxes(-2, -3), g)


def local_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    n_queries: int,
    n_keys: int,
    trunked_attn_bias: Optional[mx.array] = None,
    inf: float = 1e10,
) -> mx.array:
    """Windowed attention over the second last axis of ``q/k/v`` ``[..., H, N, d]``."""
    n = q.shape[-2]
    q_t, (k_t, v_t), pad_info = rearrange_qk_to_dense_trunk(
        q, [k, v], dim_q=-2, dim_k=[-2, -2], n_queries=n_queries, n_keys=n_keys
    )
    bias = trunk_attention_bias(n, n_queries, n_keys, pad_info, inf)
    if trunked_attn_bias is not None:
        bias = bias + trunked_attn_bias
    # merge (H, n_trunks) so every window is one "head" for the fused kernel
    out = attention(q_t, k_t, v_t, bias)  # [..., H, n_trunks, n_queries, d]
    out = out.reshape(*out.shape[:-3], -1, out.shape[-1])
    return out[..., :n, :]


def gather_pair_embedding_in_dense_trunk(x: mx.array, idx_q: mx.array, idx_k: mx.array) -> mx.array:
    """``y[..., b, i, j, :] = x[..., idx_q[b, i], idx_k[b, j], :]``."""
    return x[..., idx_q[:, :, None], idx_k[:, None, :], :]


def broadcast_token_to_local_atom_pair(
    z_token: mx.array,
    atom_to_token_idx: mx.array,
    n_queries: int,
    n_keys: int,
) -> tuple[mx.array, dict[str, Any]]:
    """Gather token-pair rows/cols for every atom local window."""
    idx_q, idx_k, pad_info = rearrange_qk_to_dense_trunk(
        atom_to_token_idx, atom_to_token_idx, -1, -1, n_queries, n_keys, compute_mask=True
    )
    return gather_pair_embedding_in_dense_trunk(z_token, idx_q, idx_k), pad_info
