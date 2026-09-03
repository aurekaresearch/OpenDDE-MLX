# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
# Copyright 2021 AlQuraishi Laboratory
"""Pair-representation updates: triangle multiplication, triangle attention
and the outer product mean (AF3 Algorithms 10, 12, 13, 14).

Large pair tensors are processed in column/row chunks that are evaluated one
at a time so the peak footprint stays bounded on unified memory.
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from opendde.model.primitives import (
    FP16_BIAS_FLOOR,
    FusedWeights,
    LayerNorm,
    Linear,
    attention,
    schedule,
)
from opendde.model.utils import flatten_final_dims, permute_final_dims

# Pair rows above this size trigger chunked triangle updates.
TRIANGLE_CHUNK_THRESHOLD = 256
TRIANGLE_CHUNK_SIZE = 128
# Working-set budgets for one triangle-attention row chunk / multiplication column chunk.
ATTENTION_CHUNK_BYTES = 2 << 30
TRIANGLE_CHUNK_BYTES = 1 << 30


class Attention(nn.Module):
    """Multi-head attention with an explicit list of additive biases (OpenFold)."""

    def __init__(
        self, c_q: int, c_k: int, c_v: int, c_hidden: int, no_heads: int, gating: bool = True
    ) -> None:
        super().__init__()
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.linear_q = Linear(c_q, c_hidden * no_heads, bias=False)
        self.linear_k = Linear(c_k, c_hidden * no_heads, bias=False)
        self.linear_v = Linear(c_v, c_hidden * no_heads, bias=False)
        self.linear_o = Linear(c_hidden * no_heads, c_q, bias=False, initializer="zeros")
        if gating:
            self.linear_g = Linear(c_q, c_hidden * no_heads, bias=False, initializer="zeros")
        self._fused = FusedWeights()

    def __call__(self, x: mx.array, biases: list[mx.array]) -> mx.array:
        """Self-attention over ``x [*, N, c]``; biases broadcast to ``[*, H, N, N]``."""

        def split_heads(t: mx.array) -> mx.array:
            return t.reshape(*t.shape[:-1], self.no_heads, self.c_hidden).swapaxes(-2, -3)

        linears = [self.linear_q, self.linear_k, self.linear_v]
        if "linear_g" in self:
            linears.append(self.linear_g)
        parts = mx.split(x @ self._fused(*linears).T, len(linears), axis=-1)
        q = split_heads(parts[0]) / math.sqrt(self.c_hidden)
        bias = None
        for b in biases:
            bias = b if bias is None else bias + b
        o = attention(q, split_heads(parts[1]), split_heads(parts[2]), bias).swapaxes(-2, -3)
        if len(parts) == 4:
            o = o * mx.sigmoid(parts[3]).reshape(*o.shape[:-2], self.no_heads, self.c_hidden)
        return self.linear_o(flatten_final_dims(o, 2))


class OuterProductMean(nn.Module):
    """Algorithm 9: MSA -> pair outer product mean."""

    def __init__(self, c_m: int, c_z: int, c_hidden: int, eps: float = 1e-3) -> None:
        super().__init__()
        self.c_hidden = c_hidden
        self.eps = eps
        self.layer_norm = LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden, bias=False)
        self.linear_2 = Linear(c_m, c_hidden, bias=False)
        self.linear_out = Linear(c_hidden**2, c_z, initializer="zeros")

    def __call__(
        self, m: mx.array, mask: Optional[mx.array] = None, chunk_size: Optional[int] = None
    ) -> mx.array:
        """``m [*, N_seq, N_res, c_m]`` -> ``[*, N_res, N_res, c_z]``."""
        if mask is None:
            mask = mx.ones(m.shape[:-1], dtype=m.dtype)
        ln = self.layer_norm(m)
        mask = mask[..., None]
        a = (self.linear_1(ln) * mask).swapaxes(-2, -3)  # [*, N_res, N_seq, c]
        b = (self.linear_2(ln) * mask).swapaxes(-2, -3)
        n_res = a.shape[-3]
        chunk = chunk_size or (TRIANGLE_CHUNK_SIZE if n_res > TRIANGLE_CHUNK_THRESHOLD else n_res)
        outs = []
        for start in range(0, n_res, chunk):
            outer = mx.einsum("...bac,...dae->...bdce", a[..., start : start + chunk, :, :], b)
            outer = outer.reshape(*outer.shape[:-2], -1)
            outs.append(self.linear_out(outer))
            schedule(outs)
        outer = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=-3)
        norm = mx.einsum("...abc,...adc->...bdc", mask, mask) + self.eps
        return outer / norm


def _peak(x: mx.array) -> mx.array:
    """Largest magnitude in ``x``, floored so the division stays finite."""
    return mx.maximum(mx.max(mx.abs(x)), mx.array(1e-4, x.dtype))


class TriangleMultiplicativeUpdate(nn.Module):
    """Algorithms 12/13: triangle multiplication (outgoing or incoming edges)."""

    def __init__(self, c_z: int, c_hidden: int, _outgoing: bool = True) -> None:
        super().__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing
        self.linear_g = Linear(c_z, c_z, bias=False, initializer="zeros")
        self.linear_z = Linear(c_hidden, c_z, bias=False, initializer="zeros")
        self.layer_norm_in = LayerNorm(c_z)
        self.layer_norm_out = LayerNorm(c_hidden)
        self.linear_a_p = Linear(c_z, c_hidden, bias=False)
        self.linear_a_g = Linear(c_z, c_hidden, bias=False, initializer="zeros")
        self.linear_b_p = Linear(c_z, c_hidden, bias=False)
        self.linear_b_g = Linear(c_z, c_hidden, bias=False, initializer="zeros")
        self._fused_a = FusedWeights()
        self._fused_b = FusedWeights()

    def _project(self, z_ln: mx.array, mask: mx.array, a: bool) -> mx.array:
        fused, linear_g, linear_p = (
            (self._fused_a, self.linear_a_g, self.linear_a_p)
            if a
            else (self._fused_b, self.linear_b_g, self.linear_b_p)
        )
        gate, proj = mx.split(z_ln @ fused(linear_g, linear_p).T, 2, axis=-1)
        return mask * mx.sigmoid(gate) * proj

    def __call__(
        self, z: mx.array, mask: Optional[mx.array] = None, chunk_size: Optional[int] = None
    ) -> mx.array:
        """``z [*, N, N, c_z]`` -> update ``[*, N, N, c_z]`` (residual not included)."""
        if mask is None:
            mask = mx.ones(z.shape[:-1], dtype=z.dtype)
        mask = mask[..., None]
        z_ln = self.layer_norm_in(z)
        n = z.shape[-2]
        # columns per chunk from a byte budget (~6 live copies of a [N, cols, c] slab)
        per_col = n * self.c_hidden * z.dtype.size * 6
        chunk = min(chunk_size or n, max(16, TRIANGLE_CHUNK_BYTES // per_col))

        a = self._project(z_ln, mask, a=True)
        # outgoing: x[i, j] = sum_k a[i, k] b[j, k]; incoming: x[i, j] = sum_k a[k, i] b[k, j]
        # materialise the batched-matmul layout once instead of per column chunk
        a = mx.contiguous(permute_final_dims(a, (2, 0, 1) if self._outgoing else (2, 1, 0)))
        # The sum over k reaches ~1e5 and overflows fp16. layer_norm_out below is invariant
        # to a positive scale, so normalising both factors is exact and keeps the matmul in
        # the input dtype.
        half = z.dtype == mx.float16
        if half:
            a = a / _peak(a)
        outs = []
        for start in range(0, n, chunk):
            cols = slice(start, start + chunk)
            if self._outgoing:
                b = self._project(z_ln[..., cols, :, :], mask[..., cols, :, :], a=False)
                b = permute_final_dims(b, (2, 1, 0))  # [*, c, k, j]
            else:
                b = self._project(z_ln[..., :, cols, :], mask[..., :, cols, :], a=False)
                b = permute_final_dims(b, (2, 0, 1))  # [*, c, k, j]
            if half:
                b = b / _peak(b)
            x = permute_final_dims(a @ b, (1, 2, 0))  # [*, i, j, c]
            x = self.linear_z(self.layer_norm_out(x))
            outs.append(x * mx.sigmoid(self.linear_g(z_ln[..., :, cols, :])))
            schedule(outs)
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=-2)


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    def __init__(self, c_z: int, c_hidden: int) -> None:
        super().__init__(c_z, c_hidden, _outgoing=True)


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    def __init__(self, c_z: int, c_hidden: int) -> None:
        super().__init__(c_z, c_hidden, _outgoing=False)


class TriangleAttention(nn.Module):
    """Algorithm 14/15: attention over pair rows (``starting``) or columns."""

    def __init__(
        self, c_in: int, c_hidden: int, no_heads: int, starting: bool = True, inf: float = 1e9
    ) -> None:
        super().__init__()
        self.starting = starting
        self.inf = inf
        self.layer_norm = LayerNorm(c_in)
        self.linear = Linear(c_in, no_heads, bias=False)
        self.mha = Attention(c_in, c_in, c_in, c_hidden, no_heads)

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None, chunk_size: Optional[int] = None
    ) -> mx.array:
        """``x [*, I, J, c_in]`` -> update ``[*, I, J, c_in]``."""
        if not self.starting:
            x = x.swapaxes(-2, -3)
            mask = None if mask is None else mask.swapaxes(-1, -2)
        x = self.layer_norm(x)
        # [*, 1, H, J, J]: shared by every row so the fused kernel broadcasts it
        biases = [permute_final_dims(self.linear(x), (2, 0, 1))[..., None, :, :, :]]
        if mask is not None:
            # 1e9 overflows fp16, and inf * 0 on an unmasked entry would be NaN
            inf = -FP16_BIAS_FLOOR if x.dtype == mx.float16 else self.inf
            biases.append((inf * (mask - 1))[..., :, None, None, :])  # [*, I, 1, 1, J]
        n = x.shape[-3]
        # rows per chunk from a byte budget: q/k/v/out live padded to 64 dims, ~8 copies
        per_row = n * self.mha.no_heads * 64 * x.dtype.size * 8
        chunk = min(chunk_size or n, max(16, ATTENTION_CHUNK_BYTES // per_row))
        outs = []
        for start in range(0, n, chunk):
            rows = slice(start, start + chunk)
            x_rows = x[..., rows, :, :]
            row_biases = [biases[0]] + [b[..., rows, :, :, :] for b in biases[1:]]
            outs.append(self.mha(x_rows, row_biases))
            schedule(outs)
        out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=-3)
        return out.swapaxes(-2, -3) if not self.starting else out
