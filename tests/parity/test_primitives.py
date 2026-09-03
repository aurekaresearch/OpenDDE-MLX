# SPDX-License-Identifier: Apache-2.0
import mlx.core as mx
import pytest

from opendde.model.primitives import AdaptiveLayerNorm, Attention, Transition
from opendde.model.triangular import (
    OuterProductMean,
    TriangleAttention,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from tests.parity.common import assert_close, load_case, load_module

pytestmark = pytest.mark.parity


def test_attention_dense_and_local():
    ref = load_case("primitives")
    attn = load_module(Attention(32, 32, 32, 8, 4), ref, "attn")
    a = mx.array(ref["attn.in:a"])
    assert_close(attn(a, a, attn_bias=mx.array(ref["attn.in:bias"])), ref["attn.out:dense"])
    out = attn(
        a, a, trunked_attn_bias=mx.array(ref["attn.in:trunk_bias"]), n_queries=32, n_keys=128
    )
    assert_close(out, ref["attn.out:local"])


def test_transition_and_adaln():
    ref = load_case("primitives")
    a = mx.array(ref["attn.in:a"])
    s = mx.array(ref["adaln.in:s"])
    assert_close(load_module(Transition(32, 2), ref, "trans")(a), ref["trans.out"])
    assert_close(load_module(AdaptiveLayerNorm(32, 16), ref, "adaln")(a, s), ref["adaln.out"])


def test_conditioned_transition_and_attention_pair_bias():
    from opendde.model.transformer import AttentionPairBias, ConditionedTransitionBlock

    ref = load_case("primitives")
    a = mx.array(ref["attn.in:a"])
    s = mx.array(ref["adaln.in:s"])
    z = mx.array(ref["apb.in:z"])
    assert_close(
        load_module(ConditionedTransitionBlock(32, 16, n=2), ref, "ctb")(a, s), ref["ctb.out"]
    )
    apb = load_module(AttentionPairBias(has_s=True, n_heads=4, c_a=32, c_s=16, c_z=24), ref, "apb")
    assert_close(apb(a=a, s=s, z=z), ref["apb.out"])
    z_norm = mx.fast.layer_norm(z, None, None, 1e-5)
    assert_close(apb(a=a, s=s, z=z_norm, z_is_normalized=True), ref["apb.out:fused"])
    apb_nos = load_module(
        AttentionPairBias(has_s=False, create_offset_ln_z=True, n_heads=4, c_a=32, c_z=24),
        ref,
        "apb_nos",
    )
    assert_close(apb_nos(a=a, s=None, z=z), ref["apb_nos.out"])


def test_triangle_updates():
    ref = load_case("triangular")
    z = mx.array(ref["in:z"])
    tmo = load_module(TriangleMultiplicationOutgoing(32, 32), ref, "tmo")
    tmi = load_module(TriangleMultiplicationIncoming(32, 32), ref, "tmi")
    assert_close(tmo(z), ref["tmo.out"], atol=1e-3, rtol=1e-3)
    assert_close(tmi(z), ref["tmi.out"], atol=1e-3, rtol=1e-3)
    assert_close(tmo(z, chunk_size=16), ref["tmo.out"], atol=1e-3, rtol=1e-3)
    assert_close(tmi(z, chunk_size=16), ref["tmi.out"], atol=1e-3, rtol=1e-3)
    tas = load_module(TriangleAttention(32, 8, 4, starting=True), ref, "tas")
    tae = load_module(TriangleAttention(32, 8, 4, starting=False), ref, "tae")
    assert_close(tas(z), ref["tas.out"])
    assert_close(tae(z), ref["tae.out"])
    assert_close(tas(z, chunk_size=16), ref["tas.out:chunk"])
    assert_close(tae(z, chunk_size=16), ref["tae.out:chunk"])
    opm = load_module(OuterProductMean(16, 32, 8), ref, "opm")
    assert_close(opm(mx.array(ref["opm.in:m"])), ref["opm.out"], atol=1e-3, rtol=1e-3)
    assert_close(opm(mx.array(ref["opm.in:m"]), chunk_size=8), ref["opm.out"], atol=1e-3, rtol=1e-3)
