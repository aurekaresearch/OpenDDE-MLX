# SPDX-License-Identifier: Apache-2.0
import mlx.core as mx
import numpy as np
import pytest

from opendde.model.confidence import ConfidenceHead
from opendde.model.embedders import (
    FourierEmbedding,
    InputFeatureEmbedder,
    RelativePositionEncoding,
    relative_position_features,
)
from opendde.model.head import DistogramHead
from opendde.model.pairformer import (
    MSABlock,
    MSAModule,
    MSAPairWeightedAveraging,
    MSAStack,
    PairformerBlock,
    PairformerStack,
    TemplateEmbedder,
)
from opendde.model.transformer import update_input_feature_dict
from tests.parity.common import assert_close, load_case, load_module

pytestmark = pytest.mark.parity

SMALL_BLOCK = dict(c_z=24, c_hidden_mul=16, c_hidden_pair_att=8, no_heads_pair=2)


def _inputs(ref: dict[str, np.ndarray]) -> dict[str, mx.array]:
    return {k[len("in:") :]: mx.array(v) for k, v in ref.items() if k.startswith("in:")}


def test_pairformer_block_and_stack():
    ref = load_case("pairformer")
    x = _inputs(ref)
    pair_mask = mx.ones(x["z"].shape[:-1])
    block = load_module(PairformerBlock(n_heads=4, c_s=32, **SMALL_BLOCK), ref, "block")
    s, z = block(x["s"], x["z"], pair_mask)
    assert_close(s, ref["block.out:s"])
    assert_close(z, ref["block.out:z"])
    _, z = block(x["s"], x["z"], pair_mask, chunk_size=8)
    assert_close(z, ref["block.out:z_chunk"])

    block_pair = load_module(PairformerBlock(c_s=0, **SMALL_BLOCK), ref, "block_pair")
    s, z = block_pair(None, x["z"], pair_mask)
    assert s is None
    assert_close(z, ref["block_pair.out:z"])

    stack = load_module(
        PairformerStack(n_blocks=2, n_heads=4, c_z=64, c_s=32, hidden_scale_up=True), ref, "stack"
    )
    s, z = stack(x["s"], x["z64"], pair_mask, extra_attn_bias=x["extra"])
    assert_close(s, ref["stack.out:s"], atol=1e-3, rtol=1e-3)
    assert_close(z, ref["stack.out:z"], atol=1e-3, rtol=1e-3)


def test_msa_modules():
    ref = load_case("pairformer")
    x = _inputs(ref)
    pair_mask = mx.ones(x["z"].shape[:-1])
    mpwa = load_module(MSAPairWeightedAveraging(c_m=16, c=8, c_z=24, n_heads=4), ref, "mpwa")
    assert_close(mpwa(x["m"], x["z"]), ref["mpwa.out"])

    msa_stack = load_module(MSAStack(c_m=16, c_z=24, c=8, msa_chunk_size=4), ref, "msa_stack")
    assert_close(msa_stack(x["m"], x["z"]), ref["msa_stack.out"])

    msa_block = load_module(
        MSABlock(c_m=16, c_z=24, c_hidden=8, msa_chunk_size=4), ref, "msa_block"
    )
    m, z = msa_block(x["m"], x["z"], pair_mask)
    assert_close(m, ref["msa_block.out:m"])
    assert_close(z, ref["msa_block.out:z"])

    msa_module = load_module(
        MSAModule(
            n_blocks=2,
            c_m=16,
            c_z=24,
            c_s_inputs=20,
            msa_chunk_size=4,
            msa_configs={"msa_depth": 8},
        ),
        ref,
        "msa_module",
    )
    feat = {k: x[k] for k in ("msa", "has_deletion", "deletion_value")}
    z = msa_module(feat, x["z"], x["s_inputs"], pair_mask)
    assert_close(z, ref["msa_module.out"], atol=1e-3, rtol=1e-3)
    assert msa_module({}, x["z"], x["s_inputs"], pair_mask) is x["z"]


def test_template_embedder():
    ref = load_case("pairformer")
    x = _inputs(ref)
    template = load_module(TemplateEmbedder(n_blocks=1, c=16, c_z=24), ref, "template")
    feat = {k: v for k, v in x.items() if k.startswith("template_") or k == "asym_id"}
    assert_close(template(feat, x["z"]), ref["template.out"])
    assert template({"asym_id": x["asym_id"]}, x["z"]) is None


def test_embedders_and_heads():
    ref = load_case("embedders")
    x = _inputs(ref)
    relpos = load_module(RelativePositionEncoding(r_max=4, s_max=2, c_z=24), ref, "relpos")
    relp = relative_position_features(x, r_max=4, s_max=2)
    assert_close(relp, ref["relpos.out:relp"])
    assert_close(relpos(relp), ref["relpos.out"])
    assert_close(relpos(x), ref["relpos.out"])

    embedder = load_module(
        InputFeatureEmbedder(c_atom=16, c_atompair=8, c_token=24), ref, "embedder"
    )
    assert_close(embedder(update_input_feature_dict(dict(x))), ref["embedder.out"])

    head = load_module(DistogramHead(c_z=24, no_bins=10), ref, "distogram")
    assert_close(head(x["z"]), ref["distogram.out"])

    fourier = load_module(FourierEmbedding(c=16), ref, "fourier")
    assert_close(fourier(x["t"]), ref["fourier.out"])


def test_confidence_head():
    ref = load_case("confidence_head")
    x = _inputs(ref)
    head = load_module(
        ConfidenceHead(
            n_blocks=1,
            c_s=32,
            c_z=24,
            c_s_inputs=20,
            b_pae=8,
            b_pde=8,
            b_plddt=10,
            b_resolved=2,
            max_atoms_per_token=ref["head.param:plddt_weight"].shape[0],
        ),
        ref,
        "head",
    )
    feat = {**x, "structural_pair_attn_bias": x["extra"]}
    pair_mask = mx.ones(x["z_trunk"].shape[:-1])
    for tag in ("structural", "fallback"):
        if tag == "fallback":
            feat.pop("structural_distogram_rep_atom_mask")
        preds = head(feat, x["s_inputs"], x["s_trunk"], x["z_trunk"], pair_mask, x["x"])
        for name, pred in zip(("plddt", "pae", "pde", "resolved"), preds):
            assert_close(pred, ref[f"head.out:{tag}:{name}"])
