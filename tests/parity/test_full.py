# SPDX-License-Identifier: Apache-2.0
"""Stage-by-stage comparison of the MLX model against PyTorch with released weights."""

import json
import os

import mlx.core as mx
import numpy as np
import pytest

from opendde.config.inference import build_inference_config
from opendde.model.checkpoint import load_checkpoint
from opendde.utils.download import resolve_checkpoint_path
from tests.parity.common import assert_close, load_case
from tests.parity.reference_full import make_job

pytestmark = pytest.mark.parity


@pytest.fixture(scope="module")
def setup():
    from opendde.data.inference.infer_dataloader import InferenceDataset
    from opendde.model.opendde import OpenDDE
    from opendde.model.transformer import update_input_feature_dict
    from runner.inference import to_mlx

    ref = load_case("full")
    configs = build_inference_config(fill_required_with_null=True)
    checkpoint = resolve_checkpoint_path(configs)
    if not os.path.exists(checkpoint):
        pytest.skip("released checkpoint missing")
    job_path = "/tmp/parity/parity_small_mlx.json"
    with open(job_path, "w") as f:
        json.dump(make_job(), f)
    configs.input_json_path = job_path
    configs.use_msa = configs.use_template = configs.use_rna_msa = False
    np.random.seed(0)
    data, _, err = InferenceDataset(configs=configs, inputs=make_job())[0]
    assert not err, err
    model = OpenDDE(configs)
    load_checkpoint(model, checkpoint, strict=True)
    model.set_compute_dtype(mx.float32, True, True)
    feat = update_input_feature_dict(to_mlx(data["input_feature_dict"]))
    return ref, model, feat


def test_trunk_and_structural_expansion(setup):
    ref, model, feat = setup
    s_inputs, s, z = model.get_pairformer_output(feat, N_cycle=2)
    assert_close(s_inputs, ref["trunk:s_inputs"], atol=1e-3, rtol=1e-3)
    assert_close(s, ref["trunk:s"], atol=2e-2, rtol=2e-2)
    assert_close(z, ref["trunk:z"], atol=2e-2, rtol=2e-2)
    s_ref, z_ref = mx.array(ref["trunk:s"]), mx.array(ref["trunk:z"])
    struct_feat, ss_inputs, ss, sz = model.expand_to_structural_tokens(
        feat, mx.array(ref["trunk:s_inputs"]), s_ref, z_ref, None
    )
    assert_close(struct_feat["structural_pair_attn_bias"], ref["struct:attn_bias"])
    assert_close(ss_inputs, ref["struct:s_inputs"], atol=1e-3, rtol=1e-3)
    assert_close(ss, ref["struct:s"], atol=2e-2, rtol=2e-2)
    assert_close(sz, ref["struct:z"], atol=2e-2, rtol=2e-2)


def test_denoiser_and_heads(setup):
    ref, model, feat = setup
    s_inputs, s, z = (mx.array(ref[f"trunk:{k}"]) for k in ("s_inputs", "s", "z"))
    struct_feat, ss_inputs, ss, sz = model.expand_to_structural_tokens(feat, s_inputs, s, z, None)
    ss_inputs, ss, sz = (mx.array(ref[f"struct:{k}"]) for k in ("s_inputs", "s", "z"))
    dm = model.diffusion_module
    pair_z = dm.diffusion_conditioning.prepare_cache(struct_feat, sz)
    assert_close(pair_z, ref["diff:pair_z"], atol=1e-2, rtol=1e-2)
    p_lm, c_l = dm.atom_attention_encoder.prepare_cache(
        struct_feat["ref_pos"],
        struct_feat["ref_charge"],
        struct_feat["ref_mask"],
        struct_feat["ref_element"],
        struct_feat["ref_atom_name_chars"],
        struct_feat["atom_to_token_idx"],
        struct_feat["d_lm"],
        struct_feat["v_lm"],
        struct_feat["pad_info"],
        z=pair_z,
    )
    cache = dm.prepare_pair_bias_cache(pair_z, struct_feat.get("structural_pair_attn_bias"), True)
    x = dm(
        mx.array(ref["diff:x_noisy"]),
        mx.array(ref["diff:t_hat"]),
        struct_feat,
        ss_inputs,
        ss,
        z_trunk=None,
        pair_z=pair_z,
        p_lm=p_lm,
        c_l=c_l,
        enable_efficient_fusion=True,
        pair_bias_cache=cache,
    )
    assert_close(x, ref["diff:x_denoised"], atol=5e-2, rtol=5e-2)

    from opendde.model import sample_confidence

    contact = sample_confidence.compute_contact_prob(
        model.distogram_head(z),
        **sample_confidence.get_bin_params(model.configs.confidence.distogram),
    )
    assert_close(contact, ref["head:contact_probs"], atol=1e-3, rtol=1e-3)
    plddt, pae, pde, resolved = model.confidence_head(
        feat=feat,
        s_inputs=s_inputs,
        s_trunk=s,
        z_trunk=z,
        pair_mask=None,
        x_pred_coords=mx.array(ref["conf:coords"]),
    )
    assert_close(plddt, ref["conf:plddt"], atol=2e-2, rtol=2e-2)
    assert_close(pae, ref["conf:pae"], atol=2e-2, rtol=2e-2)
    assert_close(pde, ref["conf:pde"], atol=2e-2, rtol=2e-2)
    assert_close(resolved, ref["conf:resolved"], atol=2e-2, rtol=2e-2)
