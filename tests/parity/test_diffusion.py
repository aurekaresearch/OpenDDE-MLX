# SPDX-License-Identifier: Apache-2.0
import mlx.core as mx
import pytest

from opendde.model.diffusion import DiffusionModule
from opendde.model.transformer import update_input_feature_dict
from tests.parity.common import assert_close, load_case, load_module

pytestmark = pytest.mark.parity


def _build():
    ref = load_case("diffusion")
    module = DiffusionModule(
        sigma_data=16.0,
        c_atom=16,
        c_atompair=8,
        c_token=32,
        c_s=24,
        c_z=20,
        c_z_pair_diffusion=12,
        c_s_inputs=28,
        atom_encoder={"n_blocks": 2, "n_heads": 2},
        transformer={"n_blocks": 2, "n_heads": 4},
        atom_decoder={"n_blocks": 2, "n_heads": 2},
    )
    weights = {k[len("param:") :]: v for k, v in ref.items() if k.startswith("param:")}
    ref = {
        **{k: v for k, v in ref.items() if not k.startswith("param:")},
        **{f"m.param:{k}": v for k, v in weights.items()},
    }
    load_module(module, ref, "m")
    feat = {k[len("feat:") :]: mx.array(v) for k, v in ref.items() if k.startswith("feat:")}
    feat = update_input_feature_dict(feat)
    return ref, module, feat


def test_diffusion_caches_and_denoise():
    ref, module, feat = _build()
    s_inputs, s_trunk, z_trunk = (
        mx.array(ref[f"in:{k}"]) for k in ("s_inputs", "s_trunk", "z_trunk")
    )
    pair_z = module.diffusion_conditioning.prepare_cache(feat, z_trunk)
    assert_close(pair_z, ref["out:pair_z"], atol=1e-3, rtol=1e-3)
    p_lm, c_l = module.atom_attention_encoder.prepare_cache(
        feat["ref_pos"],
        feat["ref_charge"],
        feat["ref_mask"],
        feat["ref_element"],
        feat["ref_atom_name_chars"],
        feat["atom_to_token_idx"],
        feat["d_lm"],
        feat["v_lm"],
        feat["pad_info"],
        z=pair_z,
    )
    assert_close(p_lm, ref["out:p_lm"], atol=1e-3, rtol=1e-3)
    assert_close(c_l, ref["out:c_l"], atol=1e-3, rtol=1e-3)
    x_noisy, t_hat = mx.array(ref["in:x_noisy"]), mx.array(ref["in:t_hat"])
    common = dict(feat=feat, s_inputs=s_inputs, s_trunk=s_trunk)
    for fusion in (False, True):
        x = module(
            x_noisy,
            t_hat,
            z_trunk=None,
            pair_z=pair_z,
            p_lm=p_lm,
            c_l=c_l,
            enable_efficient_fusion=fusion,
            **common,
        )
        assert_close(x, ref[f"out:x_denoised:fusion{int(fusion)}"], atol=2e-3, rtol=2e-3)
    cache = module.prepare_pair_bias_cache(pair_z, None, True)
    x = module(
        x_noisy,
        t_hat,
        z_trunk=None,
        pair_z=pair_z,
        p_lm=p_lm,
        c_l=c_l,
        enable_efficient_fusion=True,
        pair_bias_cache=cache,
        **common,
    )
    assert_close(x, ref["out:x_denoised:fusion1"], atol=2e-3, rtol=2e-3)
    x = module(x_noisy, t_hat, z_trunk=z_trunk, pair_z=None, p_lm=None, c_l=None, **common)
    assert_close(x, ref["out:x_denoised:nocache"], atol=2e-3, rtol=2e-3)
