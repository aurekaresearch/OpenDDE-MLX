# SPDX-License-Identifier: Apache-2.0
"""PyTorch reference dumps for the diffusion module (see reference.py for usage)."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)
os.environ.setdefault("LAYERNORM_TYPE", "torch")


def _state(module):
    return {f"param:{k}": v.detach().float().numpy() for k, v in module.state_dict().items()}


def _randomize(module, std=0.3):
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn_like(p) * std)


def make_inputs(n_token=12, n_atom=50, n_sample=2):
    from opendde.model.opendde import update_input_feature_dict

    atom_to_token = torch.sort(torch.randint(0, n_token, (n_atom,))).values
    atom_to_token[0], atom_to_token[-1] = 0, n_token - 1
    feat = {
        "atom_to_token_idx": atom_to_token,
        "ref_pos": torch.randn(n_atom, 3),
        "ref_charge": torch.randint(-1, 2, (n_atom,)),
        "ref_mask": torch.ones(n_atom, dtype=torch.long),
        "ref_element": torch.nn.functional.one_hot(torch.randint(0, 128, (n_atom,)), 128).long(),
        "ref_atom_name_chars": torch.nn.functional.one_hot(
            torch.randint(0, 64, (n_atom, 4)), 64
        ).long(),
        "ref_space_uid": atom_to_token.clone(),
        "asym_id": torch.tensor([0] * (n_token // 2) + [1] * (n_token - n_token // 2)),
        "residue_index": torch.arange(n_token),
        "entity_id": torch.tensor([0] * (n_token // 2) + [1] * (n_token - n_token // 2)),
        "token_index": torch.arange(n_token),
        "sym_id": torch.zeros(n_token, dtype=torch.long),
    }
    feat = update_input_feature_dict(feat)
    from opendde.model.modules.embedders import RelativePositionEncoding

    RelativePositionEncoding(r_max=32, s_max=2, c_z=8).generate_relp(feat, lazy=False)
    return feat


def case_diffusion():
    from opendde.model.modules.diffusion import DiffusionModule

    n_token, n_atom, n_sample = 12, 50, 2
    feat = make_inputs(n_token, n_atom, n_sample)
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
    _randomize(module, std=0.2)
    module.eval()
    s_inputs = torch.randn(n_token, 28)
    s_trunk = torch.randn(n_token, 24)
    z_trunk = torch.randn(n_token, n_token, 20)
    x_noisy = torch.randn(n_sample, n_atom, 3) * 5
    t_hat = torch.tensor([3.0, 40.0])
    out = {f"param:{k}": v for k, v in _state(module).items()}
    out = {k.replace("param:param:", "param:"): v for k, v in out.items()}
    for key in (
        "atom_to_token_idx",
        "ref_pos",
        "ref_charge",
        "ref_mask",
        "ref_element",
        "ref_atom_name_chars",
        "ref_space_uid",
        "asym_id",
        "residue_index",
        "entity_id",
        "token_index",
        "sym_id",
    ):
        out[f"feat:{key}"] = feat[key].numpy()
    out["in:s_inputs"], out["in:s_trunk"], out["in:z_trunk"] = (
        s_inputs.numpy(),
        s_trunk.numpy(),
        z_trunk.numpy(),
    )
    out["in:x_noisy"], out["in:t_hat"] = x_noisy.numpy(), t_hat.numpy()
    with torch.no_grad():
        pair_z = module.diffusion_conditioning.prepare_cache(feat["relp"], z_trunk, False)
        out["out:pair_z"] = pair_z.numpy()
        p_lm, c_l = module.atom_attention_encoder.prepare_cache(
            ref_pos=feat["ref_pos"],
            ref_charge=feat["ref_charge"],
            ref_mask=feat["ref_mask"],
            ref_element=feat["ref_element"],
            ref_atom_name_chars=feat["ref_atom_name_chars"],
            atom_to_token_idx=feat["atom_to_token_idx"],
            d_lm=feat["d_lm"],
            v_lm=feat["v_lm"],
            pad_info=feat["pad_info"],
            r_l=True,
            z=pair_z,
        )
        out["out:p_lm"], out["out:c_l"] = p_lm.numpy().copy(), c_l.numpy().copy()
        p_lm_noz, _ = module.atom_attention_encoder.prepare_cache(
            ref_pos=feat["ref_pos"],
            ref_charge=feat["ref_charge"],
            ref_mask=feat["ref_mask"],
            ref_element=feat["ref_element"],
            ref_atom_name_chars=feat["ref_atom_name_chars"],
            atom_to_token_idx=feat["atom_to_token_idx"],
            d_lm=feat["d_lm"],
            v_lm=feat["v_lm"],
            pad_info=feat["pad_info"],
            r_l=None,
            z=None,
        )
        out["out:p_lm_noz"] = p_lm_noz.numpy()
        out["feat:d_lm"] = feat["d_lm"].numpy()
        out["feat:v_lm"] = feat["v_lm"].float().numpy()
        out["feat:mask_trunked"] = feat["pad_info"]["mask_trunked"].float().numpy()
        for fusion in (False, True):
            x = module(
                x_noisy=x_noisy,
                t_hat_noise_level=t_hat,
                input_feature_dict=feat,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                pair_z=pair_z,
                p_lm=p_lm,
                c_l=c_l,
                enable_efficient_fusion=fusion,
                inplace_safe=True,
            )
            out[f"out:x_denoised:fusion{int(fusion)}"] = x.numpy()
        x = module(
            x_noisy=x_noisy,
            t_hat_noise_level=t_hat,
            input_feature_dict=feat,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=None,
            p_lm=None,
            c_l=None,
            inplace_safe=True,
        )
        out["out:x_denoised:nocache"] = x.numpy()
    return out


CASES = {"diffusion": case_diffusion}


if __name__ == "__main__":
    sys.path.insert(0, os.environ["OPENDDE_TORCH_REPO"])
    case, output = sys.argv[1], sys.argv[2]
    arrays = CASES[case]()
    np.savez(output, **arrays)
    print(f"wrote {len(arrays)} arrays to {output}")
