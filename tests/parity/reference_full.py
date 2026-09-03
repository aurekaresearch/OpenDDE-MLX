# SPDX-License-Identifier: Apache-2.0
"""End-to-end PyTorch reference with the released checkpoint on a small protein.

    OPENDDE_TORCH_REPO=/path/to/OpenDDE python tests/parity/reference_full.py full /tmp/parity/full.npz

Dumps trunk outputs, structural-token expansion, one denoiser evaluation on a
fixed noisy input, distogram contacts and confidence logits for fixed
coordinates so the MLX model can be checked stage by stage with real weights.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"


def make_job():
    return [
        {
            "name": "parity_small",
            "sequences": [{"proteinChain": {"sequence": SEQUENCE, "count": 1}}],
        }
    ]


def case_full():
    os.environ["LAYERNORM_TYPE"] = "torch"
    from opendde.config.inference import build_inference_config
    from opendde.data.inference.infer_dataloader import InferenceDataset
    from opendde.model.opendde import OpenDDE

    job_path = "/tmp/parity/parity_small.json"
    os.makedirs("/tmp/parity", exist_ok=True)
    with open(job_path, "w") as f:
        json.dump(make_job(), f)
    configs = build_inference_config(fill_required_with_null=True)
    configs.input_json_path = job_path
    configs.use_msa = False
    configs.use_template = False
    configs.use_rna_msa = False
    configs.triangle_multiplicative = "torch"
    configs.triangle_attention = "torch"
    configs.enable_efficient_fusion = True
    configs.enable_diffusion_shared_vars_cache = True
    np.random.seed(0)
    dataset = InferenceDataset(configs=configs, inputs=make_job())
    data, _, err = dataset[0]
    assert not err, err
    feat = data["input_feature_dict"]

    model = OpenDDE(configs)
    ckpt = torch.load(
        os.path.expanduser("~/.cache/opendde/checkpoint/opendde.pt"),
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict({k[len("module.") :]: v for k, v in ckpt["model"].items()}, strict=True)
    model.eval()
    out = {}
    with torch.no_grad():
        feat = model.relative_position_encoding.generate_relp(feat, lazy=True)
        from opendde.model.opendde import update_input_feature_dict

        feat = update_input_feature_dict(feat)
        s_inputs, s, z = model.get_pairformer_output(
            feat, N_cycle=2, inplace_safe=True, chunk_size=None
        )
        out["trunk:s_inputs"], out["trunk:s"], out["trunk:z"] = (
            s_inputs.numpy(),
            s.numpy(),
            z.numpy(),
        )
        struct_feat, ss_inputs, ss, sz = model.expand_to_structural_tokens(
            feat, s_inputs, s, z, inplace_safe=True, chunk_size=None, lazy_relp=True
        )
        out["struct:s_inputs"], out["struct:s"], out["struct:z"] = (
            ss_inputs.numpy(),
            ss.numpy(),
            sz.numpy(),
        )
        out["struct:attn_bias"] = struct_feat["structural_pair_attn_bias"].numpy()
        # diffusion caches + one denoise step on the structural branch
        dm = model.diffusion_module
        pair_z = dm.diffusion_conditioning.prepare_cache(struct_feat["relp"], sz, False)
        p_lm, c_l = dm.atom_attention_encoder.prepare_cache(
            ref_pos=struct_feat["ref_pos"],
            ref_charge=struct_feat["ref_charge"],
            ref_mask=struct_feat["ref_mask"],
            ref_element=struct_feat["ref_element"],
            ref_atom_name_chars=struct_feat["ref_atom_name_chars"],
            atom_to_token_idx=struct_feat["atom_to_token_idx"],
            d_lm=struct_feat["d_lm"],
            v_lm=struct_feat["v_lm"],
            pad_info=struct_feat["pad_info"],
            r_l=True,
            z=pair_z,
            inplace_safe=False,
        )
        out["diff:pair_z"] = pair_z.numpy().copy()
        g = torch.Generator().manual_seed(0)
        x_noisy = torch.randn(2, feat["atom_to_token_idx"].shape[0], 3, generator=g) * 20
        t_hat = torch.tensor([20.0, 2.0])
        out["diff:x_noisy"], out["diff:t_hat"] = x_noisy.numpy(), t_hat.numpy()
        x = dm(
            x_noisy=x_noisy,
            t_hat_noise_level=t_hat,
            input_feature_dict=struct_feat,
            s_inputs=ss_inputs,
            s_trunk=ss,
            z_trunk=None,
            pair_z=pair_z,
            p_lm=p_lm,
            c_l=c_l,
            inplace_safe=True,
            enable_efficient_fusion=True,
        )
        out["diff:x_denoised"] = x.numpy()
        # heads on the residue branch with fixed coordinates
        coords = x_noisy[:1] * 0.1
        out["conf:coords"] = coords.numpy()
        out["head:contact_probs"] = model.compute_distogram_contact_probs(z).numpy()
        plddt, pae, pde, resolved = model.confidence_head(
            input_feature_dict=feat,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_mask=None,
            x_pred_coords=coords,
            triangle_multiplicative="torch",
            triangle_attention="torch",
            inplace_safe=True,
            chunk_size=None,
        )
        out["conf:plddt"], out["conf:pae"], out["conf:pde"], out["conf:resolved"] = (
            plddt.numpy(),
            pae.numpy(),
            pde.numpy(),
            resolved.numpy(),
        )
    return out


if __name__ == "__main__":
    sys.path.insert(0, os.environ["OPENDDE_TORCH_REPO"])
    case, output = sys.argv[1], sys.argv[2]
    arrays = {"full": case_full}[case]()
    np.savez(output, **arrays)
    print(f"wrote {len(arrays)} arrays to {output}")
