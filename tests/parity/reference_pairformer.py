# SPDX-License-Identifier: Apache-2.0
"""PyTorch reference dumps for the Pairformer / MSA / template / embedder / confidence ports.

    OPENDDE_TORCH_REPO=/path/to/OpenDDE python tests/parity/reference_pairformer.py <case> out.npz

Cases: ``pairformer``, ``embedders``, ``confidence``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

from reference import _randomize, _state

N_TOKEN = 12
N_ATOM = 50


def _dump(out: dict[str, np.ndarray], name: str, module: torch.nn.Module) -> None:
    _randomize(module)
    out.update({f"{name}.{k}": v for k, v in _state(module).items()})


def _token_features() -> dict[str, torch.Tensor]:
    asym_id = torch.tensor([0] * 5 + [1] * 4 + [2] * 3)
    return {
        "asym_id": asym_id,
        "residue_index": torch.tensor([0, 1, 2, 2, 3, 0, 1, 2, 3, 0, 1, 2]),
        "entity_id": torch.tensor([0] * 5 + [1] * 4 + [1] * 3),
        "sym_id": torch.tensor([0] * 5 + [0] * 4 + [1] * 3),
        "token_index": torch.arange(N_TOKEN),
    }


def _atom_features() -> dict[str, torch.Tensor]:
    atom_to_token_idx = torch.sort(torch.randint(0, N_TOKEN, (N_ATOM,)))[0]
    atom_to_token_idx[:N_TOKEN] = torch.arange(N_TOKEN)  # every token owns >= 1 atom
    atom_to_token_idx = torch.sort(atom_to_token_idx)[0]
    counts = torch.bincount(atom_to_token_idx, minlength=N_TOKEN)
    tokatom = torch.cat([torch.arange(int(c)) for c in counts])
    rep_mask = torch.zeros(N_ATOM, dtype=torch.bool)
    rep_mask[torch.cumsum(counts, 0) - 1] = True  # last atom of each token
    structural_mask = torch.zeros(N_ATOM, dtype=torch.bool)
    structural_mask[torch.cumsum(counts, 0) - counts] = True  # first atom of each token
    return {
        "atom_to_token_idx": atom_to_token_idx,
        "atom_to_tokatom_idx": tokatom,
        "ref_space_uid": atom_to_token_idx.clone(),
        "ref_pos": torch.randn(N_ATOM, 3) * 3,
        "ref_charge": torch.randn(N_ATOM),
        "ref_mask": torch.ones(N_ATOM),
        "ref_element": torch.nn.functional.one_hot(torch.randint(0, 128, (N_ATOM,)), 128).float(),
        "ref_atom_name_chars": torch.nn.functional.one_hot(
            torch.randint(0, 64, (N_ATOM, 4)), 64
        ).float(),
        "restype": torch.randn(N_TOKEN, 32),
        "profile": torch.randn(N_TOKEN, 32),
        "deletion_mean": torch.randn(N_TOKEN),
        "distogram_rep_atom_mask": rep_mask,
        "structural_distogram_rep_atom_mask": structural_mask,
    }


def case_pairformer() -> dict[str, np.ndarray]:
    from opendde.model.modules.pairformer import (
        MSABlock,
        MSAModule,
        MSAPairWeightedAveraging,
        MSAStack,
        PairformerBlock,
        PairformerStack,
        TemplateEmbedder,
    )

    out: dict[str, np.ndarray] = {}
    n = 16
    s = torch.randn(n, 32)
    z = torch.randn(n, n, 24)
    pair_mask = torch.ones(n, n)
    out["in:s"], out["in:z"] = s.numpy(), z.numpy()

    block = PairformerBlock(
        n_heads=4, c_z=24, c_s=32, c_hidden_mul=16, c_hidden_pair_att=8, no_heads_pair=2
    )
    _dump(out, "block", block)
    s_out, z_out = block(s, z, pair_mask)
    out["block.out:s"], out["block.out:z"] = s_out.detach().numpy(), z_out.detach().numpy()
    _, z_chunk = block(s, z, pair_mask, chunk_size=8)
    out["block.out:z_chunk"] = z_chunk.detach().numpy()

    block_pair = PairformerBlock(
        c_z=24, c_s=0, c_hidden_mul=16, c_hidden_pair_att=8, no_heads_pair=2
    )
    _dump(out, "block_pair", block_pair)
    out["block_pair.out:z"] = block_pair(None, z, pair_mask)[1].detach().numpy()

    z64 = torch.randn(n, n, 64)
    extra = torch.randn(n, n)
    out["in:z64"], out["in:extra"] = z64.numpy(), extra.numpy()
    stack = PairformerStack(n_blocks=2, n_heads=4, c_z=64, c_s=32, hidden_scale_up=True)
    _dump(out, "stack", stack)
    s_out, z_out = stack(s, z64, pair_mask, extra_attn_bias=extra)
    out["stack.out:s"], out["stack.out:z"] = s_out.detach().numpy(), z_out.detach().numpy()

    m = torch.randn(6, n, 16)
    out["in:m"] = m.numpy()
    mpwa = MSAPairWeightedAveraging(c_m=16, c=8, c_z=24, n_heads=4)
    _dump(out, "mpwa", mpwa)
    out["mpwa.out"] = mpwa(m, z).detach().numpy()

    msa_stack = MSAStack(c_m=16, c_z=24, c=8, msa_chunk_size=4)
    _dump(out, "msa_stack", msa_stack)
    out["msa_stack.out"] = msa_stack(m.clone(), z).detach().numpy()

    msa_block = MSABlock(c_m=16, c_z=24, c_hidden=8, msa_chunk_size=4)
    _dump(out, "msa_block", msa_block)
    m_out, z_out = msa_block(m.clone(), z, pair_mask)
    out["msa_block.out:m"], out["msa_block.out:z"] = m_out.detach().numpy(), z_out.detach().numpy()

    msa_feat = {
        "msa": torch.randint(0, 31, (6, n)),
        "has_deletion": (torch.rand(6, n) > 0.7).float(),
        "deletion_value": torch.rand(6, n),
    }
    s_inputs = torch.randn(n, 20)
    out["in:s_inputs"] = s_inputs.numpy()
    out.update({f"in:{k}": v.numpy() for k, v in msa_feat.items()})
    msa_module = MSAModule(
        n_blocks=2, c_m=16, c_z=24, c_s_inputs=20, msa_chunk_size=4, msa_configs={"msa_depth": 8}
    )
    _dump(out, "msa_module", msa_module)
    out["msa_module.out"] = msa_module(msa_feat, z, s_inputs, pair_mask).detach().numpy()

    asym_id = torch.tensor([0] * 9 + [1] * 7)
    tmpl_feat = {
        "asym_id": asym_id,
        "template_aatype": torch.randint(0, 32, (2, n)),
        "template_distogram": torch.rand(2, n, n, 39),
        "template_pseudo_beta_mask": (torch.rand(2, n, n) > 0.2).float(),
        "template_unit_vector": torch.randn(2, n, n, 3),
        "template_backbone_frame_mask": (torch.rand(2, n, n) > 0.2).float(),
    }
    out.update({f"in:{k}": v.numpy() for k, v in tmpl_feat.items()})
    template = TemplateEmbedder(n_blocks=1, c=16, c_z=24)
    _dump(out, "template", template)
    out["template.out"] = template(tmpl_feat, z, pair_mask).detach().numpy()
    return out


def case_embedders() -> dict[str, np.ndarray]:
    from opendde.model.modules.embedders import (
        FourierEmbedding,
        InputFeatureEmbedder,
        RelativePositionEncoding,
    )
    from opendde.model.modules.head import DistogramHead
    from opendde.model.opendde import update_input_feature_dict

    out: dict[str, np.ndarray] = {}
    feat = _token_features()
    out.update({f"in:{k}": v.numpy() for k, v in feat.items()})
    relpos = RelativePositionEncoding(r_max=4, s_max=2, c_z=24)
    _dump(out, "relpos", relpos)
    relp = relpos.generate_relp(dict(feat))["relp"]
    out["relpos.out:relp"] = relp.numpy()
    out["relpos.out"] = relpos(relp).detach().numpy()

    atom_feat = _atom_features()
    out.update({f"in:{k}": v.numpy() for k, v in atom_feat.items()})
    embedder = InputFeatureEmbedder(c_atom=16, c_atompair=8, c_token=24)
    _dump(out, "embedder", embedder)
    out["embedder.out"] = embedder(update_input_feature_dict(dict(atom_feat))).detach().numpy()

    z = torch.randn(N_TOKEN, N_TOKEN, 24)
    out["in:z"] = z.numpy()
    head = DistogramHead(c_z=24, no_bins=10)
    _dump(out, "distogram", head)
    out["distogram.out"] = head(z).detach().numpy()

    t = torch.rand(3) * 5
    out["in:t"] = t.numpy()
    fourier = FourierEmbedding(c=16)
    _dump(out, "fourier", fourier)
    out["fourier.out"] = fourier(t).detach().numpy()
    return out


def case_confidence() -> dict[str, np.ndarray]:
    from opendde.model.modules.confidence import ConfidenceHead

    out: dict[str, np.ndarray] = {}
    atom_feat = _atom_features()
    out.update({f"in:{k}": v.numpy() for k, v in atom_feat.items()})
    extra = torch.randn(N_TOKEN, N_TOKEN)
    out["in:extra"] = extra.numpy()
    s_inputs = torch.randn(N_TOKEN, 20)
    s_trunk = torch.randn(N_TOKEN, 32) * 300
    z_trunk = torch.randn(N_TOKEN, N_TOKEN, 24)
    x = torch.randn(2, N_ATOM, 3) * 8
    pair_mask = torch.ones(N_TOKEN, N_TOKEN)
    out["in:s_inputs"], out["in:s_trunk"] = s_inputs.numpy(), s_trunk.numpy()
    out["in:z_trunk"], out["in:x"] = z_trunk.numpy(), x.numpy()

    head = ConfidenceHead(
        n_blocks=1,
        c_s=32,
        c_z=24,
        c_s_inputs=20,
        b_pae=8,
        b_pde=8,
        b_plddt=10,
        b_resolved=2,
        max_atoms_per_token=int(atom_feat["atom_to_tokatom_idx"].max()) + 1,
    )
    _dump(out, "head", head)
    with torch.no_grad():  # keep real distance bins after randomisation
        head.lower_bins.copy_(torch.arange(3.25, 52.0, 1.25))
        head.upper_bins.copy_(torch.cat([head.lower_bins[1:], torch.tensor([1e6])]))
    out.update({f"head.{k}": v for k, v in _state(head).items() if k.startswith("param:")})
    feat = {**atom_feat, "structural_pair_attn_bias": extra}
    for tag in ("structural", "fallback"):
        if tag == "fallback":
            feat.pop("structural_distogram_rep_atom_mask")
        preds = head(feat, s_inputs, s_trunk, z_trunk, pair_mask, x)
        for name, pred in zip(("plddt", "pae", "pde", "resolved"), preds):
            out[f"head.out:{tag}:{name}"] = pred.detach().numpy()
    return out


CASES = {
    "pairformer": case_pairformer,
    "embedders": case_embedders,
    "confidence_head": case_confidence,
}


def main() -> None:
    repo = os.environ.get("OPENDDE_TORCH_REPO")
    if repo:
        sys.path.insert(0, repo)
    case, output = sys.argv[1], sys.argv[2]
    arrays = CASES[case]()
    np.savez(output, **arrays)
    print(f"wrote {len(arrays)} arrays to {output}")


if __name__ == "__main__":
    main()
