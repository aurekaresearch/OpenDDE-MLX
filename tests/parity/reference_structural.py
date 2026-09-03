# SPDX-License-Identifier: Apache-2.0
"""PyTorch reference outputs for structural tokens, shape complementarity and confidence.

Run with the *original* OpenDDE repository (PyTorch CPU):

    OPENDDE_TORCH_REPO=/path/to/OpenDDE python tests/parity/reference_structural.py <case> out.npz

Cases: ``structural`` (StructuralTokenExpander), ``shape_comp``
(compute_shape_complementarity_fields), ``confidence`` (compute_full_data_and_summary)
and ``clash`` (metrics.clash.Clash).
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)
os.environ.setdefault("LAYERNORM_TYPE", "torch")


def _state(module: torch.nn.Module) -> dict[str, np.ndarray]:
    return {f"param:{k}": v.detach().float().numpy() for k, v in module.state_dict().items()}


def _randomize(module: torch.nn.Module, std: float = 0.3) -> None:
    """Replace zero-initialised weights so every path contributes."""
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn_like(p) * std)


def structural_token_inputs() -> dict[str, np.ndarray]:
    """10 protein + 10 DNA residues (bb/sc, bb/base tokens) + 3 single-token ligand residues."""
    parent, role, prev_parent, next_parent, polymer_type = [], [], [], [], []
    residue_asym, residue_index = [], []
    chains = [(0, 10, (1, 2), 1), (1, 10, (3, 4), 2), (2, 3, (0,), 0)]
    res = 0
    for asym, n_res, roles, ptype in chains:
        for i in range(n_res):
            residue_asym.append(asym)
            residue_index.append(i)
            for r in roles:
                parent.append(res)
                role.append(r)
                is_bb = r in (1, 3, 5)
                prev_parent.append(res - 1 if is_bb and i > 0 else -1)
                next_parent.append(res + 1 if is_bb and i < n_res - 1 else -1)
                polymer_type.append(ptype)
            res += 1
    return {
        "parent_residue_idx": np.array(parent, dtype=np.int64),
        "subtoken_role_id": np.array(role, dtype=np.int64),
        "prev_parent_residue_idx": np.array(prev_parent, dtype=np.int64),
        "next_parent_residue_idx": np.array(next_parent, dtype=np.int64),
        "structural_polymer_type": np.array(polymer_type, dtype=np.int64),
        "asym_id": np.array(residue_asym, dtype=np.int64),
        "residue_index": np.array(residue_index, dtype=np.int64),
    }


def case_structural() -> dict[str, np.ndarray]:
    from opendde.model.modules.structural_tokens import StructuralTokenExpander

    out: dict[str, np.ndarray] = {}
    feat_np = structural_token_inputs()
    feat = {k: torch.from_numpy(v) for k, v in feat_np.items()}
    out.update({f"in:{k}": v for k, v in feat_np.items()})
    n_res = feat_np["asym_id"].shape[0]
    s_inputs = torch.randn(n_res, 24)
    s = torch.randn(n_res, 32)
    z = torch.randn(n_res, n_res, 16)
    out["in:s_inputs_res"], out["in:s_res"], out["in:z_res"] = (
        s_inputs.numpy(),
        s.numpy(),
        z.numpy(),
    )
    for name, mode in (("full", "full"), ("none", "none")):
        mod = StructuralTokenExpander(c_s=32, c_z=16, c_s_inputs=24, pair_projection_mode=mode)
        _randomize(mod)
        # mirror the released checkpoint prefix so ``remap_key`` renumbers the MLP indices
        out.update(
            {f"{name}.param:structural_token_expander.{k[6:]}": v for k, v in _state(mod).items()}
        )
        with torch.no_grad():
            s_inputs_struct, s_struct, z_struct, pair_features = mod(feat, s_inputs, s, z)
        out[f"{name}.out:s_inputs_struct"] = s_inputs_struct.numpy()
        out[f"{name}.out:s_struct"] = s_struct.numpy()
        out[f"{name}.out:z_struct"] = z_struct.numpy()
        for key, value in pair_features.items():
            out[f"{name}.out:{key}"] = value.numpy()
    return out


def residue_shape_comp_inputs() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Two 12-residue protein chains, 4 atoms per residue, overlapping boxes."""
    n_res_per_chain, atoms_per_res = 12, 4
    n_token = 2 * n_res_per_chain
    n_atom = n_token * atoms_per_res
    rng = np.random.default_rng(1)
    coord = rng.uniform(0.0, 14.0, size=(2, n_atom, 3)).astype(np.float32)
    coord[:, n_atom // 2 :] += 8.0
    atom_mask = np.ones(n_atom, dtype=bool)
    atom_mask[[3, 10, 20, 57]] = False  # includes rep atom 20 of token 5
    feat = {
        "token_index": np.arange(n_token, dtype=np.int64),
        "atom_to_token_idx": np.repeat(np.arange(n_token), atoms_per_res).astype(np.int64),
        "distogram_rep_atom_mask": (np.arange(n_atom) % atoms_per_res == 0),
        "asym_id": np.repeat(np.array([0, 1]), n_res_per_chain).astype(np.int64),
        "is_protein": np.ones(n_atom, dtype=np.float32),
        "is_ligand": np.zeros(n_atom, dtype=np.float32),
        "is_dna": np.zeros(n_atom, dtype=np.float32),
        "is_rna": np.zeros(n_atom, dtype=np.float32),
        "residue_index": np.tile(np.arange(n_res_per_chain), 2).astype(np.int64),
        # structural expansion: atoms (0, 1) -> backbone token, (2, 3) -> sidechain token
        "parent_residue_idx": np.repeat(np.arange(n_token), 2).astype(np.int64),
        "structural_token_index": np.arange(2 * n_token, dtype=np.int64),
        "atom_to_structural_token_idx": (np.arange(n_atom) // 2).astype(np.int64),
        "structural_distogram_rep_atom_mask": (np.arange(n_atom) % 2 == 0),
        "subtoken_role_id": np.tile(np.array([1, 2]), n_token).astype(np.int64),
    }
    return coord, atom_mask, feat


def case_shape_comp() -> dict[str, np.ndarray]:
    from opendde.model.shape_complementarity import (
        compute_shape_complementarity_fields,
        structural_shape_comp_feature_dict,
        summarize_shape_comp_pair,
    )

    out: dict[str, np.ndarray] = {}
    coord_np, atom_mask_np, feat_np = residue_shape_comp_inputs()
    out["in:coordinate"], out["in:atom_mask"] = coord_np, atom_mask_np
    out.update({f"in:{k}": v for k, v in feat_np.items()})
    feat = {k: torch.from_numpy(v) for k, v in feat_np.items()}
    coord, atom_mask = torch.from_numpy(coord_np), torch.from_numpy(atom_mask_np)
    for name, fd in (("residue", feat), ("structural", structural_shape_comp_feature_dict(feat))):
        with torch.no_grad():
            fields = compute_shape_complementarity_fields(coord, fd, atom_mask=atom_mask)
            assert fields["shape_comp_pair_mask"].sum() > 32, "interface too small"
            out.update({f"{name}.out:{k}": v.numpy() for k, v in fields.items()})
            pair_mean, pair_topk_mean, valid_frac = summarize_shape_comp_pair(
                fields["shape_comp_pair"], fields["shape_comp_pair_mask"]
            )
        out[f"{name}.out:summary_pair_mean"] = pair_mean.numpy()
        out[f"{name}.out:summary_pair_topk_mean"] = pair_topk_mean.numpy()
        out[f"{name}.out:summary_valid_pair_frac"] = valid_frac.numpy()
    return out


def confidence_configs() -> SimpleNamespace:
    bins = lambda lo, hi, n: SimpleNamespace(min_bin=lo, max_bin=hi, no_bins=n)  # noqa: E731
    return SimpleNamespace(
        confidence=SimpleNamespace(
            plddt=bins(0.0, 1.0, 50), pde=bins(0.0, 32.0, 64), pae=bins(0.0, 32.0, 64)
        ),
        metrics=SimpleNamespace(clash=SimpleNamespace(af3_clash_threshold=1.1)),
    )


def confidence_inputs() -> dict[str, np.ndarray]:
    """Two 12-token protein chains (4 atoms/token) + a 6-token ligand; chain ids have a gap."""
    rng = np.random.default_rng(2)
    n_token, n_atom = 30, 102
    token_asym_id = np.array([0] * 12 + [2] * 12 + [3] * 6, dtype=np.int64)
    atom_to_token_idx = np.concatenate([np.repeat(np.arange(24), 4), np.arange(24, 30)]).astype(
        np.int64
    )
    token_has_frame = np.array([True] * 24 + [False] * 6)
    token_has_frame[[4, 17]] = False
    coord = rng.uniform(0.0, 30.0, size=(2, n_atom, 3)).astype(np.float32)
    coord[1, :96] = rng.uniform(0.0, 3.0, size=(96, 3))  # dense -> AF3 clash in sample 1
    return {
        "pae_logits": (2.0 * rng.standard_normal((2, n_token, n_token, 64))).astype(np.float32),
        "plddt_logits": (2.0 * rng.standard_normal((2, n_atom, 50))).astype(np.float32),
        "pde_logits": (2.0 * rng.standard_normal((2, n_token, n_token, 64))).astype(np.float32),
        "contact_probs": rng.uniform(0.0, 1.0, size=(n_token, n_token)).astype(np.float32),
        "token_asym_id": token_asym_id,
        "token_has_frame": token_has_frame,
        "atom_coordinate": coord,
        "atom_to_token_idx": atom_to_token_idx,
        "atom_is_polymer": np.array([1] * 96 + [0] * 6, dtype=np.int64),
    }


def case_confidence() -> dict[str, np.ndarray]:
    from opendde.model.sample_confidence import compute_full_data_and_summary

    out: dict[str, np.ndarray] = {}
    inputs = confidence_inputs()
    out.update({f"in:{k}": v for k, v in inputs.items()})
    tensors = {k: torch.from_numpy(v) for k, v in inputs.items()}
    summary, full_data = compute_full_data_and_summary(
        confidence_configs(), **tensors, N_recycle=3, return_full_data=True
    )
    assert bool(summary[1]["has_clash"]) and not bool(summary[0]["has_clash"])
    for i, (summary_i, full_i) in enumerate(zip(summary, full_data)):
        out.update({f"summary{i}:{k}": v.numpy() for k, v in summary_i.items()})
        out.update({f"full{i}:{k}": v.numpy() for k, v in full_i.items()})
    return out


def clash_inputs() -> dict[str, np.ndarray]:
    """Protein (10 tokens x 3 atoms), DNA (5 x 4) and a ligand (5 x 1) bonded to the protein."""
    rng = np.random.default_rng(3)
    n_atom = 55
    coord = rng.uniform(0.0, 20.0, size=(2, n_atom, 3)).astype(np.float32)
    coord[1] = rng.uniform(0.0, 2.0, size=(n_atom, 3))
    atom_type = np.array([2] * 30 + [3] * 20 + [1] * 5)
    elements = np.zeros((n_atom, 10), dtype=np.float32)
    elements[np.arange(n_atom), rng.integers(0, 10, size=n_atom)] = 1.0
    return {
        "pred_coordinate": coord,
        "asym_id": np.array([0] * 10 + [1] * 5 + [2] * 5, dtype=np.int64),
        "atom_to_token_idx": np.concatenate(
            [np.repeat(np.arange(10), 3), np.repeat(np.arange(10, 15), 4), np.arange(15, 20)]
        ).astype(np.int64),
        "is_ligand": (atom_type == 1).astype(np.int64),
        "is_protein": (atom_type == 2).astype(np.int64),
        "is_dna": (atom_type == 3).astype(np.int64),
        "is_rna": (atom_type == 4).astype(np.int64),
        "mol_id": np.array([0] * 30 + [1] * 20 + [0] * 5, dtype=np.int64),
        "elements_one_hot": elements,
    }


def case_clash() -> dict[str, np.ndarray]:
    from opendde.metrics.clash import Clash

    out: dict[str, np.ndarray] = {}
    inputs = clash_inputs()
    out.update({f"in:{k}": v for k, v in inputs.items()})
    result = Clash()(**{k: torch.from_numpy(v) for k, v in inputs.items()})
    out["summary:af3_clash"] = result["summary"]["af3_clash"].numpy()
    out["summary:vdw_clash"] = result["summary"]["vdw_clash"].numpy()
    out["summary:chain_types"] = np.array(result["summary"]["chain_types"])
    out["summary:skipped_pairs"] = np.array(result["summary"]["skipped_pairs"], dtype=np.int64)
    out["details:af3_clash"] = result["details"]["af3_clash"].numpy()
    for (s, i, j), pairs in result["details"]["vdw_clash"].items():
        out[f"details:vdw_clash:{s}_{i}_{j}"] = pairs.numpy()
    return out


CASES = {
    "structural": case_structural,
    "shape_comp": case_shape_comp,
    "confidence": case_confidence,
    "clash": case_clash,
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
