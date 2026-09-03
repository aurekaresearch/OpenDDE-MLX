# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Inter-chain steric clash detection (AF3 and van der Waals criteria), NumPy only."""

import logging
from typing import Any, Optional

import numpy as np

from opendde.data.constants import rdkit_vdws

logger = logging.getLogger(__name__)

RDKIT_VDWS = np.asarray(rdkit_vdws, dtype=np.float32)
ID2TYPE = {0: "UNK", 1: "lig", 2: "prot", 3: "dna", 4: "rna"}


def get_vdw_radii(elements_one_hot: np.ndarray) -> np.ndarray:
    """Van der Waals radius per atom from its element one-hot ``[N_atom, N_elem]``."""
    return RDKIT_VDWS[elements_one_hot.argmax(axis=1)]


def _remap_contiguous(asym_id: np.ndarray) -> np.ndarray:
    """Relabel chain ids to ``0..N_chain-1`` (chains may have been filtered out)."""
    return np.unique(asym_id, return_inverse=True)[1]


class Clash:
    """Flag chain pairs whose predicted coordinates clash.

    AF3 criterion: > 100 atom pairs (or > 50% of the smaller chain) closer than
    ``af3_clash_threshold`` between two polymer chains. VDW criterion: any atom
    pair closer than ``vdw_clash_threshold`` times the sum of their radii.
    """

    def __init__(
        self,
        af3_clash_threshold: float = 1.1,
        vdw_clash_threshold: float = 0.75,
        compute_af3_clash: bool = True,
        compute_vdw_clash: bool = True,
    ) -> None:
        self.af3_clash_threshold = af3_clash_threshold
        self.vdw_clash_threshold = vdw_clash_threshold
        self.compute_af3_clash = compute_af3_clash
        self.compute_vdw_clash = compute_vdw_clash

    def __call__(
        self,
        pred_coordinate: Any,
        asym_id: Any,
        atom_to_token_idx: Any,
        is_ligand: Any,
        is_protein: Any,
        is_dna: Any,
        is_rna: Any,
        mol_id: Optional[Any] = None,
        elements_one_hot: Optional[Any] = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Args:
            pred_coordinate: ``[N_sample, N_atom, 3]``; asym_id ``[N_token]``;
            atom_to_token_idx and the ``is_*`` atom flags ``[N_atom]``.
        """
        chain_info = self.get_chain_info(
            asym_id,
            atom_to_token_idx,
            is_ligand,
            is_protein,
            is_dna,
            is_rna,
            mol_id,
            elements_one_hot,
        )
        return self._check_clash_per_chain_pairs(np.asarray(pred_coordinate), **chain_info)

    def get_chain_info(
        self,
        asym_id: Any,
        atom_to_token_idx: Any,
        is_ligand: Any,
        is_protein: Any,
        is_dna: Any,
        is_rna: Any,
        mol_id: Optional[Any] = None,
        elements_one_hot: Optional[Any] = None,
    ) -> dict[str, Any]:
        asym_id = _remap_contiguous(np.asarray(asym_id).astype(np.int64))
        atom_to_token_idx = np.asarray(atom_to_token_idx).astype(np.int64)
        n_chains = int(asym_id.max()) + 1
        atom_type = (
            1 * np.asarray(is_ligand)
            + 2 * np.asarray(is_protein)
            + 3 * np.asarray(is_dna)
            + 4 * np.asarray(is_rna)
        ).astype(np.int64)
        if self.compute_vdw_clash:
            assert mol_id is not None and elements_one_hot is not None
            mol_id = np.asarray(mol_id)
            elements_one_hot = np.asarray(elements_one_hot)

        chain_types, asym_id_to_mol_id = [], {}
        atom_asym_id = asym_id[atom_to_token_idx]
        for aid in range(n_chains):
            atom_type_i = np.unique(atom_type[atom_asym_id == aid])
            assert len(atom_type_i) == 1
            if atom_type_i[0] == 0:
                logger.warning("Unknown asym_id type: not in ligand / protein / dna / rna")
            chain_types.append(ID2TYPE[int(atom_type_i[0])])
            if self.compute_vdw_clash:
                asym_id_to_mol_id[aid] = int(np.unique(mol_id[atom_asym_id == aid]).item())
        return {
            "n_chains": n_chains,
            "atom_asym_id": atom_asym_id,
            "chain_types": chain_types,
            "elements_one_hot": elements_one_hot,
            "asym_id_to_mol_id": asym_id_to_mol_id,
        }

    def get_chain_pair_violations(
        self,
        pred_coordinate: np.ndarray,
        violation_type: str,
        chain_1_mask: np.ndarray,
        chain_2_mask: np.ndarray,
        elements_one_hot: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Clashing atom pairs ``[N_clash, 2]`` (af3) or ``[N_clash, 3]`` with global ids (vdw)."""
        coords_1 = pred_coordinate[chain_1_mask]
        coords_2 = pred_coordinate[chain_2_mask]
        pred_dist = np.linalg.norm(coords_1[:, None, :] - coords_2[None, :, :], axis=-1)
        if violation_type == "af3":
            row, col = np.nonzero(pred_dist < self.af3_clash_threshold)
            return np.stack((row, col), axis=-1)
        assert elements_one_hot is not None
        vdw_sum = (
            get_vdw_radii(elements_one_hot[chain_1_mask])[:, None]
            + get_vdw_radii(elements_one_hot[chain_2_mask])[None, :]
        )
        relative = pred_dist / vdw_sum
        row, col = np.nonzero(relative < self.vdw_clash_threshold)
        global_row = np.flatnonzero(chain_1_mask)[row]
        global_col = np.flatnonzero(chain_2_mask)[col]
        return np.stack((global_row, global_col, relative[row, col]), axis=-1)

    def _check_clash_per_chain_pairs(
        self,
        pred_coordinate: np.ndarray,
        n_chains: int,
        atom_asym_id: np.ndarray,
        chain_types: list[str],
        elements_one_hot: Optional[np.ndarray],
        asym_id_to_mol_id: dict[int, int],
    ) -> dict[str, dict[str, Any]]:
        n_sample = pred_coordinate.shape[0]
        af3_flag = np.zeros((n_sample, n_chains, n_chains), dtype=bool)
        af3_details = np.zeros((n_sample, n_chains, n_chains, 2), dtype=np.float32)
        vdw_flag = np.zeros((n_sample, n_chains, n_chains), dtype=bool)
        vdw_details: dict[tuple[int, int, int], np.ndarray] = {}
        skipped_pairs: list[tuple[int, int]] = []
        chain_masks = [atom_asym_id == aid for aid in range(n_chains)]

        for sample_id in range(n_sample):
            coords = pred_coordinate[sample_id]
            for i in range(n_chains):
                if chain_types[i] == "UNK":
                    continue
                for j in range(i + 1, n_chains):
                    if chain_types[j] == "UNK":
                        continue
                    pair_type = {chain_types[i], chain_types[j]}
                    skip_bonded_ligand = (
                        self.compute_vdw_clash
                        and "lig" in pair_type
                        and len(pair_type) > 1
                        and asym_id_to_mol_id[i] == asym_id_to_mol_id[j]
                    )
                    if skip_bonded_ligand:
                        logger.warning(
                            "mol_id %d may contain bonded ligand to polymers", asym_id_to_mol_id[i]
                        )
                        skipped_pairs.append((i, j))
                    if self.compute_vdw_clash and not skip_bonded_ligand:
                        pairs = self.get_chain_pair_violations(
                            coords, "vdw", chain_masks[i], chain_masks[j], elements_one_hot
                        )
                        if pairs.shape[0] > 0:
                            vdw_details[(sample_id, i, j)] = pairs
                            vdw_flag[sample_id, i, j] = vdw_flag[sample_id, j, i] = True
                    if "lig" in pair_type or not self.compute_af3_clash:
                        continue  # AF3 clash only considers polymer chains
                    total = self.get_chain_pair_violations(
                        coords, "af3", chain_masks[i], chain_masks[j]
                    ).shape[0]
                    relative = total / min(chain_masks[i].sum(), chain_masks[j].sum())
                    af3_details[sample_id, i, j] = af3_details[sample_id, j, i] = (total, relative)
                    af3_flag[sample_id, i, j] = af3_flag[sample_id, j, i] = (
                        total > 100 or relative > 0.5
                    )
        return {
            "summary": {
                "af3_clash": af3_flag if self.compute_af3_clash else None,
                "vdw_clash": vdw_flag if self.compute_vdw_clash else None,
                "chain_types": chain_types,
                "skipped_pairs": skipped_pairs,
            },
            "details": {
                "af3_clash": af3_details if self.compute_af3_clash else None,
                "vdw_clash": vdw_details if self.compute_vdw_clash else None,
            },
        }
