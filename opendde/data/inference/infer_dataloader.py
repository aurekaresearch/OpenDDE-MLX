# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import json
import logging
import os
import time
import traceback
import warnings
from typing import Any

import numpy as np
from biotite.structure import AtomArray

from opendde.data.core import ccd
from opendde.data.inference.input_validation import validate_inference_jobs
from opendde.data.inference.json_to_feature import SampleDictToFeatures
from opendde.data.msa.msa_featurizer import InferenceMSAFeaturizer
from opendde.data.template.template_featurizer import InferenceTemplateFeaturizer
from opendde.data.template.template_utils import TemplateHitFeaturizer
from opendde.data.utils import data_type_transform, make_dummy_feature

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", module="biotite")


def dict_to_array(feature_dict: dict[str, Any]) -> dict[str, np.ndarray]:
    """Convert MSA/template features to arrays: integer dtypes -> int64, floats -> float32."""
    for key, value in feature_dict.items():
        value = np.asarray(value)
        if np.issubdtype(value.dtype, np.integer):
            value = value.astype(np.int64)
        elif np.issubdtype(value.dtype, np.floating):
            value = value.astype(np.float32)
        feature_dict[key] = value
    return feature_dict


class InferenceDataset:
    """Featurize inference jobs from an input JSON (or pre-validated ``inputs``)."""

    def __init__(
        self,
        configs,
        inputs: list[dict[str, Any]] | None = None,
    ) -> None:
        self.configs = configs

        self.input_json_path = configs.input_json_path
        self.dump_dir = configs.dump_dir
        self.use_msa = configs.use_msa
        self.msa_pair_as_unpair = configs.get("msa_pair_as_unpair", True)
        self.use_rna_msa = configs.get("use_rna_msa", True)
        self.use_template = configs.get("use_template", True)
        ccd.set_ccd_cache_paths(
            components_file=configs.data.ccd_components_file,
            rdkit_mol_pkl=configs.data.ccd_components_rdkit_mol_file,
        )
        if inputs is None:
            with open(self.input_json_path, "r") as f:
                inputs = validate_inference_jobs(json.load(f))
        self.inputs = inputs
        if self.use_template:
            template_mmcif_dir = configs.data.template.prot_template_mmcif_dir
            fetch_remote = configs.data.template.get("fetch_remote", True)
            if not fetch_remote:
                assert template_mmcif_dir is not None and os.path.exists(template_mmcif_dir), (
                    "Inference with template depends on the mmcif directory.\n"
                    "The mmcif directory containing cif files should be placed under "
                    "$OPENDDE_ROOT_DIR/search_database/mmcif.\n"
                    "You can download it from PDB https://www.wwpdb.org/ftp/pdb-ftp-sites or\n"
                    "refer to scripts/download_opendde_data.sh to download inference dependency "
                    "files, set use_template=false for inference, or set "
                    "data.template.fetch_remote=true to download mmCIF files on demand from PDBe."
                )
            elif template_mmcif_dir:
                os.makedirs(template_mmcif_dir, exist_ok=True)
            self.online_template_featurizer = TemplateHitFeaturizer(
                mmcif_dir=configs.data.template.prot_template_mmcif_dir,
                template_cache_dir=configs.data.template.prot_template_cache_dir,
                max_hits=4,
                kalign_binary_path=configs.data.template.kalign_binary_path,
                max_template_date="2021-09-30",
                release_dates_path=configs.data.template.release_dates_path,
                obsolete_pdbs_path=configs.data.template.obsolete_pdbs_path,
                _shuffle_top_k_prefiltered=None,
                _max_template_candidates_num=20,
                fetch_remote=fetch_remote,
            )
        else:
            self.online_template_featurizer = None

    def process_one(
        self,
        single_sample_dict: dict[str, Any],
    ) -> tuple[dict[str, Any], AtomArray, dict[str, float]]:
        """
        Featurize a single sample.

        Returns:
            A tuple of (feature/dimension dict, AtomArray, time tracking statistics).
        """
        t0 = time.time()
        sample2feat = SampleDictToFeatures(single_sample_dict)
        features_dict, atom_array, token_array = sample2feat.get_feature_dict()
        features_dict["distogram_rep_atom_mask"] = np.asarray(
            atom_array.distogram_rep_atom_mask, dtype=np.int64
        )
        # Includes ligands as well.
        entity_poly_type_and_seqs = sample2feat.entity_poly_type_and_seqs
        t1 = time.time()
        msa_features = (
            InferenceMSAFeaturizer.make_msa_feature(
                bioassembly=single_sample_dict["sequences"],
                atom_array=atom_array,
                msa_pair_as_unpair=self.msa_pair_as_unpair,
                use_rna_msa=self.use_rna_msa,
            )
            if self.use_msa
            else {}
        )
        template_features = InferenceTemplateFeaturizer.make_template_feature(
            bioassembly=single_sample_dict["sequences"],
            atom_array=atom_array,
            use_template=self.use_template,
            online_template_featurizer=self.online_template_featurizer,
        )
        # Make dummy features for not implemented features
        dummy_feats = []
        if len(template_features) == 0:
            dummy_feats.append("template")
        else:
            features_dict.update(dict_to_array(template_features))
        if len(msa_features) == 0:
            dummy_feats.append("msa")
        else:
            features_dict.update(dict_to_array(msa_features))
        features_dict = make_dummy_feature(features_dict=features_dict, dummy_feats=dummy_feats)
        feat = data_type_transform(feat_or_label_dict=features_dict)
        t2 = time.time()

        data: dict[str, Any] = {"input_feature_dict": feat}
        stats = {}
        for mol_type in ["ligand", "protein", "dna", "rna"]:
            mol_type_mask = feat[f"is_{mol_type}"].astype(bool)
            stats[f"{mol_type}/atom"] = int(mol_type_mask.sum())
            stats[f"{mol_type}/token"] = len(np.unique(feat["atom_to_token_idx"][mol_type_mask]))
        data.update(
            {
                "N_asym": len(np.unique(feat["asym_id"])),
                "N_token": feat["token_index"].shape[0],
                "N_atom": feat["atom_to_token_idx"].shape[0],
                "N_msa": feat["msa"].shape[0],
            }
        )

        def formatted_key(key):
            type_, unit = key.split("/")
            type_ = {"protein": "prot", "ligand": "lig"}.get(type_, type_)
            return f"N_{type_}_{unit}"

        data.update({formatted_key(k): v for k, v in stats.items()})
        data["entity_poly_type"] = entity_poly_type_and_seqs["entity_poly_type"]
        time_tracker = {"parse": t1 - t0, "featurizer": t2 - t1}
        return data, atom_array, time_tracker

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[dict[str, Any], AtomArray | None, str]:
        sample_name = f"job_{index}"
        try:
            single_sample_dict = self.inputs[index]
            sample_name = single_sample_dict["name"]
            logger.info(f"Featurizing {sample_name}...")
            data, atom_array, _ = self.process_one(single_sample_dict=single_sample_dict)
            error_message = ""
        except Exception as e:
            data, atom_array = {}, None
            error_message = f"{e}:\n{traceback.format_exc()}"
        data["sample_name"] = sample_name
        data["sample_index"] = index
        return data, atom_array, error_message
