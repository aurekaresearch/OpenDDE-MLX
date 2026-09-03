# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Default model and confidence configuration for ``opendde_v1``."""

from typing import Any

from opendde.config.extend_types import GlobalConfigValue, ListValue, ValueMaybeNone

runtime_configs: dict[str, Any] = {
    "load_checkpoint_path": "",
    "load_strict": True,
}

model_configs: dict[str, Any] = {
    "c_s": 384,
    "c_z": 384,
    "c_s_inputs": 449,  # c_token + 32 + 32 + 1
    "c_atom": 128,
    "c_atompair": 16,
    "c_token": 384,
    "n_blocks": 48,
    "max_atoms_per_token": 24,
    "no_bins": 96,
    "sigma_data": 16.0,
    "hidden_scale_up": True,
    "enable_diffusion_shared_vars_cache": False,
    "enable_efficient_fusion": False,
    # Inference defaults to bf16; fp32 and fp16 are available via --dtype.
    "dtype": "bf16",
    # Stages that always run in fp32 even under a reduced dtype.
    "skip_amp": {
        "sample_diffusion": True,
        "confidence_head": True,
    },
    "infer_setting": {
        "chunk_size": ValueMaybeNone(256),
        "dynamic_chunk_size": True,
        "chunk_size_thresholds": {
            "1024": -1,
            "1536": 512,
            "2048": 256,
            "2560": 128,
        },
        "sample_diffusion_chunk_size": ValueMaybeNone(5),
        # Cache the per-block diffusion pair biases when they fit in this budget.
        "diffusion_bias_cache_gb": 6.0,
    },
    "inference_noise_scheduler": {
        "s_max": 160.0,
        "s_min": 4e-4,
        "rho": 7,
        "sigma_data": 16.0,
    },
    "sample_diffusion": {
        "gamma0": 0.8,
        "gamma_min": 1.0,
        "noise_scale_lambda": 1.003,
        "step_scale_eta": 1.5,
        "N_step": 200,
        "N_sample": 5,
    },
    "model": {
        "N_model_seed": 1,
        "N_cycle": 10,
        "input_embedder": {
            "c_atom": GlobalConfigValue("c_atom"),
            "c_atompair": GlobalConfigValue("c_atompair"),
            "c_token": GlobalConfigValue("c_token"),
        },
        "relative_position_encoding": {
            "r_max": 32,
            "s_max": 2,
            "c_z": GlobalConfigValue("c_z"),
        },
        "template_embedder": {
            "c": 64,
            "c_z": GlobalConfigValue("c_z"),
            "n_blocks": 2,
            "hidden_scale_up": GlobalConfigValue("hidden_scale_up"),
        },
        "msa_module": {
            "c_m": 128,
            "c_z": GlobalConfigValue("c_z"),
            "c_s_inputs": GlobalConfigValue("c_s_inputs"),
            "n_blocks": 4,
            "hidden_scale_up": GlobalConfigValue("hidden_scale_up"),
            "msa_chunk_size": ValueMaybeNone(2048),
        },
        "pairformer": {
            "n_blocks": GlobalConfigValue("n_blocks"),
            "c_z": GlobalConfigValue("c_z"),
            "c_s": GlobalConfigValue("c_s"),
            "n_heads": 16,
            "hidden_scale_up": GlobalConfigValue("hidden_scale_up"),
        },
        "structural_token_expansion": {
            "enable": True,
            "n_roles": 7,
            "pair_projection_mode": "full",
            "pair_chunk_size": ValueMaybeNone(128),
            "pair_output_space": "residue",
            "structural_refiner": {
                "enable": True,
                "n_blocks": 4,
                "n_heads": 8,
                "num_intermediate_factor": 2,
                "hidden_scale_up": True,
            },
        },
        "diffusion_module": {
            "sigma_data": GlobalConfigValue("sigma_data"),
            "c_token": 768,
            "c_atom": GlobalConfigValue("c_atom"),
            "c_atompair": GlobalConfigValue("c_atompair"),
            "c_z": GlobalConfigValue("c_z"),
            "c_z_pair_diffusion": 128,
            "c_s": GlobalConfigValue("c_s"),
            "c_s_inputs": GlobalConfigValue("c_s_inputs"),
            "atom_encoder": {"n_blocks": 3, "n_heads": 4},
            "transformer": {"n_blocks": 24, "n_heads": 16},
            "atom_decoder": {"n_blocks": 3, "n_heads": 4},
        },
        "confidence_head": {
            "c_z": GlobalConfigValue("c_z"),
            "c_s": GlobalConfigValue("c_s"),
            "c_s_inputs": GlobalConfigValue("c_s_inputs"),
            "n_blocks": 4,
            "max_atoms_per_token": GlobalConfigValue("max_atoms_per_token"),
            "hidden_scale_up": GlobalConfigValue("hidden_scale_up"),
            "distance_bin_start": 3.25,
            "distance_bin_end": 52.0,
            "distance_bin_step": 1.25,
        },
        "distogram_head": {
            "c_z": GlobalConfigValue("c_z"),
            "no_bins": GlobalConfigValue("no_bins"),
        },
    },
}

confidence_configs: dict[str, Any] = {
    "confidence": {
        "weight": {"alpha_shape_comp": 3e-2},
        "plddt": {"min_bin": 0, "max_bin": 1.0, "no_bins": 50, "normalize": True, "eps": 1e-6},
        "pde": {"min_bin": 0, "max_bin": 32, "no_bins": 64, "eps": 1e-6},
        "resolved": {"eps": 1e-6},
        "pae": {"min_bin": 0, "max_bin": 32, "no_bins": 64, "eps": 1e-6},
        "distogram": {"min_bin": 2.25, "max_bin": 25.75, "no_bins": 96, "eps": 1e-6},
        "shape_comp": {
            "pair_weight": 0.4,
            "token_weight": 1.0,
            "global_weight": 0.4,
            "density_sigma": 1.5,
            "interface_cutoff": 16.0,
            "gap_mean": 6.0,
            "gap_scale": 3.0,
            "clash_distance": 2.0,
            "clash_scale": 0.5,
            "pool_temperature": 16.0,
            "normal_strength_min": 1e-4,
            "pair_chunk_size": ValueMaybeNone(128),
            "debug_pair_map": False,
            "eps": 1e-6,
        },
    },
    "metrics": {
        "complex_ranker_keys": ListValue(["plddt", "gpde", "ranking_score"]),
        "chain_ranker_keys": ListValue(["chain_ptm", "chain_plddt"]),
        "interface_ranker_keys": ListValue(
            ["chain_pair_iptm", "chain_pair_iptm_global", "chain_pair_plddt"]
        ),
        "clash": {"af3_clash_threshold": 1.1, "vdw_clash_threshold": 0.75},
    },
}

configs: dict[str, Any] = {**runtime_configs, **model_configs, **confidence_configs}
