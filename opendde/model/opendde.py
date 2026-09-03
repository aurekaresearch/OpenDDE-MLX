# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""The OpenDDE prediction loop: trunk recycling, structural token expansion,
diffusion sampling, distogram and confidence heads."""

import time
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from opendde.config.schema import OpenDDEConfig
from opendde.data.tokenizer import STRUCTURAL_TOKEN_ROLES
from opendde.model import sample_confidence
from opendde.model.confidence import ConfidenceHead
from opendde.model.diffusion import DiffusionModule
from opendde.model.embedders import InputFeatureEmbedder, RelativePositionEncoding
from opendde.model.generator import InferenceNoiseScheduler, sample_diffusion
from opendde.model.head import DistogramHead
from opendde.model.pairformer import MSAModule, PairformerStack, TemplateEmbedder
from opendde.model.primitives import LayerNorm, LinearNoBias
from opendde.model.shape_complementarity import (
    build_shape_comp_pred_outputs,
    compute_shape_complementarity_fields,
    get_shape_comp_atom_mask,
)
from opendde.model.structural_tokens import StructuralTokenExpander
from opendde.model.transformer import update_input_feature_dict
from opendde.utils.logger import get_logger

logger = get_logger(__name__)

_RESIDUE_ONLY_KEYS = {
    "msa",
    "has_deletion",
    "deletion_value",
    "msa_mask",
    "profile",
    "deletion_mean",
    "token_bonds",
}
_STRUCTURAL_REQUIRED = (
    "parent_residue_idx",
    "subtoken_role_id",
    "structural_token_index",
    "atom_to_structural_token_idx",
    "atom_to_structural_tokatom_idx",
    "structural_distogram_rep_atom_mask",
    "structural_pae_rep_atom_mask",
    "structural_has_frame",
    "structural_frame_atom_index",
)
_RESIDUE_LEVEL_KEYS = (
    "token_index",
    "asym_id",
    "residue_index",
    "entity_id",
    "sym_id",
    "atom_to_token_idx",
    "atom_to_tokatom_idx",
    "has_frame",
    "frame_atom_index",
    "pae_rep_atom_mask",
    "distogram_rep_atom_mask",
)


class OpenDDE(nn.Module):
    """Implements the OpenDDE inference pipeline."""

    def __init__(self, configs: OpenDDEConfig) -> None:
        super().__init__()
        self.configs = configs
        self.N_cycle = configs.model.N_cycle
        self.N_model_seed = configs.model.N_model_seed
        self.inference_noise_scheduler = InferenceNoiseScheduler(
            **configs.inference_noise_scheduler
        )

        self.input_embedder = InputFeatureEmbedder(**configs.model.input_embedder)
        self.relative_position_encoding = RelativePositionEncoding(
            **configs.model.relative_position_encoding
        )
        self.template_embedder = TemplateEmbedder(**configs.model.template_embedder)
        self.msa_module = MSAModule(**configs.model.msa_module, msa_configs=configs.data["msa"])
        self.pairformer_stack = PairformerStack(**configs.model.pairformer)
        self.diffusion_module = DiffusionModule(**configs.model.diffusion_module.to_dict())
        self.distogram_head = DistogramHead(**configs.model.distogram_head)
        self.confidence_head = ConfidenceHead(**configs.model.confidence_head)

        c_s, c_z, c_s_inputs = configs.c_s, configs.c_z, configs.c_s_inputs
        self.linear_no_bias_sinit = LinearNoBias(c_s_inputs, c_s)
        self.linear_no_bias_zinit1 = LinearNoBias(c_s, c_z)
        self.linear_no_bias_zinit2 = LinearNoBias(c_s, c_z)
        self.linear_no_bias_token_bond = LinearNoBias(1, c_z)
        self.linear_no_bias_z_cycle = LinearNoBias(c_z, c_z, initializer="zeros")
        self.linear_no_bias_s = LinearNoBias(c_s, c_s, initializer="zeros")
        self.layernorm_z_cycle = LayerNorm(c_z)
        self.layernorm_s = LayerNorm(c_s)

        ste = configs.model.structural_token_expansion
        self.enable_structural_token_expansion = ste.enable
        self.pair_output_space = ste.pair_output_space
        if self.pair_output_space not in {"residue", "structural"}:
            raise ValueError("pair_output_space must be 'residue' or 'structural'")
        self.enable_structural_token_refiner = ste.enable and ste.structural_refiner.enable
        if ste.enable:
            self.structural_token_expander = StructuralTokenExpander(
                c_s=c_s,
                c_z=c_z,
                c_s_inputs=c_s_inputs,
                n_roles=ste.n_roles,
                pair_projection_mode=ste.pair_projection_mode,
                pair_chunk_size=ste.pair_chunk_size,
            )
            if self.enable_structural_token_refiner:
                refiner = ste.structural_refiner
                self.structural_token_refiner = PairformerStack(
                    n_blocks=refiner.n_blocks,
                    n_heads=refiner.n_heads,
                    c_z=c_z,
                    c_s=c_s,
                    num_intermediate_factor=refiner.num_intermediate_factor,
                    hidden_scale_up=refiner.hidden_scale_up,
                )

    # ------------------------------------------------------------------ dtype
    def set_compute_dtype(
        self, dtype: mx.Dtype, fp32_diffusion: bool, fp32_confidence: bool
    ) -> None:
        """Cast trunk weights to ``dtype``; keep fp32-critical layers and selected stages in fp32."""
        self.set_dtype(mx.float32)
        if dtype == mx.float32:
            return
        self.set_dtype(dtype)
        for _, module in self.named_modules():
            if getattr(module, "precision", False) or type(module).__name__ == "FourierEmbedding":
                module.set_dtype(mx.float32)
        if fp32_diffusion:
            self.diffusion_module.set_dtype(mx.float32)
        if fp32_confidence:
            self.confidence_head.set_dtype(mx.float32)
        for name in ("lower_bins", "upper_bins"):
            setattr(
                self.confidence_head, name, getattr(self.confidence_head, name).astype(mx.float32)
            )

    # ------------------------------------------------------------------ trunk
    def get_pairformer_output(
        self, feat: dict[str, Any], N_cycle: int, chunk_size: Optional[int] = None
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Algorithm 1 lines 1-13: input embedding and the recycled Pairformer trunk."""
        dtype = self.linear_no_bias_sinit.weight.dtype
        s_inputs = self.input_embedder(feat).astype(dtype)  # [N_token, c_s_inputs]
        s_init = self.linear_no_bias_sinit(s_inputs)
        z_init = (
            self.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )
        z_init = z_init + self.relative_position_encoding(feat).astype(dtype)
        z_init = z_init + self.linear_no_bias_token_bond(
            feat["token_bonds"][..., None].astype(dtype)
        )
        mx.eval(s_inputs, s_init, z_init)

        z = mx.zeros_like(z_init)
        s = mx.zeros_like(s_init)
        for cycle in range(N_cycle):
            z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
            template_update = self.template_embedder(feat, z, chunk_size=chunk_size)
            if template_update is not None:
                z = z + template_update
            z = self.msa_module(feat, z, s_inputs, pair_mask=None, chunk_size=chunk_size)
            s = s_init + self.linear_no_bias_s(self.layernorm_s(s))
            s, z = self.pairformer_stack(s, z, pair_mask=None, chunk_size=chunk_size)
            mx.eval(s, z)
            logger.info("Trunk cycle %d/%d done", cycle + 1, N_cycle)
        return s_inputs, s, z

    # ------------------------------------------------------- structural tokens
    def expand_to_structural_tokens(
        self,
        feat: dict[str, Any],
        s_inputs: mx.array,
        s: mx.array,
        z: mx.array,
        chunk_size: Optional[int],
    ) -> tuple[dict[str, Any], mx.array, mx.array, mx.array]:
        """Expand residue tokens into structural (backbone/side-chain) tokens after the trunk."""
        if not self.enable_structural_token_expansion:
            return feat, s_inputs, s, z
        missing = [k for k in _STRUCTURAL_REQUIRED if k not in feat]
        if missing:
            raise KeyError(f"Structural token expansion needs features: {missing}")

        struct = dict(feat)
        for key in _RESIDUE_LEVEL_KEYS:
            struct[f"residue_level_{key}"] = feat[key]
        parent = feat["parent_residue_idx"]
        s_inputs, s, z, pair_features = self.structural_token_expander(feat, s_inputs, s, z)
        mx.eval(s_inputs, s, z, *pair_features.values())
        struct["token_index"] = feat["structural_token_index"]
        struct["atom_to_token_idx"] = feat["atom_to_structural_token_idx"]
        struct["atom_to_tokatom_idx"] = feat["atom_to_structural_tokatom_idx"]
        for key in ("asym_id", "residue_index", "entity_id", "sym_id"):
            struct[key] = mx.take(feat[key], parent, axis=-1)
        struct["has_frame"] = feat["structural_has_frame"]
        struct["frame_atom_index"] = feat["structural_frame_atom_index"]
        struct["pae_rep_atom_mask"] = feat["structural_pae_rep_atom_mask"]
        struct["distogram_rep_atom_mask"] = feat["structural_distogram_rep_atom_mask"]
        struct.update(pair_features)
        if self.enable_structural_token_refiner:
            s, z = self.structural_token_refiner(
                s,
                z,
                pair_mask=None,
                chunk_size=chunk_size,
                extra_attn_bias=struct.get("structural_pair_attn_bias"),
            )
        mx.eval(s_inputs, s, z)
        for key in list(struct):
            if key in _RESIDUE_ONLY_KEYS or key.startswith("template_"):
                struct.pop(key)
        return struct, s_inputs, s, z

    # ------------------------------------------------------ residue-level pooling
    @staticmethod
    def get_parent_representative_token_idx(
        parent: np.ndarray, role: np.ndarray, n_residue: int
    ) -> np.ndarray:
        """Backbone token (or first token) representing each parent residue."""
        backbone = {STRUCTURAL_TOKEN_ROLES[k] for k in ("protein_bb", "dna_bb", "rna_bb")}
        rep = np.full(n_residue, -1, dtype=np.int64)
        for idx, p in enumerate(parent):
            if rep[p] < 0:
                rep[p] = idx
        for idx, (p, r) in enumerate(zip(parent, role)):
            if r in backbone:
                rep[p] = idx
        if np.any(rep < 0):
            raise ValueError(f"No structural representative token for every residue: {rep}")
        return rep

    @staticmethod
    def pool_pair_matrix_to_residue_max(
        values: mx.array, parent: np.ndarray, n_residue: int
    ) -> mx.array:
        """Max-pool a structural ``[..., N_s, N_s]`` matrix into residue pairs."""
        pair_index = mx.array((parent[:, None] * n_residue + parent[None, :]).reshape(-1))
        flat = values.reshape(*values.shape[:-2], -1)
        out = mx.full(
            (*flat.shape[:-1], n_residue * n_residue),
            float(np.finfo(np.float32).min),
            dtype=values.dtype,
        )
        out = out.at[..., pair_index].maximum(flat)
        return out.reshape(*flat.shape[:-1], n_residue, n_residue)

    def get_residue_level_confidence_inputs(
        self,
        feat: dict[str, Any],
        pae_logits: mx.array,
        pde_logits: mx.array,
        contact_probs: mx.array,
    ) -> dict[str, mx.array]:
        """Map structural-token pair logits back to residue tokens for the public outputs."""
        parent = feat.get("parent_residue_idx")
        role = feat.get("subtoken_role_id")
        has_residue = all(
            f"residue_level_{k}" in feat for k in ("asym_id", "has_frame", "atom_to_token_idx")
        )
        n_struct = None if parent is None else parent.shape[0]
        structural = (
            parent is not None
            and role is not None
            and has_residue
            and pae_logits.shape[-3] == n_struct
            and contact_probs.shape[-1] == n_struct
        )
        if not structural:
            return {
                "pae_logits": pae_logits,
                "pde_logits": pde_logits,
                "contact_probs": contact_probs,
                "token_asym_id": feat["asym_id"],
                "token_has_frame": feat["has_frame"],
                "atom_to_token_idx": feat["atom_to_token_idx"],
            }
        parent_np = np.asarray(parent)
        n_residue = int(parent_np.max()) + 1
        rep = mx.array(
            self.get_parent_representative_token_idx(parent_np, np.asarray(role), n_residue)
        )
        return {
            "pae_logits": mx.take(mx.take(pae_logits, rep, axis=-3), rep, axis=-2),
            "pde_logits": mx.take(mx.take(pde_logits, rep, axis=-3), rep, axis=-2),
            "contact_probs": self.pool_pair_matrix_to_residue_max(
                contact_probs.astype(mx.float32), parent_np, n_residue
            ),
            "token_asym_id": feat["residue_level_asym_id"],
            "token_has_frame": feat["residue_level_has_frame"],
            "atom_to_token_idx": feat["residue_level_atom_to_token_idx"],
        }

    # ------------------------------------------------------- shape complementarity
    def _should_compute_shape_comp(self) -> bool:
        cfg = self.configs.confidence.shape_comp
        alpha = float(self.configs.confidence.weight.alpha_shape_comp)
        weights = (cfg.pair_weight, cfg.token_weight, cfg.global_weight)
        return any(alpha * float(w) > 0 for w in weights) or bool(cfg.debug_pair_map)

    def add_shape_complementarity_predictions(
        self, pred: dict[str, Any], feat: dict[str, Any], coordinate: mx.array
    ) -> None:
        if not self._should_compute_shape_comp():
            return
        keep_pair_map = bool(self.configs.confidence.shape_comp.debug_pair_map)
        pred["shape_comp_uses_structural_tokens"] = mx.array(
            [int("residue_level_token_index" in feat)]
        )
        shape_comp = compute_shape_complementarity_fields(
            coordinate=coordinate.astype(mx.float32),
            feat_dict=feat,
            atom_mask=get_shape_comp_atom_mask(feat),
            return_pair_map=keep_pair_map,
            **self.configs.confidence.shape_comp,
        )
        pred.update(build_shape_comp_pred_outputs(shape_comp, keep_pair_map=keep_pair_map))

    # -------------------------------------------------------------- chunk sizes
    def _resolve_chunk_size(self, n_token: int) -> Optional[int]:
        """Attention chunk from the token-count threshold table, bounded by an N^2 score budget."""
        settings = self.configs.infer_setting
        if not settings.dynamic_chunk_size:
            return settings.chunk_size
        chunk_size: Optional[int] = 32
        for threshold, value in sorted(
            (int(k), v) for k, v in settings.chunk_size_thresholds.items()
        ):
            if n_token <= threshold:
                chunk_size = None if value == -1 else value
                break
        budget_chunk = max(1, 450_000_000 // max(1, n_token * n_token))
        power_of_two = 1 << (budget_chunk.bit_length() - 1)
        bounded = min(chunk_size or n_token, power_of_two)
        if chunk_size is None and bounded >= n_token:
            return None
        return bounded

    # ------------------------------------------------------------------ forward
    def _main_inference_loop(
        self, feat: dict[str, Any], N_cycle: int, seed: Optional[int]
    ) -> tuple[dict[str, Any], dict[str, float]]:
        t0 = time.time()
        N_token = feat["residue_index"].shape[-1]
        chunk_size = self._resolve_chunk_size(N_token)
        pred: dict[str, Any] = {}
        timing: dict[str, float] = {}

        s_inputs, s, z = self.get_pairformer_output(feat, N_cycle, chunk_size)
        residue_branch = (feat, s_inputs, s, z)
        structural_chunk = chunk_size
        if self.enable_structural_token_expansion:
            structural_chunk = self._resolve_chunk_size(feat["parent_residue_idx"].shape[-1])
        feat, s_inputs, s, z = self.expand_to_structural_tokens(
            feat, s_inputs, s, z, structural_chunk
        )
        if self.enable_structural_token_expansion and self.pair_output_space == "residue":
            pair_feat, pair_s_inputs, pair_s, pair_z = residue_branch
        else:
            pair_feat, pair_s_inputs, pair_s, pair_z = feat, s_inputs, s, z
        del residue_branch
        for key in list(feat):
            if key.startswith("template_") or key in {
                "msa",
                "has_deletion",
                "deletion_value",
                "profile",
                "deletion_mean",
            }:
                del feat[key]
        timing["pairformer"] = time.time() - t0
        logger.info(
            "Stage pairformer: %.1fs, peak memory %.1f GB",
            timing["pairformer"],
            mx.get_peak_memory() / 1e9,
        )

        # ---- diffusion sampling (fp32)
        t1 = time.time()
        s_inputs32, s32, z32 = (x.astype(mx.float32) for x in (s_inputs, s, z))
        sd = self.configs.sample_diffusion
        noise_schedule = self.inference_noise_scheduler(N_step=sd.N_step)
        diff_pair_z = p_lm = c_l = pair_bias_cache = None
        if self.configs.enable_diffusion_shared_vars_cache:
            diff_pair_z = self.diffusion_module.diffusion_conditioning.prepare_cache(feat, z32)
            p_lm, c_l = self.diffusion_module.atom_attention_encoder.prepare_cache(
                feat["ref_pos"],
                feat["ref_charge"],
                feat["ref_mask"],
                feat["ref_element"],
                feat["ref_atom_name_chars"],
                feat["atom_to_token_idx"],
                feat["d_lm"],
                feat["v_lm"],
                feat["pad_info"],
                z=diff_pair_z,
            )
            mx.eval(p_lm, c_l)
            transformer = self.diffusion_module.diffusion_transformer
            n_heads = transformer.blocks[0].attention_pair_bias.n_heads
            cache_bytes = transformer.n_blocks * n_heads * diff_pair_z.shape[-2] ** 2 * 4
            if cache_bytes <= self.configs.infer_setting.diffusion_bias_cache_gb * 1e9:
                pair_bias_cache = self.diffusion_module.prepare_pair_bias_cache(
                    diff_pair_z,
                    feat.get("structural_pair_attn_bias"),
                    self.configs.enable_efficient_fusion,
                )
        pred["coordinate"] = sample_diffusion(
            denoise_net=self.diffusion_module,
            feat=feat,
            s_inputs=s_inputs32,
            s_trunk=s32,
            z_trunk=None if diff_pair_z is not None else z32,
            pair_z=diff_pair_z,
            p_lm=p_lm,
            c_l=c_l,
            noise_schedule=noise_schedule,
            N_sample=sd.N_sample,
            gamma0=sd.gamma0,
            gamma_min=sd.gamma_min,
            noise_scale_lambda=sd.noise_scale_lambda,
            step_scale_eta=sd.step_scale_eta,
            diffusion_chunk_size=self.configs.infer_setting.sample_diffusion_chunk_size,
            enable_efficient_fusion=self.configs.enable_efficient_fusion,
            pair_bias_cache=pair_bias_cache,
            seed=seed,
        )
        del diff_pair_z, p_lm, c_l, pair_bias_cache
        timing["diffusion"] = time.time() - t1
        logger.info(
            "Stage diffusion: %.1fs, peak memory %.1f GB",
            timing["diffusion"],
            mx.get_peak_memory() / 1e9,
        )

        # ---- distogram contacts and confidence
        t2 = time.time()
        pred["contact_probs"] = sample_confidence.compute_contact_prob(
            distogram_logits=self.distogram_head(pair_z.astype(mx.float32)),
            **sample_confidence.get_bin_params(self.configs.confidence.distogram),
        )
        plddt, pae, pde, resolved = self.confidence_head(
            feat=pair_feat,
            s_inputs=pair_s_inputs,
            s_trunk=pair_s,
            z_trunk=pair_z,
            pair_mask=None,
            x_pred_coords=pred["coordinate"],
            chunk_size=chunk_size,
        )
        pred.update({"plddt": plddt, "pae": pae, "pde": pde, "resolved": resolved})
        mx.eval(pred["contact_probs"], plddt, pae, pde, resolved)
        timing["confidence"] = time.time() - t2
        logger.info(
            "Stage confidence: %.1fs, peak memory %.1f GB",
            timing["confidence"],
            mx.get_peak_memory() / 1e9,
        )
        del pair_s_inputs, pair_s, pair_z, s_inputs, s, z, s_inputs32, s32, z32

        # ---- post processing: shape complementarity, residue-level summaries
        self.add_shape_complementarity_predictions(pred, pair_feat, pred["coordinate"])
        residue_inputs = self.get_residue_level_confidence_inputs(
            pair_feat, pred["pae"], pred["pde"], pred["contact_probs"]
        )
        if pred["pae"].shape[-3:-1] != residue_inputs["pae_logits"].shape[-3:-1]:
            pred["structural_pae"], pred["structural_pde"] = pred["pae"], pred["pde"]
        pred["pae"], pred["pde"] = residue_inputs["pae_logits"], residue_inputs["pde_logits"]
        pred["contact_probs"] = residue_inputs["contact_probs"]
        pred["summary_confidence"], pred["full_data"] = (
            sample_confidence.compute_full_data_and_summary(
                configs=self.configs,
                pae_logits=residue_inputs["pae_logits"],
                plddt_logits=pred["plddt"],
                pde_logits=residue_inputs["pde_logits"],
                contact_probs=residue_inputs["contact_probs"],
                token_asym_id=residue_inputs["token_asym_id"],
                token_has_frame=residue_inputs["token_has_frame"],
                atom_coordinate=pred["coordinate"],
                atom_to_token_idx=residue_inputs["atom_to_token_idx"],
                atom_is_polymer=1 - feat["is_ligand"],
                N_recycle=N_cycle,
                return_full_data=bool(self.configs.need_atom_confidence),
            )
        )
        timing["model_forward"] = time.time() - t0
        return pred, timing

    def __call__(
        self, feat: dict[str, Any], seed: Optional[int] = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run structure prediction on one featurised input (values are ``mx.array``)."""
        feat = update_input_feature_dict(dict(feat))
        if self.N_model_seed == 1:
            pred, timing = self._main_inference_loop(feat, self.N_cycle, seed)
            return pred, {"time": timing}

        preds, timings = [], []
        for model_seed in range(self.N_model_seed):
            pred, timing = self._main_inference_loop(
                dict(feat), self.N_cycle, None if seed is None else seed + model_seed
            )
            preds.append(pred)
            timings.append(timing)
        merged = {
            key: mx.concatenate([p[key] for p in preds], axis=0)
            for key in ("coordinate", "plddt", "pae", "pde", "resolved")
        }
        merged["summary_confidence"] = sum((p["summary_confidence"] for p in preds), [])
        merged["full_data"] = sum((p["full_data"] for p in preds), [])
        return merged, {"time": timings}
