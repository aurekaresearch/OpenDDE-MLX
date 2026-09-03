# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Typed view over the resolved inference configuration tree."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict

InferenceDtype = Literal["bf16", "fp16", "fp32"]
INFERENCE_DTYPE_CHOICES: tuple[InferenceDtype, ...] = ("bf16", "fp16", "fp32")


class BaseConfig(BaseModel):
    """Pydantic base that keeps ``ml_collections.ConfigDict`` style access."""

    model_config = ConfigDict(extra="allow", validate_assignment=False)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return key in type(self).model_fields or key in (self.__pydantic_extra__ or {})

    def keys(self):
        return self.model_dump().keys()

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class WeightConfig(BaseConfig):
    alpha_shape_comp: float


class BinConfig(BaseConfig):
    min_bin: float
    max_bin: float
    no_bins: int
    eps: float


class PlddtConfig(BinConfig):
    normalize: bool


class ResolvedConfig(BaseConfig):
    eps: float


class ShapeCompConfig(BaseConfig):
    pair_weight: float
    token_weight: float
    global_weight: float
    density_sigma: float
    interface_cutoff: float
    gap_mean: float
    gap_scale: float
    clash_distance: float
    clash_scale: float
    pool_temperature: float
    normal_strength_min: float
    pair_chunk_size: Optional[int]
    debug_pair_map: bool
    eps: float


class ConfidenceConfig(BaseConfig):
    weight: WeightConfig
    plddt: PlddtConfig
    pde: BinConfig
    resolved: ResolvedConfig
    pae: BinConfig
    distogram: BinConfig
    shape_comp: ShapeCompConfig


class ClashConfig(BaseConfig):
    af3_clash_threshold: float
    vdw_clash_threshold: float


class MetricsConfig(BaseConfig):
    complex_ranker_keys: list[str]
    chain_ranker_keys: list[str]
    interface_ranker_keys: list[str]
    clash: ClashConfig


class MsaDataConfig(BaseConfig):
    enable_prot_msa: bool
    msa_pool_size: int
    msa_depth: int
    max_paired_per_species: int
    max_input_sequences: int


class TemplateDataConfig(BaseConfig):
    enable_prot_template: bool
    max_templates: int
    fetch_remote: bool
    prot_template_mmcif_dir: str
    prot_template_cache_dir: str
    kalign_binary_path: str
    release_dates_path: str
    obsolete_pdbs_path: str


class DataConfig(BaseConfig):
    msa: MsaDataConfig
    template: TemplateDataConfig
    ccd_components_file: str
    ccd_components_rdkit_mol_file: str


class InferSettingConfig(BaseConfig):
    chunk_size: Optional[int]
    dynamic_chunk_size: bool
    chunk_size_thresholds: dict[str, int]
    sample_diffusion_chunk_size: Optional[int]
    diffusion_bias_cache_gb: float


class NoiseSchedulerConfig(BaseConfig):
    s_max: float
    s_min: float
    rho: int
    sigma_data: float


class SampleDiffusionConfig(BaseConfig):
    gamma0: float
    gamma_min: float
    noise_scale_lambda: float
    step_scale_eta: float
    N_step: int
    N_sample: int


class SkipAmpConfig(BaseConfig):
    sample_diffusion: bool
    confidence_head: bool


class InputEmbedderConfig(BaseConfig):
    c_atom: int
    c_atompair: int
    c_token: int


class RelativePositionEncodingConfig(BaseConfig):
    r_max: int
    s_max: int
    c_z: int


class TemplateEmbedderConfig(BaseConfig):
    c: int
    c_z: int
    n_blocks: int
    hidden_scale_up: bool


class MsaModuleConfig(BaseConfig):
    c_m: int
    c_z: int
    c_s_inputs: int
    n_blocks: int
    hidden_scale_up: bool
    msa_chunk_size: Optional[int]


class PairformerConfig(BaseConfig):
    n_blocks: int
    c_z: int
    c_s: int
    n_heads: int
    hidden_scale_up: bool


class AttentionBlockConfig(BaseConfig):
    n_blocks: int
    n_heads: int


class StructuralRefinerConfig(BaseConfig):
    enable: bool
    n_blocks: int
    n_heads: int
    num_intermediate_factor: int
    hidden_scale_up: bool


class StructuralTokenExpansionConfig(BaseConfig):
    enable: bool
    n_roles: int
    pair_projection_mode: str
    pair_chunk_size: Optional[int]
    pair_output_space: str
    structural_refiner: StructuralRefinerConfig


class DiffusionModuleConfig(BaseConfig):
    sigma_data: float
    c_token: int
    c_atom: int
    c_atompair: int
    c_z: int
    c_z_pair_diffusion: int
    c_s: int
    c_s_inputs: int
    atom_encoder: AttentionBlockConfig
    transformer: AttentionBlockConfig
    atom_decoder: AttentionBlockConfig


class ConfidenceHeadConfig(BaseConfig):
    c_z: int
    c_s: int
    c_s_inputs: int
    n_blocks: int
    max_atoms_per_token: int
    hidden_scale_up: bool
    distance_bin_start: float
    distance_bin_end: float
    distance_bin_step: float


class DistogramHeadConfig(BaseConfig):
    c_z: int
    no_bins: int


class ModelConfig(BaseConfig):
    N_model_seed: int
    N_cycle: int
    input_embedder: InputEmbedderConfig
    relative_position_encoding: RelativePositionEncodingConfig
    template_embedder: TemplateEmbedderConfig
    msa_module: MsaModuleConfig
    pairformer: PairformerConfig
    structural_token_expansion: StructuralTokenExpansionConfig
    diffusion_module: DiffusionModuleConfig
    confidence_head: ConfidenceHeadConfig
    distogram_head: DistogramHeadConfig


class OpenDDEConfig(BaseConfig):
    load_checkpoint_path: str
    load_strict: bool
    c_s: int
    c_z: int
    c_s_inputs: int
    c_atom: int
    c_atompair: int
    c_token: int
    n_blocks: int
    max_atoms_per_token: int
    no_bins: int
    sigma_data: float
    hidden_scale_up: bool
    enable_diffusion_shared_vars_cache: bool
    enable_efficient_fusion: bool
    dtype: InferenceDtype
    skip_amp: SkipAmpConfig
    infer_setting: InferSettingConfig
    inference_noise_scheduler: NoiseSchedulerConfig
    sample_diffusion: SampleDiffusionConfig
    model: ModelConfig
    confidence: ConfidenceConfig
    metrics: MetricsConfig
    data: DataConfig
    model_name: str
    seeds: list[int]
    dump_dir: str
    need_atom_confidence: bool
    sorted_by_ranking_score: bool
    input_json_path: Optional[str]
    load_checkpoint_dir: str
    use_msa: bool
    msa_pair_as_unpair: bool
    use_template: bool
    use_rna_msa: bool
