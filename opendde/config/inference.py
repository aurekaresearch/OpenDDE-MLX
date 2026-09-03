# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Build and validate the inference configuration."""

import copy
from collections.abc import Mapping
from typing import Any, Optional

from opendde.config.config import parse_configs
from opendde.config.data import data_configs
from opendde.config.inference_defaults import inference_configs
from opendde.config.model_base import configs as configs_base
from opendde.config.model_registry import model_configs
from opendde.config.schema import OpenDDEConfig


def deep_update(configs: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively update nested config dictionaries in place."""
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(configs.get(key), Mapping):
            deep_update(configs[key], value)
        else:
            configs[key] = copy.deepcopy(value)
    return configs


def make_base_inference_config(model_name: Optional[str] = None) -> dict[str, Any]:
    """Return an isolated base inference config tree."""
    configs = {
        **copy.deepcopy(configs_base),
        "data": copy.deepcopy(data_configs),
        **copy.deepcopy(inference_configs),
    }
    if model_name is not None:
        configs["model_name"] = model_name
    return configs


def build_inference_config(
    arg_str: Optional[str] = None,
    model_name: Optional[str] = None,
    fill_required_with_null: bool = True,
) -> OpenDDEConfig:
    """Merge model defaults with dotted CLI overrides (``--model.N_cycle 4``)."""
    first_pass = parse_configs(
        configs=make_base_inference_config(model_name=model_name),
        arg_str=arg_str,
        fill_required_with_null=fill_required_with_null,
    )
    selected_model_name = first_pass.model_name
    if selected_model_name not in model_configs:
        supported = ", ".join(sorted(model_configs))
        raise ValueError(
            f"Unsupported model_name {selected_model_name!r}. Available models: {supported}."
        )
    base_configs = make_base_inference_config(model_name=model_name)
    deep_update(base_configs, model_configs[selected_model_name])
    merged = parse_configs(
        configs=base_configs,
        arg_str=arg_str,
        fill_required_with_null=fill_required_with_null,
    )
    return OpenDDEConfig.model_validate(merged.to_dict())


def validate_inference_schedule(configs: OpenDDEConfig) -> None:
    """Reject empty/negative inference schedules before model initialization."""
    for option, value in (
        ("model.N_cycle", configs.model.N_cycle),
        ("model.N_model_seed", configs.model.N_model_seed),
        ("sample_diffusion.N_step", configs.sample_diffusion.N_step),
        ("sample_diffusion.N_sample", configs.sample_diffusion.N_sample),
    ):
        if value < 1:
            raise ValueError(f"{option} must be at least 1, got {value}.")
    for option, value in (
        ("infer_setting.chunk_size", configs.infer_setting.chunk_size),
        (
            "infer_setting.sample_diffusion_chunk_size",
            configs.infer_setting.sample_diffusion_chunk_size,
        ),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{option} must be at least 1 or null, got {value}.")
    seen: set[int] = set()
    for raw_threshold, chunk_size in configs.infer_setting.chunk_size_thresholds.items():
        try:
            threshold = int(raw_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"infer_setting.chunk_size_thresholds keys must be positive integers; got {raw_threshold!r}."
            ) from exc
        if threshold < 1 or threshold in seen:
            raise ValueError(
                f"infer_setting.chunk_size_thresholds has an invalid or duplicate threshold {raw_threshold!r}."
            )
        seen.add(threshold)
        if chunk_size != -1 and chunk_size < 1:
            raise ValueError(
                f"infer_setting.chunk_size_thresholds values must be -1 or at least 1; got {chunk_size}."
            )
