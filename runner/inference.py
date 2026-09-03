# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Single-machine inference runner."""

import json
import logging
import os
import random
import shutil
import time
import traceback
from typing import Any, Optional

import mlx.core as mx
import numpy as np

from opendde.config.config import parse_sys_args
from opendde.config.inference import build_inference_config, validate_inference_schedule
from opendde.config.schema import OpenDDEConfig
from opendde.data.inference.infer_dataloader import InferenceDataset
from opendde.data.inference.input_validation import validate_inference_jobs, validate_inference_seed
from opendde.model.checkpoint import count_parameters, load_checkpoint
from opendde.model.opendde import OpenDDE
from opendde.utils.download import download_inference_cache, resolve_checkpoint_path
from opendde.utils.logging_config import init_logging
from opendde.utils.seed import seed_everything
from runner.dumper import DataDumper

logger = logging.getLogger(__name__)
_DTYPES = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}


def to_mlx(features: dict[str, Any]) -> dict[str, Any]:
    """Convert a NumPy feature dict to ``mx.array`` values (int64/float32 preserved)."""
    return {k: mx.array(v) if isinstance(v, np.ndarray) else v for k, v in features.items()}


class InferenceRunner:
    """Load the model once and predict every job/seed pair."""

    def __init__(self, configs: OpenDDEConfig) -> None:
        validate_inference_schedule(configs)
        self.configs = configs
        download_inference_cache(configs)
        self.dump_dir = configs.dump_dir
        self.error_dir = os.path.join(self.dump_dir, "ERR")
        os.makedirs(self.dump_dir, exist_ok=True)
        shutil.rmtree(self.error_dir, ignore_errors=True)
        if mx.default_device() == mx.gpu:
            # keep GPU buffers resident and the allocator cache bounded
            mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
            mx.set_cache_limit(4 << 30)
        self.init_model()
        self.dumper = DataDumper(
            base_dir=self.dump_dir,
            need_atom_confidence=configs.need_atom_confidence,
            sorted_by_ranking_score=configs.sorted_by_ranking_score,
        )

    def init_model(self) -> None:
        checkpoint_path = resolve_checkpoint_path(self.configs)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Loading checkpoint %s (strict=%s)", checkpoint_path, self.configs.load_strict)
        t0 = time.time()
        self.model = OpenDDE(self.configs)
        load_checkpoint(self.model, checkpoint_path, strict=self.configs.load_strict)
        self.model.set_compute_dtype(
            _DTYPES[self.configs.dtype],
            fp32_diffusion=self.configs.skip_amp.sample_diffusion,
            fp32_confidence=self.configs.skip_amp.confidence_head,
        )
        mx.eval(self.model.parameters())
        logger.info(
            "Model ready: %.1fM parameters, dtype=%s, %.1fs",
            count_parameters(self.model) / 1e6,
            self.configs.dtype,
            time.time() - t0,
        )

    def predict(self, data: dict[str, Any], seed: int) -> dict[str, Any]:
        seed_everything(seed)
        mx.reset_peak_memory()
        prediction, log = self.model(to_mlx(data["input_feature_dict"]), seed=seed)
        times = log["time"] if isinstance(log["time"], dict) else log["time"][-1]
        timing = ", ".join(f"{k} {v:.1f}s" for k, v in times.items())
        logger.info("Stages: %s; peak memory %.1f GB", timing, mx.get_peak_memory() / 1e9)
        return prediction

    def close(self) -> None:
        self.model = None
        mx.clear_cache()


def resolve_job_seeds(
    json_data: list[dict[str, Any]], cli_seeds: Optional[list[int]]
) -> list[list[int]]:
    """Per job: command line > JSON ``modelSeeds`` > one random seed."""
    schedule = []
    for job in json_data:
        configured = cli_seeds if cli_seeds else job.get("modelSeeds")
        seeds = [
            validate_inference_seed(s, location=f"seed for job {job.get('name', '<unnamed>')!r}")
            for s in (configured or [])
        ]
        schedule.append(seeds or [random.randint(1, 65536)])
    return schedule


def infer_predict(runner: InferenceRunner, configs: OpenDDEConfig) -> None:
    """Featurise every job in ``configs.input_json_path`` and predict it for every seed."""
    logger.info("Loading data from %s", configs.input_json_path)
    with open(configs.input_json_path, "r", encoding="utf-8") as handle:
        json_data = validate_inference_jobs(json.load(handle))
    cli_seeds = [int(s) for s in configs.seeds] if configs.seeds else None
    job_seeds = resolve_job_seeds(json_data, cli_seeds)
    logger.info("Seed schedule: %s", job_seeds)

    dataset = InferenceDataset(configs=configs, inputs=json_data)
    errors: list[str] = []
    t_start = time.time()
    for index in range(len(dataset)):
        seed_everything(job_seeds[index][0])
        data, atom_array, error_message = dataset[index]
        sample_name = str(data["sample_name"])
        if error_message:
            logger.error("Data error for %s: %s", sample_name, error_message)
            _write_error(runner.error_dir, sample_name, error_message)
            errors.append(f"{sample_name}: {error_message}")
            continue
        logger.info(
            "[%d/%d] %s: N_asym %d, N_token %d, N_atom %d, N_msa %d",
            index + 1,
            len(dataset),
            sample_name,
            data["N_asym"],
            data["N_token"],
            data["N_atom"],
            data["N_msa"],
        )
        for seed in job_seeds[index]:
            t0 = time.time()
            try:
                prediction = runner.predict(data, seed)
                runner.dumper.dump(
                    group_name="",
                    pdb_id=sample_name,
                    seed=seed,
                    pred_dict=prediction,
                    atom_array=atom_array,
                    entity_poly_type={
                        k: v for k, v in data["entity_poly_type"].items() if v != "non-polymer"
                    },
                )
                logger.info(
                    "%s [seed:%d] done in %.1fs. Results saved to %s",
                    sample_name,
                    seed,
                    time.time() - t0,
                    configs.dump_dir,
                )
            except Exception as exc:
                message = f"{sample_name} [seed:{seed}] failed: {exc}\n{traceback.format_exc()}"
                logger.error(message)
                _write_error(runner.error_dir, sample_name, message)
                errors.append(message)
            finally:
                mx.clear_cache()
    logger.info("Job completed in %.1fs.", time.time() - t_start)
    if errors:
        raise RuntimeError(f"{len(errors)} inference sample(s) failed. First error:\n{errors[0]}")


def _write_error(error_dir: str, sample_name: str, message: str) -> None:
    os.makedirs(error_dir, exist_ok=True)
    with open(os.path.join(error_dir, f"{sample_name}.txt"), "a", encoding="utf-8") as handle:
        handle.write(message)


def main(configs: OpenDDEConfig) -> None:
    runner = InferenceRunner(configs)
    try:
        infer_predict(runner, runner.configs)
    finally:
        runner.close()


def run() -> None:
    """``python -m runner.inference --input_json_path ... --model.N_cycle 4``"""
    init_logging()
    try:
        arg_str = parse_sys_args()
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None
    configs = build_inference_config(arg_str=arg_str, fill_required_with_null=True)
    logger.info(
        "Inference by OpenDDE-MLX: model %s, cycle=%d, step=%d, dtype=%s",
        configs.model_name,
        configs.model.N_cycle,
        configs.sample_diffusion.N_step,
        configs.dtype,
    )
    main(configs)


if __name__ == "__main__":
    run()
