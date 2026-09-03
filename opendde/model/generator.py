# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Noise schedule and the diffusion sampler (AF3 Algorithm 18)."""

from typing import Any, Callable, Optional

import mlx.core as mx
import numpy as np

from opendde.model.utils import centre_random_augmentation
from opendde.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceNoiseScheduler:
    """EDM noise levels ``sigma_data * (s_max^(1/rho) + t (s_min^(1/rho) - s_max^(1/rho)))^rho``."""

    def __init__(
        self, s_max: float = 160.0, s_min: float = 4e-4, rho: float = 7, sigma_data: float = 16.0
    ) -> None:
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.rho = rho

    def __call__(self, N_step: int = 200) -> np.ndarray:
        """Noise levels ``[N_step + 1]`` ending at 0."""
        t = np.arange(N_step + 1, dtype=np.float64) / N_step
        levels = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.rho)
                + t * (self.s_min ** (1 / self.rho) - self.s_max ** (1 / self.rho))
            )
            ** self.rho
        )
        levels[-1] = 0.0
        return levels.astype(np.float32)


def sample_diffusion(
    denoise_net: Callable[..., mx.array],
    feat: dict[str, Any],
    s_inputs: mx.array,
    s_trunk: mx.array,
    z_trunk: Optional[mx.array],
    pair_z: Optional[mx.array],
    p_lm: Optional[mx.array],
    c_l: Optional[mx.array],
    noise_schedule: np.ndarray,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: Optional[int] = None,
    enable_efficient_fusion: bool = False,
    pair_bias_cache: Optional[list[mx.array]] = None,
    seed: Optional[int] = None,
) -> mx.array:
    """Denoise from pure noise to coordinates ``[..., N_sample, N_atom, 3]``.

    All randomness (initial noise, augmentation, per-step noise) comes from one
    NumPy generator seeded with ``seed`` so rollouts are reproducible.
    """
    N_atom = feat["atom_to_token_idx"].shape[-1]
    batch_shape = s_inputs.shape[:-2]
    rng = np.random.default_rng(seed)
    num_steps = len(noise_schedule) - 1

    def _normal(shape: tuple[int, ...]) -> mx.array:
        return mx.array(rng.standard_normal(shape).astype(np.float32))

    def _rollout(chunk_n_sample: int) -> mx.array:
        x_l = float(noise_schedule[0]) * _normal((*batch_shape, chunk_n_sample, N_atom, 3))
        for step_i in range(num_steps):
            c_tau_last, c_tau = float(noise_schedule[step_i]), float(noise_schedule[step_i + 1])
            x_l = centre_random_augmentation(x_l, rng, N_sample=1)[..., 0, :, :]
            gamma = gamma0 if c_tau > gamma_min else 0.0
            t_hat = c_tau_last * (gamma + 1)
            delta_noise_level = float(np.sqrt(t_hat**2 - c_tau_last**2))
            x_noisy = x_l + noise_scale_lambda * delta_noise_level * _normal(x_l.shape)
            t_hat_levels = mx.full((*batch_shape, chunk_n_sample), t_hat, dtype=mx.float32)
            x_denoised = denoise_net(
                x_noisy=x_noisy,
                t_hat_noise_level=t_hat_levels,
                feat=feat,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                pair_z=pair_z,
                p_lm=p_lm,
                c_l=c_l,
                enable_efficient_fusion=enable_efficient_fusion,
                pair_bias_cache=pair_bias_cache,
            )
            # Euler step; AF3 line 9 uses x_l_hat in the delta, which we believe is a typo.
            x_l = x_noisy + step_scale_eta * (c_tau - t_hat) * (x_noisy - x_denoised) / t_hat
            mx.eval(x_l)
        return x_l

    if diffusion_chunk_size is None or diffusion_chunk_size >= N_sample:
        return _rollout(N_sample)
    chunks = []
    for start in range(0, N_sample, diffusion_chunk_size):
        chunks.append(_rollout(min(diffusion_chunk_size, N_sample - start)))
    return mx.concatenate(chunks, axis=-3)
