# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Diffusion conditioning and the denoising network (AF3 Algorithms 20, 21)."""

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from opendde.model.embedders import FourierEmbedding, RelativePositionEncoding
from opendde.model.primitives import LayerNorm, LinearNoBias, Transition
from opendde.model.transformer import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    DiffusionTransformer,
)


class DiffusionConditioning(nn.Module):
    """Algorithm 21: build the single/pair conditioning for one noise level."""

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_z: int = 128,
        c_z_pair_diffusion: Optional[int] = None,
        c_s: int = 384,
        c_s_inputs: int = 449,
        c_noise_embedding: int = 256,
    ) -> None:
        super().__init__()
        self.sigma_data = sigma_data
        self.c_z_pair_diffusion = c_z if c_z_pair_diffusion is None else c_z_pair_diffusion
        self.compress_pair_z = self.c_z_pair_diffusion != c_z
        c_zp = self.c_z_pair_diffusion
        self.relpe = RelativePositionEncoding(c_z=c_zp)
        if self.compress_pair_z:
            self.layernorm_z_trunk = LayerNorm(c_z, create_offset=False)
            self.linear_no_bias_z_trunk = LinearNoBias(c_z, c_zp, precision=True)
        self.layernorm_z = LayerNorm(2 * c_zp, create_offset=False)
        self.linear_no_bias_z = LinearNoBias(2 * c_zp, c_zp, precision=True)
        self.transition_z1 = Transition(c_zp, n=2)
        self.transition_z2 = Transition(c_zp, n=2)
        self.layernorm_s = LayerNorm(c_s + c_s_inputs, create_offset=False)
        self.linear_no_bias_s = LinearNoBias(c_s + c_s_inputs, c_s, precision=True)
        self.fourier_embedding = FourierEmbedding(c=c_noise_embedding)
        self.layernorm_n = LayerNorm(c_noise_embedding, create_offset=False)
        self.linear_no_bias_n = LinearNoBias(c_noise_embedding, c_s, precision=True)
        self.transition_s1 = Transition(c_s, n=2)
        self.transition_s2 = Transition(c_s, n=2)

    def prepare_cache(self, feat: dict[str, Any], z_trunk: mx.array) -> mx.array:
        """Pair conditioning ``[..., N, N, c_z_pair_diffusion]`` (constant during sampling)."""
        z_pair_trunk = z_trunk
        if self.compress_pair_z:
            z_pair_trunk = self.linear_no_bias_z_trunk(self.layernorm_z_trunk(z_trunk))
        pair_z = mx.concatenate(
            [z_pair_trunk, self.relpe(feat).astype(z_pair_trunk.dtype)], axis=-1
        )
        pair_z = self.linear_no_bias_z(self.layernorm_z(pair_z))
        pair_z = pair_z + self.transition_z1(pair_z)
        pair_z = pair_z + self.transition_z2(pair_z)
        mx.eval(pair_z)
        return pair_z

    def __call__(
        self,
        t_hat_noise_level: mx.array,
        feat: dict[str, Any],
        s_inputs: mx.array,
        s_trunk: mx.array,
        z_trunk: Optional[mx.array],
        pair_z: Optional[mx.array],
    ) -> tuple[mx.array, mx.array]:
        """Returns single conditioning ``[..., N_sample, N, c_s]`` and pair conditioning."""
        if pair_z is None:
            pair_z = self.prepare_cache(feat, z_trunk)
        single_s = self.linear_no_bias_s(
            self.layernorm_s(mx.concatenate([s_trunk, s_inputs], axis=-1))
        )
        noise_ratio = mx.maximum(t_hat_noise_level / self.sigma_data, 1e-10)
        noise_n = self.fourier_embedding(mx.log(noise_ratio) / 4).astype(single_s.dtype)
        single_s = (
            single_s[..., None, :, :]
            + self.linear_no_bias_n(self.layernorm_n(noise_n))[..., None, :]
        )
        single_s = single_s + self.transition_s1(single_s)
        single_s = single_s + self.transition_s2(single_s)
        return single_s, pair_z


class DiffusionModule(nn.Module):
    """Algorithm 20: EDM-style denoiser built from atom encoder, token transformer, atom decoder."""

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        c_z_pair_diffusion: Optional[int] = None,
        c_s_inputs: int = 449,
        atom_encoder: dict[str, int] = {"n_blocks": 3, "n_heads": 4},
        transformer: dict[str, Any] = {"n_blocks": 24, "n_heads": 16},
        atom_decoder: dict[str, int] = {"n_blocks": 3, "n_heads": 4},
    ) -> None:
        super().__init__()
        self.sigma_data = sigma_data
        c_zp = c_z if c_z_pair_diffusion is None else c_z_pair_diffusion
        self.diffusion_conditioning = DiffusionConditioning(
            sigma_data=sigma_data, c_z=c_z, c_z_pair_diffusion=c_zp, c_s=c_s, c_s_inputs=c_s_inputs
        )
        self.atom_attention_encoder = AtomAttentionEncoder(
            **atom_encoder,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            has_coords=True,
            c_s=c_s,
            c_z=c_zp,
        )
        self.layernorm_s = LayerNorm(c_s, create_offset=False)
        self.linear_no_bias_s = LinearNoBias(c_s, c_token, precision=True, initializer="zeros")
        self.diffusion_transformer = DiffusionTransformer(
            **transformer, c_a=c_token, c_s=c_s, c_z=c_zp
        )
        self.layernorm_a = LayerNorm(c_token, create_offset=False)
        self.atom_attention_decoder = AtomAttentionDecoder(
            **atom_decoder, c_token=c_token, c_atom=c_atom, c_atompair=c_atompair
        )
        self.normalize = LayerNorm(c_zp, create_offset=False, create_scale=False)

    def _conditioned_atom_cache(
        self, p_lm: mx.array, c_l: mx.array, s_trunk: mx.array, atom_to_token_idx: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Step-invariant atom conditioning, memoised for the current rollout inputs."""
        key = (p_lm, c_l, s_trunk)
        cached = getattr(self, "_atom_cache", None)
        if cached is None or any(a is not b for a, b in zip(cached[0], key)):
            values = self.atom_attention_encoder.condition(
                p_lm, c_l, s_trunk[..., None, :, :], atom_to_token_idx
            )
            mx.eval(values)
            self._atom_cache = (key, values)
            cached = self._atom_cache
        return cached[1]

    def prepare_pair_bias_cache(
        self, pair_z: mx.array, extra_attn_bias: Optional[mx.array], enable_efficient_fusion: bool
    ) -> list[mx.array]:
        """Per-block token pair biases, computed once per rollout instead of once per step."""
        z = pair_z.astype(mx.float32)
        if enable_efficient_fusion:
            z = self.normalize(z)
        return self.diffusion_transformer.pair_bias_cache(
            z, enable_efficient_fusion, extra_attn_bias
        )

    def f_forward(
        self,
        r_noisy: mx.array,
        t_hat_noise_level: mx.array,
        feat: dict[str, Any],
        s_inputs: mx.array,
        s_trunk: mx.array,
        z_trunk: Optional[mx.array],
        pair_z: Optional[mx.array],
        p_lm: Optional[mx.array],
        c_l: Optional[mx.array],
        enable_efficient_fusion: bool = False,
        pair_bias_cache: Optional[list[mx.array]] = None,
    ) -> mx.array:
        """``F_theta(c_in * x, c_noise(sigma))``: predict the coordinate update ``[..., N_sample, N_atom, 3]``."""
        s_single, z_pair = self.diffusion_conditioning(
            t_hat_noise_level,
            feat,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=pair_z,
        )
        conditioned = p_lm is not None and c_l is not None
        if conditioned:
            p_lm, c_l = self._conditioned_atom_cache(p_lm, c_l, s_trunk, feat["atom_to_token_idx"])
        a_token, q_skip, c_skip, p_skip = self.atom_attention_encoder(
            feat["atom_to_token_idx"],
            feat["ref_pos"],
            feat["ref_charge"],
            feat["ref_mask"],
            feat["ref_atom_name_chars"],
            feat["ref_element"],
            feat["d_lm"],
            feat["v_lm"],
            feat["pad_info"],
            r_l=r_noisy,
            s=s_trunk[..., None, :, :],
            z=z_pair,
            p_lm=p_lm,
            c_l=c_l,
            conditioned=conditioned,
        )
        dtype = self.layernorm_a.weight.dtype
        a_token = a_token.astype(dtype) + self.linear_no_bias_s(self.layernorm_s(s_single)).astype(
            dtype
        )
        z = z_pair.astype(dtype)
        if enable_efficient_fusion and pair_bias_cache is None:
            z = self.normalize(z)
        a_token = self.diffusion_transformer(
            a=a_token,
            s=s_single.astype(dtype),
            z=z,
            extra_attn_bias=feat.get("structural_pair_attn_bias"),
            z_is_normalized=enable_efficient_fusion,
            pair_bias_cache=pair_bias_cache,
        )
        a_token = self.layernorm_a(a_token)
        return self.atom_attention_decoder(
            feat["atom_to_token_idx"], a_token, q_skip, c_skip, p_skip
        )

    def __call__(
        self,
        x_noisy: mx.array,
        t_hat_noise_level: mx.array,
        feat: dict[str, Any],
        s_inputs: mx.array,
        s_trunk: mx.array,
        z_trunk: Optional[mx.array],
        pair_z: Optional[mx.array],
        p_lm: Optional[mx.array],
        c_l: Optional[mx.array],
        enable_efficient_fusion: bool = False,
        pair_bias_cache: Optional[list[mx.array]] = None,
    ) -> mx.array:
        """One denoising step: ``x_noisy [..., N_sample, N_atom, 3]`` -> ``x_denoised``."""
        sigma = t_hat_noise_level[..., None, None]
        r_noisy = x_noisy / mx.sqrt(self.sigma_data**2 + sigma**2)
        r_update = self.f_forward(
            r_noisy,
            t_hat_noise_level,
            feat,
            s_inputs,
            s_trunk,
            z_trunk,
            pair_z,
            p_lm,
            c_l,
            enable_efficient_fusion=enable_efficient_fusion,
            pair_bias_cache=pair_bias_cache,
        )
        s_ratio = sigma / self.sigma_data
        return x_noisy / (1 + s_ratio**2) + sigma / mx.sqrt(1 + s_ratio**2) * r_update
