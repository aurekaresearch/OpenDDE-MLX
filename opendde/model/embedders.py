# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Input embedders (AF3 Algorithms 2, 3 and 22)."""

import math
from typing import Any, Union

import mlx.core as mx
import mlx.nn as nn

from opendde.model.primitives import LinearNoBias
from opendde.model.transformer import AtomAttentionEncoder


def _one_hot(x: mx.array, num_classes: int) -> mx.array:
    return (x[..., None] == mx.arange(num_classes)).astype(mx.float32)


class InputFeatureEmbedder(nn.Module):
    """Algorithm 2: atom encoder output concatenated with per-token input features."""

    def __init__(self, c_atom: int = 128, c_atompair: int = 16, c_token: int = 384) -> None:
        super().__init__()
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.atom_attention_encoder = AtomAttentionEncoder(
            c_atom=c_atom, c_atompair=c_atompair, c_token=c_token, has_coords=False
        )
        self.input_feature = {"restype": 32, "profile": 32, "deletion_mean": 1}

    def __call__(self, feat: dict[str, Any]) -> mx.array:
        """Returns ``s_inputs [..., N_token, c_token + 65]``."""
        a, _, _, _ = self.atom_attention_encoder(
            feat["atom_to_token_idx"],
            feat["ref_pos"],
            feat["ref_charge"],
            feat["ref_mask"],
            feat["ref_atom_name_chars"],
            feat["ref_element"],
            feat["d_lm"],
            feat["v_lm"],
            feat["pad_info"],
        )
        batch_shape = feat["restype"].shape[:-1]
        per_token = [
            feat[name].reshape(*batch_shape, d).astype(a.dtype)
            for name, d in self.input_feature.items()
        ]
        return mx.concatenate([a, *per_token], axis=-1)


def relative_position_features(feat: dict[str, Any], r_max: int, s_max: int) -> mx.array:
    """Algorithm 3 lines 1-10: one-hot relative position features ``[..., N, N, 4r+2s+7]``."""
    asym_id = feat["asym_id"]
    residue_index = feat["residue_index"]
    entity_id = feat["entity_id"]
    token_index = feat["token_index"]
    sym_id = feat["sym_id"]

    same_chain = (asym_id[..., :, None] == asym_id[..., None, :]).astype(mx.int32)
    same_residue = (residue_index[..., :, None] == residue_index[..., None, :]).astype(mx.int32)
    same_entity = (entity_id[..., :, None] == entity_id[..., None, :]).astype(mx.int32)

    def clipped(idx: mx.array, gate: mx.array, vmax: int) -> mx.array:
        d = mx.clip(idx[..., :, None] - idx[..., None, :] + vmax, 0, 2 * vmax)
        return d * gate + (1 - gate) * (2 * vmax + 1)

    a_rel_pos = _one_hot(clipped(residue_index, same_chain, r_max), 2 * (r_max + 1))
    a_rel_token = _one_hot(clipped(token_index, same_chain * same_residue, r_max), 2 * (r_max + 1))
    a_rel_chain = _one_hot(clipped(sym_id, same_entity, s_max), 2 * (s_max + 1))
    return mx.concatenate(
        [a_rel_pos, a_rel_token, same_entity[..., None].astype(mx.float32), a_rel_chain], axis=-1
    )


class RelativePositionEncoding(nn.Module):
    """Algorithm 3: linear projection of the relative position features."""

    def __init__(self, r_max: int = 32, s_max: int = 2, c_z: int = 128) -> None:
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z
        self.linear_no_bias = LinearNoBias(4 * r_max + 2 * s_max + 7, c_z)

    def __call__(self, feat_or_relp: Union[dict[str, Any], mx.array]) -> mx.array:
        """Accepts the feature dict or a prebuilt ``relp``; returns ``[..., N, N, c_z]``."""
        relp = feat_or_relp
        if isinstance(relp, dict):
            relp = relative_position_features(relp, self.r_max, self.s_max)
        return self.linear_no_bias(relp)


class FourierEmbedding(nn.Module):
    """Algorithm 22: ``cos(2 pi (t w + b))`` with fixed random ``w``, ``b``."""

    def __init__(self, c: int, seed: int = 42) -> None:
        super().__init__()
        self.c = c
        key_w, key_b = mx.random.split(mx.random.key(seed))
        self.w = mx.random.normal((c,), key=key_w)
        self.b = mx.random.normal((c,), key=key_b)

    def __call__(self, t_hat_noise_level: mx.array) -> mx.array:
        """``[..., N_sample] -> [..., N_sample, c]``."""
        return mx.cos(2 * math.pi * (t_hat_noise_level[..., None] * self.w + self.b))
