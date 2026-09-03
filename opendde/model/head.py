# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Distogram head (AF3 Algorithm 1, line 17)."""

import mlx.core as mx
import mlx.nn as nn

from opendde.model.primitives import Linear


class DistogramHead(nn.Module):
    """Symmetrised pair logits over ``no_bins`` distance bins."""

    def __init__(self, c_z: int = 128, no_bins: int = 64) -> None:
        super().__init__()
        self.c_z = c_z
        self.no_bins = no_bins
        self.linear = Linear(c_z, no_bins, initializer="zeros")

    def __call__(self, z: mx.array) -> mx.array:
        """``z [*, N, N, c_z] -> logits [*, N, N, no_bins]``."""
        logits = self.linear(z)
        return logits + logits.swapaxes(-2, -3)
