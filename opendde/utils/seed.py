# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import random

import mlx.core as mx
import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and MLX random state."""
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)
