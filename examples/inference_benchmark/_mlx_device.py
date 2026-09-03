# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Run the MLX entry point pinned to one device.

Usage: python _mlx_device.py <cpu|gpu> <runner args...>
"""

import os
import sys

device = sys.argv.pop(1)
here = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != here]

import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)

from runner.inference import run  # noqa: E402

run()
