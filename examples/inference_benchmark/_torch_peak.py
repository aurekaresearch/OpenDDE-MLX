# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Run the upstream PyTorch entry point while sampling MPS memory.

Usage: python _torch_peak.py <opendde_torch_repo> <runner args...>
Prints ``PEAK_MEMORY_BYTES <n>`` on exit.
"""

import os
import sys
import threading

repo = sys.argv.pop(1)
here = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != here]
sys.path.insert(0, repo)

import torch  # noqa: E402

peak = 0
stop = threading.Event()


def _sample() -> None:
    global peak
    while not stop.wait(0.05):
        peak = max(peak, torch.mps.driver_allocated_memory())


sampler = threading.Thread(target=_sample, daemon=True)
sampler.start()
try:
    from runner.inference import run

    run()
finally:
    stop.set()
    sampler.join(timeout=1)
    print(f"PEAK_MEMORY_BYTES {peak}", flush=True)
