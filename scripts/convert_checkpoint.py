#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Convert a released OpenDDE ``.pt`` checkpoint into ``.safetensors`` for MLX.

Usage:
    python scripts/convert_checkpoint.py ~/.cache/opendde/checkpoint/opendde.pt
"""

import argparse
import logging
import os

from opendde.model.checkpoint import convert_torch_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pt_path", help="PyTorch checkpoint (.pt)")
    parser.add_argument("-o", "--output", help="Output .safetensors path (default: next to input)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    output = args.output or os.path.splitext(args.pt_path)[0] + ".safetensors"
    convert_torch_checkpoint(args.pt_path, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
