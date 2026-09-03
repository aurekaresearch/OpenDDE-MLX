#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Reproducible MLX vs PyTorch-MPS inference benchmark.

    python examples/inference_benchmark/run_benchmark.py --out results/mymac.json
    python examples/inference_benchmark/run_benchmark.py --torch-repo ~/src/OpenDDE

Each case is one protein chain, no MSA, so the only variable is the engine.
"""

import argparse
import glob
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SEED = 42
SCHEDULE = {
    "cycle": "model.N_cycle",
    "step": "sample_diffusion.N_step",
    "sample": "sample_diffusion.N_sample",
}
MLX_TIME = re.compile(r"model_forward ([\d.]+)s; peak memory ([\d.]+) GB")
TORCH_TIME = re.compile(r"Model forward time: ([\d.]+)s")
TORCH_PEAK = re.compile(r"PEAK_MEMORY_BYTES (\d+)")


def machine() -> str:
    """Chip name plus installed memory, e.g. ``Apple M3 Pro, 36 GB``."""
    chip = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
    ).stdout.strip()
    total = int(
        subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout
    )
    return f"{chip}, {round(total / 1024**3)} GB"


def confidence(dump_dir: str) -> dict[str, float]:
    """Read the summary confidence of the top-ranked sample."""
    matches = glob.glob(f"{dump_dir}/*/seed_*/predictions/*summary_confidence*.json")
    if not matches:
        return {}
    data = json.load(open(sorted(matches)[0], encoding="utf-8"))
    return {k: round(float(data[k]), 4) for k in ("plddt", "ptm", "ranking_score") if k in data}


def run_case(backend: str, dtype: str, length: int, args: argparse.Namespace) -> dict:
    """Run one (backend, dtype, length) cell and return its measurements."""
    job = f"{backend}_{length}_{dtype}"
    dump_dir = os.path.join(args.work_dir, job)
    schedule = [x for k, key in SCHEDULE.items() for x in (f"--{key}", str(getattr(args, k)))]
    runner = [
        "--input_json_path",
        f"{HERE}/inputs/bench_{length}.json",
        "--dump_dir",
        dump_dir,
        "--dtype",
        dtype,
        "--use_msa",
        "false",
        *schedule,
    ]
    engine, _, device = backend.partition("-")
    device = device or "gpu"
    if engine == "mlx":
        cmd = [
            sys.executable,
            f"{HERE}/_mlx_device.py",
            device,
            *runner,
            "--skip_amp.sample_diffusion",
            "false",
            "--skip_amp.confidence_head",
            "false",
        ]
        cwd = REPO
    else:
        cmd = [
            args.torch_python,
            f"{HERE}/_torch_peak.py",
            args.torch_repo,
            *runner,
            "--device",
            "mps" if device == "gpu" else "cpu",
        ]
        cwd = args.torch_repo

    print(f"  {job} ...", end="", flush=True)
    started = time.time()
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    peak_rss = 0.0
    watcher = threading.Thread(target=lambda: None)
    if device != "gpu":
        done = threading.Event()

        def _watch() -> None:
            nonlocal peak_rss
            while not done.wait(0.2):
                rss = subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(proc.pid)], capture_output=True, text=True
                ).stdout.strip()
                if rss.isdigit():
                    peak_rss = max(peak_rss, float(rss))

        watcher = threading.Thread(target=_watch, daemon=True)
        watcher.start()
    output = proc.communicate()[0]
    if device != "gpu":
        done.set()
        watcher.join(timeout=1)
    result: dict = {"backend": backend, "dtype": dtype, "n_token": length}

    pattern = MLX_TIME if engine == "mlx" else TORCH_TIME
    found = pattern.search(output)
    if not found:
        reason = next(
            (ln for ln in reversed(output.splitlines()) if "Error" in ln or "error" in ln), ""
        )
        print(f" FAILED ({reason[:90]})")
        return {**result, "status": "failed", "reason": reason[:300]}

    result["forward_s"] = round(float(found.group(1)), 1)
    result["memory_kind"] = "gpu" if device == "gpu" else "rss"
    if device != "gpu":
        result["peak_gb"] = round(peak_rss / 1e6, 1)
    elif engine == "mlx":
        result["peak_gb"] = round(float(found.group(2)), 1)
    elif peak := TORCH_PEAK.search(output):
        result["peak_gb"] = round(int(peak.group(1)) / 1e9, 1)
    result["wall_s"] = round(time.time() - started, 1)
    result.update(confidence(dump_dir), status="ok")
    print(
        f" {result['forward_s']}s, {result.get('peak_gb', '?')} GB, "
        f"pLDDT {result.get('plddt', '?')}"
    )
    return result


def table(results: list[dict]) -> str:
    """Render the results as a markdown table."""
    lengths = sorted({r["n_token"] for r in results})
    cells = {(r["backend"], r["dtype"], r["n_token"]): r for r in results}
    combos = sorted({(r["backend"], r["dtype"]) for r in results}, reverse=True)
    header = " | ".join(f"{b} {d}" for b, d in combos)
    lines = [f"| Residues | {header} |", "| ---: |" + " ---: |" * len(combos)]
    for n in lengths:
        row = []
        for combo in combos:
            cell = cells.get((*combo, n))
            row.append(
                "failed"
                if not cell or cell["status"] != "ok"
                else f"{cell['forward_s']} s / {cell.get('peak_gb', '?')} GB / {cell.get('plddt', 0):.1f}"
            )
        lines.append(f"| {n} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", default="100,200,300")
    parser.add_argument("--dtypes", default="fp32,bf16")
    parser.add_argument("--torch-repo", help="Upstream PyTorch OpenDDE checkout (enables MPS runs)")
    parser.add_argument("--torch-python", help="Interpreter for the MPS runs (default: its .venv)")
    parser.add_argument("--backends", default="mlx,torch", help="mlx, torch, mlx-cpu, torch-cpu")
    parser.add_argument("--cycle", type=int, default=4)
    parser.add_argument("--step", type=int, default=30)
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--work-dir", default="/tmp/opendde_benchmark")
    parser.add_argument("--out", default=os.path.join(HERE, "results", "local.json"))
    args = parser.parse_args()
    args.torch_repo = (
        os.path.abspath(os.path.expanduser(args.torch_repo)) if args.torch_repo else None
    )

    if args.torch_repo and not args.torch_python:
        venv = os.path.join(args.torch_repo, ".venv", "bin", "python")
        args.torch_python = venv if os.path.exists(venv) else sys.executable
    backends = [b for b in args.backends.split(",") if b.startswith("mlx") or args.torch_repo]
    print(f"Machine: {machine()}\nPython: {platform.python_version()}")
    results = []
    for length in [int(x) for x in args.lengths.split(",")]:
        for backend in backends:
            for dtype in args.dtypes.split(","):
                results.append(run_case(backend, dtype, length, args))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    settings = f"cycle={args.cycle} step={args.step} sample={args.sample} use_msa=false"
    payload = {"machine": machine(), "seed": SEED, "settings": settings, "results": results}
    json.dump(payload, open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\nforward time / peak memory / pLDDT\n\n{table(results)}\n\nSaved {args.out}")


if __name__ == "__main__":
    main()
