# OpenDDE-MLX

![Status](https://img.shields.io/badge/status-preview-orange)
![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
[![PyPI](https://img.shields.io/pypi/v/opendde-mlx)](https://pypi.org/project/opendde-mlx/)

OpenDDE-MLX is a rewrite of [OpenDDE](https://github.com/aurekaresearch/OpenDDE),
the open-source all-atom biomolecular co-folding model, on top of
[MLX](https://github.com/ml-explore/mlx). It runs the released
checkpoint natively on Apple Silicon with unified memory, without PyTorch,
CUDA, cuEquivariance, or any multi-device machinery.

- **Apple Silicon only.** Metal GPU through MLX; no fallbacks for other platforms.
- **Single machine.** The Fold-CP context-parallel path and all distributed code are gone.
- **Same weights, same outputs.** Every module is verified against the PyTorch
  implementation with numerical parity tests; the released `.pt` checkpoint is
  converted once to `.safetensors`.
- **Faster, and far smaller.** 1.5-2.1x the speed of upstream on the same GPU,
  4-7x the CPU path, in up to 5.5x less memory. See [Benchmarks](#benchmarks).


> [!IMPORTANT]
> OpenDDE-MLX is a preview. The model, CLI flags, and JSON fields follow the
> upstream 1.1.x release.


## Installation

Apple Silicon macOS with CPython `3.11`, `3.12` or `3.13`. We recommend
[`uv`](https://docs.astral.sh/uv/getting-started/installation/). Choose one of
the following methods.

### Install from PyPI

```bash
uv venv --python 3.12
uv pip install --python .venv opendde-mlx
```

Converting a released `.pt` checkpoint additionally needs PyTorch (CPU):

```bash
uv pip install --python .venv "opendde-mlx[convert]"
```

### Install from source

```bash
git clone git@github.com:aurekaresearch/OpenDDE-MLX.git
cd OpenDDE-MLX
uv venv --python 3.12
uv pip install --python .venv -e .
```

After either installation, verify the environment with:

```bash
uv run --no-project --python .venv opendde-mlx doctor
```

## Model and runtime data

Checkpoints and runtime assets live under `OPENDDE_ROOT_DIR`
(default `~/.cache/opendde`):

```text
$OPENDDE_ROOT_DIR/
├── checkpoint/opendde.safetensors   # converted weights (or opendde.pt)
├── common/                          # CCD components (downloaded automatically)
└── search_database/                 # only for local template / RNA-MSA search
```

`opendde-mlx pred` downloads the CCD assets and the released `opendde.pt` on first
use and converts it to `opendde.safetensors` (the conversion needs the
`convert` extra). To convert an existing checkpoint explicitly:

```bash
opendde-mlx convert ~/.cache/opendde/checkpoint/opendde.pt
```

Released checkpoints (same as upstream):

| Checkpoint        | Use case                     |
| ----------------- | ---------------------------- |
| `opendde.pt`      | General-purpose checkpoint   |
| `opendde_abag.pt` | Antibody-antigen tuned       |

## Quick start

```bash
opendde-mlx pred \
  -i examples/example_without_msa.json \
  -o ./output \
  --use_msa false
```

With MSA (uses the public ColabFold MMseqs2 server), an explicit dtype and a
lighter schedule:

```bash
opendde-mlx pred \
  -i examples/example.json \
  -o ./output \
  --dtype bf16 \
  --cycle 4 \
  --step 50 \
  --sample 2
```

Outputs are written to `output/<name>/seed_<seed>/predictions/`:
`<name>_sample_<rank>.cif`, `<name>_summary_confidence_sample_<rank>.json`,
and `<name>_full_data_sample_<rank>.json`.

The CLI mirrors upstream OpenDDE — same commands (`pred`, `json`, `msa`, `mt`,
`prep`), same flags, same JSON formats — so its
[documentation](https://github.com/aurekaresearch/OpenDDE) applies here. Dotted
config overrides work through the module entry point:

```bash
python -m runner.inference \
  --input_json_path examples/input.json \
  --model.N_cycle 4 \
  --sample_diffusion.N_step 50 \
  --dtype bf16
```

### Coming from OpenDDE

`OPENDDE_ROOT_DIR` (default `~/.cache/opendde`) has the same layout in both
projects, so CCD assets and checkpoints are reused, and your inference JSONs,
commands and output files carry over unchanged. Two things differ:

- **Weights are read as `.safetensors`.** Parameter names are unchanged, so it
  is a one-time format conversion of the checkpoint you already have (see
  [Model and runtime data](#model-and-runtime-data)); PyTorch is not needed
  afterwards.
- **The default dtype is `bf16`** instead of upstream's `fp32`, and the flags
  for features this port drops are gone: `--device`, `--foldcp_*`,
  `--trimul_kernel`, `--triatt_kernel`, `--enable_tf32`, `--deterministic`,
  `--use_default_params` and `--use_tfg_guidance`.

## Benchmarks

Three single-chain protein jobs of 100, 200 and 300 residues, prefixes of the
same chain so that length is the only variable. No MSA, no templates.
**Four recycles, 30 diffusion steps, one sample, seed 42.**

Each cell is **model forward time · peak memory**. The GPU rows use PyTorch's
MPS backend and MLX's Metal backend on the same integrated GPU (18 cores on the
M3 Pro, 32 on the M1 Max); memory is what the framework reports for its own
allocations, or peak process RSS for the CPU rows. Apple Silicon has unified
memory, so both come out of the same pool as the RAM column.

> [!NOTE]
> These are deliberately small runs. A default job (10 cycles, 200 steps, 5
> samples) on a real complex peaks near 15 GB, and the runner raises the Metal
> wired limit so those buffers stay resident. Macs with less than 24 GB of
> unified memory are not recommended: the job will still run, but the machine
> will be under heavy memory pressure and unpleasant to use meanwhile.

Everything needed to reproduce this, including the inputs and a one-command
script, is in
[examples/inference_benchmark](examples/inference_benchmark/README.md).

| Chip | RAM | Device | Precision | 100 residues | 200 residues | 300 residues |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| M3 Pro | 36 GB | PyTorch CPU | fp32 | 39.6 s · 8.5 GB | 166.3 s · 17.9 GB | 458.2 s · 25.9 GB |
| M3 Pro | 36 GB | MLX CPU | fp32 | 69.5 s · 12.2 GB | 302.4 s · 24.0 GB | 776.3 s · 24.2 GB |
| M3 Pro | 36 GB | PyTorch GPU | fp32 | 15.7 s · 6.9 GB | 78.5 s · 15.9 GB | 355.5 s · 37.3 GB |
| M3 Pro | 36 GB | PyTorch GPU | bf16 | 15.5 s · 5.9 GB | 72.6 s · 12.0 GB | 187.5 s · 24.1 GB |
| M3 Pro | 36 GB | MLX GPU | fp32 | 13.3 s · 6.1 GB | 50.6 s · 8.2 GB | 125.0 s · 10.3 GB |
| M3 Pro | 36 GB | MLX GPU | fp16 | 10.8 s · 3.1 GB | 40.8 s · 5.1 GB | 92.9 s · 6.8 GB |
| M3 Pro | 36 GB | **MLX GPU** | **bf16** | **10.5 s · 3.1 GB** | **39.8 s · 5.1 GB** | **91.3 s · 6.8 GB** |
| M1 Max | 32 GB | PyTorch CPU | fp32 | 42.6 s · 9.9 GB | 174.5 s · 18.5 GB | 522.6 s · 22.1 GB |
| M1 Max | 32 GB | MLX CPU | fp32 | 77.0 s · 12.3 GB | 368.1 s · 15.7 GB | 981.2 s · 19.0 GB |
| M1 Max | 32 GB | PyTorch GPU | fp32 | 17.1 s · 7.7 GB | 54.7 s · 17.9 GB | 578.7 s · 36.4 GB |
| M1 Max | 32 GB | PyTorch GPU | bf16 | 21.5 s · 10.1 GB | 61.9 s · 19.6 GB | 152.2 s · 33.5 GB |
| M1 Max | 32 GB | MLX GPU | fp32 | 10.0 s · 6.1 GB | 35.5 s · 8.2 GB | 91.3 s · 10.3 GB |
| M1 Max | 32 GB | **MLX GPU** | **fp16** | **10.9 s · 3.1 GB** | **29.8 s · 5.1 GB** | **64.8 s · 6.8 GB** |
| M1 Max | 32 GB | MLX GPU | bf16 | 9.8 s · 3.1 GB | 33.3 s · 5.1 GB | 72.2 s · 6.8 GB |

### Speed-up of the fastest MLX GPU dtype

Each side uses whichever dtype was faster on that machine: for MLX that is
bf16 on the M3 Pro and fp16 on the M1 Max.

| Chip | Baseline | 100 residues | 200 residues | 300 residues |
| --- | --- | ---: | ---: | ---: |
| M3 Pro | vs PyTorch CPU | 3.8x | 4.2x | 5.0x |
| M3 Pro | vs PyTorch GPU | 1.5x | 1.8x | 2.1x |
| M1 Max | vs PyTorch CPU | 4.3x | 5.9x | 8.1x |
| M1 Max | vs PyTorch GPU | 1.7x | 1.8x | 2.3x |

### A full production run

The table above uses a reduced schedule to keep every cell comparable. This
one is what you actually get from `opendde-mlx pred` with the shipped defaults:
**10 recycles, 200 diffusion steps, 5 samples**, MLX GPU fp16.

| Chip | RAM | 100 residues | 200 residues | 300 residues |
| --- | ---: | ---: | ---: | ---: |
| M3 Pro | 36 GB | 53 s · 3.1 GB | 157 s · 5.1 GB | 315 s · 6.8 GB |
| M1 Max | 32 GB | 45 s · 3.1 GB | 112 s · 5.1 GB | 218 s · 6.8 GB |

Results from other Macs are welcome. Run the script and open a pull request
adding your JSON to `examples/inference_benchmark/results/`:

```bash
python examples/inference_benchmark/run_benchmark.py \
  --dtypes fp32,bf16,fp16 \
  --out examples/inference_benchmark/results/<your_chip>.json
```

## Performance notes

- The Pairformer trunk runs in bf16 by default; the diffusion module and
  confidence head stay in fp32. Add `--fp32_diffusion false` /
  `--fp32_confidence false` to run them reduced too, or `--dtype fp32` for a
  full-precision trunk.
- `--dtype fp16` is about 10% faster than bf16 on M1/M2, whose GPUs have no
  native `bfloat`; on M3 and later bf16 is native and marginally faster. fp16
  needs two guards against its narrow exponent range: the triangle
  multiplication normalises both factors before the sum over tokens (exact,
  because the following LayerNorm is scale-invariant), and attention mask
  biases are clamped to a magnitude that stays finite.
- Diffusion pair biases for all 24 transformer blocks and the step-invariant
  atom conditioning are computed once per rollout when they fit in
  `infer_setting.diffusion_bias_cache_gb` (6 GB).
- Pair updates (triangle multiplication / attention, transitions) are chunked
  and double-buffered automatically for large complexes so peak memory stays
  bounded; attention always hits the fused Metal SDPA kernel.

## Input format

See [docs/infer_json_format.md](docs/infer_json_format.md) for the JSON schema
(`proteinChain`, `dnaSequence`, `rnaSequence`, `ligand`, `ion`,
`covalent_bonds`) and [docs/msa_template_pipeline.md](docs/msa_template_pipeline.md)
for MSA/template preparation.

## Citation

Please cite the OpenDDE technical report:
[OpenDDE: An Open-Source Drug Design Engine](https://arxiv.org/abs/2607.03787).

## License

Apache-2.0, same as upstream OpenDDE.
