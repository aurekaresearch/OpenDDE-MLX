# Inference Instructions

Reference for installing OpenDDE-MLX, preparing runtime data, and running the
`opendde-mlx` commands on Apple Silicon.

## Install

From PyPI:

```bash
uv venv --python 3.12
uv pip install --python .venv opendde-mlx
uv run --no-project --python .venv opendde-mlx doctor
```

From source:

```bash
git clone git@github.com:aurekaresearch/OpenDDE-MLX.git
cd OpenDDE-MLX
uv venv --python 3.12
uv pip install --python .venv -e .
uv run --no-project --python .venv opendde-mlx doctor
```

Requirements: macOS on Apple Silicon (M1 or newer), Python 3.11-3.13, MLX 0.30+.
PyTorch (CPU) is optional and only used to convert `.pt` checkpoints; install it
with the `convert` extra (`opendde-mlx[convert]`) or directly.

## Runtime data

```text
$OPENDDE_ROOT_DIR/                      # default ~/.cache/opendde
├── checkpoint/opendde.safetensors      # converted weights
├── checkpoint/opendde.pt               # released weights (kept for conversion)
├── common/components.cif               # CCD (auto-downloaded)
├── common/components.cif.rdkit_mol.pkl
└── search_database/                    # local template / RNA-MSA search only
```

`opendde-mlx pred` downloads the missing managed assets with size/SHA-256
verification. From a source checkout the upstream helper script still works:

```bash
export OPENDDE_ROOT_DIR=/path/to/opendde_data
bash scripts/download_opendde_data.sh --skip-search-database
opendde-mlx convert $OPENDDE_ROOT_DIR/checkpoint/opendde.pt
```

## Prediction

```bash
opendde-mlx pred -i examples/example_without_msa.json -o ./output --use_msa false
```

| Option                    | Default      | Meaning                                              |
| ------------------------- | ------------ | ---------------------------------------------------- |
| `-i/--input`              | required     | JSON file or directory of JSON files                 |
| `-o/--out_dir`            | `./output`   | Output directory                                     |
| `-s/--seeds`              | JSON/random  | Comma-separated seeds                                |
| `-c/--cycle`              | 10           | Pairformer recycling iterations                      |
| `-p/--step`               | 200          | Diffusion steps                                      |
| `-e/--sample`             | 5            | Diffusion samples                                    |
| `-d/--dtype`              | `bf16`       | `fp16` is faster on M1/M2; `fp32` for full precision  |
| `--fp32_diffusion`        | true         | Keep the diffusion module in fp32 under `bf16`       |
| `--fp32_confidence`       | true         | Keep the confidence head in fp32 under `bf16`        |
| `--use_msa`               | true         | Protein MSA (ColabFold MMseqs2 server)               |
| `--use_template`          | false        | Template search / features                           |
| `--use_rna_msa`           | false        | RNA MSA                                              |
| `--enable_cache`          | true         | Cache diffusion conditioning across steps            |
| `--enable_fusion`         | true         | Share the pair LayerNorm across diffusion blocks     |
| `--need_atom_confidence`  | true         | Write atom-level confidence JSON                     |
| `--load_checkpoint_path`  | released     | Explicit `.safetensors` or `.pt` checkpoint          |

Outputs per job and seed:

```text
output/<name>/seed_<seed>/predictions/
├── <name>_sample_<rank>.cif
├── <name>_summary_confidence_sample_<rank>.json
└── <name>_full_data_sample_<rank>.json
```

Samples are ranked by `ranking_score`; `rank 0` is the best sample.

## Dotted overrides

Any config key can be set through the module entry point:

```bash
python -m runner.inference \
  --input_json_path examples/input.json --dump_dir ./output \
  --model.N_cycle 4 --sample_diffusion.N_step 50 --sample_diffusion.N_sample 2 \
  --skip_amp.sample_diffusion false --dtype fp16
```

## Memory guidance

At least 24 GB of unified memory is recommended: a default job on a real
complex peaks near 15 GB, and the runner raises the Metal wired limit to keep
those buffers resident, so a smaller Mac stays under heavy memory pressure for
the whole run.

Peak memory grows with `N_token^2 * c_z`. On a 36 GB machine complexes up to
roughly 1500 tokens run in fp32; keep the `bf16` default, use fewer samples
(`--sample 1`) or `--infer_setting.sample_diffusion_chunk_size 1` for larger
inputs. Pair operations are chunked automatically above 256 tokens with a
fixed byte budget per chunk, and each stage logs its time and peak memory.

## Speed guidance

The `bf16` default is the right choice on M3 and later; on M1/M2, whose GPUs
have no native `bfloat`, `--dtype fp16` is about 10% faster. The trunk is
compute-bound (about 1.7 s per Pairformer block at 600 tokens on an M3 Pro),
so `--cycle` is the main lever. Diffusion cost is dominated by per-step
kernel launches; `--fp32_diffusion false` gains about 15% per step.

## MSA, templates and RNA MSA

See [msa_template_pipeline.md](./msa_template_pipeline.md). `opendde-mlx msa`,
`opendde-mlx mt` and `opendde-mlx prep` are unchanged from upstream and require the
same external tools (HMMER, Kalign) and databases.
