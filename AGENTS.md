# AGENTS.md — OpenDDE-MLX Agent Guide

Applies to the whole repository.

## Scope

- Apple Silicon only, single machine, MLX only. No PyTorch, CUDA, MPS, Fold-CP,
  cuEquivariance or multi-process code may be (re)introduced in runtime code.
  PyTorch is allowed only in `tests/parity/` and for `.pt` checkpoint conversion.
- Keep the code concise: prefer removing a code path over adding a fallback.

## Repository map

- `opendde/model/`: MLX model. `primitives.py` (Linear/LayerNorm/attention),
  `triangular.py`, `transformer.py`, `pairformer.py`, `diffusion.py`,
  `confidence.py`, `structural_tokens.py`, `opendde.py` (prediction loop),
  `generator.py` (sampler), `sample_confidence.py`, `shape_complementarity.py`,
  `checkpoint.py` (`.pt` -> `.safetensors`, key remap, loading).
- `opendde/data/`: NumPy featurisation pipeline (parsers, CCD, tokenizer, MSA, templates).
- `opendde/config/`: defaults (`model_base.py`), typed schema, CLI merge engine.
- `runner/`: Click CLI (`opendde-mlx`), inference runner, output dumper, MSA/template search.
- `tests/parity/`: numerical parity against the upstream PyTorch implementation.

## Conventions

- Parameter attribute names must match the released checkpoint keys
  (`/tmp/keys.txt` style listing via `mx.load(...).keys()`); lists replace
  `nn.ModuleList`, and `nn.Sequential` indices are remapped in `checkpoint.py`.
- Modules are functional: no in-place updates; chunk large pair tensors instead.
- Feature dict values are `mx.array` (int64 for indices/masks, float32 otherwise).
- Randomness comes from NumPy generators seeded per job seed.

## Commands

```bash
uv venv --python 3.12 && uv pip install --python .venv -e . && uv pip install --python .venv --group dev
.venv/bin/python -m pytest tests -q -m "not parity"
export OPENDDE_TORCH_REPO=/path/to/OpenDDE   # upstream repo for parity dumps
.venv/bin/python tests/parity/reference.py primitives /tmp/parity/primitives.npz  # etc.
.venv/bin/python -m pytest tests/parity -q
ruff check . && ruff format --check .
```

## Releasing

Push a `vX.Y.Z` tag that matches `opendde/version.py` and has a `CHANGELOG.md`
section. `.github/workflows/release.yml` validates the metadata, builds once,
verifies the wheel and sdist in clean environments, publishes to PyPI through
trusted publishing, and creates the GitHub Release from the same build. It only
runs in the `aurekaresearch/OpenDDE-MLX` repository.

```bash
.venv/bin/python scripts/check_release.py --tag v0.1.0
```

## Handoff checklist

- Parity tests for every touched module pass; new modules get a reference dump + test.
- `opendde-mlx pred -i examples/parity_small.json -o /tmp/out --use_msa false --cycle 2 --step 20 --sample 1` runs.
- No checkpoints, outputs or large files are committed.
