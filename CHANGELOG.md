# Changelog

## 0.1.0 (2026-09-03)

Initial MLX rewrite of OpenDDE 1.1.1 for Apple Silicon.

- Complete inference model in MLX: input embedder, Pairformer trunk with MSA and
  template modules, structural token expansion and refiner, diffusion module,
  distogram and confidence heads, shape complementarity, confidence summaries.
- Released `opendde.pt` checkpoints load after a one-time conversion to
  `.safetensors` (`opendde convert`); parameter names are unchanged.
- Inference runs the trunk in bf16 by default (upstream defaults to fp32);
  `--dtype fp32` and `--dtype fp16` are also available. fp16 needs a normalised
  triangle multiplication and clamped attention mask biases to stay inside its
  exponent range.
- NumPy featurisation pipeline (bit-identical features to the PyTorch pipeline).
- Same CLI surface (`opendde pred/json/msa/mt/prep/doctor`) plus `opendde convert`.
- Numerical parity tests against the PyTorch implementation for every module and
  a stage-by-stage end-to-end check with the released weights.
- Removed: Fold-CP context parallelism, torch.distributed, cuEquivariance and
  Triton kernels, CUDA LayerNorm, Docker image, training-free guidance (TFG),
  device selection (`--device`), TF32 and deterministic-algorithm switches.
