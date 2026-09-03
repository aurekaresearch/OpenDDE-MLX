# Benchmarks

The reproducible benchmark suite, its inputs and the full results for an
Apple M3 Pro and an Apple M1 Max live in
[examples/inference_benchmark](../examples/inference_benchmark/README.md):
MLX against upstream PyTorch, on the GPU and the CPU, at 100/200/300 residues,
with forward time, peak memory and confidence for every cell. The main tables
are also inlined in the [README](../README.md#benchmarks).

Headline, 300-residue chain, forward time and peak memory:

| Engine | M3 Pro (36 GB) | M1 Max (32 GB) |
| --- | ---: | ---: |
| PyTorch CPU fp32 | 458.2 s / 25.9 GB | 522.6 s / 22.1 GB |
| PyTorch GPU fp32 | 355.5 s / 37.3 GB | 578.7 s / 36.4 GB |
| MLX GPU bf16 | 91.3 s / 6.8 GB | 72.2 s / 6.8 GB |

Upstream fp32 exceeds physical memory on both machines at this size, so those
cells are swap-bound. MLX bf16 needs 5.5x less memory for the same job.
