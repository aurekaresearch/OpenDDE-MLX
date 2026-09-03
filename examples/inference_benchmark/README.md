# Inference benchmark

A small, self-contained benchmark that compares this MLX port against the
upstream PyTorch implementation on the same Mac, at three sequence lengths and
on both the GPU and the CPU. Everything needed to reproduce it is in this
folder.

## Protocol

Three single-chain protein jobs of 100, 200 and 300 residues. They are prefixes
of the same 352-residue chain (`7wux` chain A), so length is the only variable.
No MSA, no templates, no ligands.

```
--model.N_cycle 4 --sample_diffusion.N_step 30 --sample_diffusion.N_sample 1
--use_msa false
```

These are deliberately small so the benchmark stays runnable on any Mac. A
default job (10 cycles, 200 steps, 5 samples) on a real complex peaks near
15 GB; Macs with less than 24 GB of unified memory are not recommended for
production runs.

Seed 42. Reported time is model forward time, excluding featurisation and
checkpoint loading. GPU peak memory is what the framework reports
(`mx.get_peak_memory` / `torch.mps.driver_allocated_memory`); CPU peak memory is
peak process RSS. pLDDT is the top-ranked sample's summary confidence.

## Reproduce

```bash
# MLX only
python examples/inference_benchmark/run_benchmark.py --dtypes fp32,bf16,fp16 \
  --out results/mymac.json

# MLX + upstream PyTorch, GPU and CPU
python examples/inference_benchmark/run_benchmark.py \
  --torch-repo ~/src/OpenDDE --backends mlx,torch,mlx-cpu,torch-cpu
```

The script prints a markdown table and writes a JSON record per machine. Full
results for the two machines below are in `results/`.

## Results

Forward time · peak memory.

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

Results from other Macs are welcome: run the script above and open a pull
request adding your JSON to `results/`.

### A full production run

`opendde-mlx pred` defaults to **10 recycles, 200 diffusion steps and 5 samples**.
On MLX GPU fp16 that costs:

| Chip | RAM | 100 residues | 200 residues | 300 residues |
| --- | ---: | ---: | ---: | ---: |
| M3 Pro | 36 GB | 53 s · 3.1 GB | 157 s · 5.1 GB | 315 s · 6.8 GB |
| M1 Max | 32 GB | 45 s · 3.1 GB | 112 s · 5.1 GB | 218 s · 6.8 GB |

The reduced schedule above exists so that every engine, dtype and length stays
comparable within a reasonable wall time; this is what a real job looks like.

## Speed-up of the fastest MLX GPU dtype

Each side uses whichever dtype was faster on that machine: for MLX that is
bf16 on the M3 Pro and fp16 on the M1 Max.

| Chip | Baseline | 100 residues | 200 residues | 300 residues |
| --- | --- | ---: | ---: | ---: |
| M3 Pro | vs PyTorch CPU | 3.8x | 4.2x | 5.0x |
| M3 Pro | vs PyTorch GPU | 1.5x | 1.8x | 2.1x |
| M1 Max | vs PyTorch CPU | 4.3x | 5.9x | 8.1x |
| M1 Max | vs PyTorch GPU | 1.7x | 1.8x | 2.3x |

## What the numbers say

**A full run costs 4-5x the reduced schedule**, and no more memory: peak is set
by the pair representation, not by the number of samples or steps.

**Memory is the real story.** At 300 residues upstream fp32 peaks at 37.3 GB on
a 36 GB machine and 36.4 GB on a 32 GB one; both are past the limit, so the
355 s and 579 s in those cells are swap, not arithmetic. MLX bf16 needs 6.8 GB
for the same job, 5.5x less, which is what keeps its scaling clean. Treat the
two fp32 300-residue GPU cells as high-variance: an earlier run of the same
M1 cell measured 331 s.

**The GPU is worth 4-7x over the CPU** once the engine is efficient enough to
stay resident. Note that upstream's CPU path is about 1.8x faster than MLX's
CPU path; MLX does not target the CPU, and that column is a baseline, not a
recommendation.

**Reduced precision behaves differently per generation.** M3 has native
`bfloat` in the GPU (matmul: 4.00 fp32 vs 4.58 bf16 vs 4.58 fp16 TFLOPS), M1
does not (6.43 fp32, 6.03 bf16, 7.09 fp16). So on the M1 Max upstream bf16 is
*slower* than fp32 at 100 and 200 residues, and only wins at 300 where halving
memory traffic beats everything else; and MLX fp16 beats MLX bf16 there by about
10%. On the M3 Pro bf16 is the best MLX dtype. Both win everywhere against fp32
because the memory saving comes on top of the arithmetic.

**Accuracy is equivalent, and the spread is the noise floor.** Within one
machine and one length, all seven configurations agree to about 1 pLDDT point.
Across machines the same configuration can differ by more (MLX at 200 residues:
45.0 on M3, 42.0 on M1), because a single diffusion sample of a truncated chain
without MSA sits in a chaotic regime where different GPU rounding leads to a
different structure. MLX CPU and MLX GPU on the same machine agree to four
decimal places. Do not read these values as a quality comparison; they are a
check that no configuration is broken.

## fp16

`--dtype fp16` is measured for MLX only; upstream maps `--dtype` to bf16 or
fp32 and has no fp16 path. fp16 is roughly 10% faster than bf16 on the M1 Max
and marginally slower on the M3 Pro, matching the matmul throughput of the two
generations. Its 5-bit exponent needs two guards, both in this repository: the
triangle multiplication normalises its two factors before the sum over tokens
(exact, because the LayerNorm that follows is invariant to a positive scale),
and attention mask biases are clamped so that constants like `1e9` do not
become `inf` and turn a masked softmax row into NaN.

## Caveat: upstream bf16 on recent PyTorch

`--dtype bf16 --device mps` raises `Index put requires the source and
destination dtypes match` in the structural token expander under torch 2.14
(it works under 2.7.1). Three assignments in
`opendde/model/modules/structural_tokens.py` need `.to(flat_delta.dtype)` on the
right-hand side. That patch was applied to the upstream checkout to measure the
`torch GPU bf16` column on the M3 Pro, then reverted. It is an upstream dtype
bug, not a hardware limit.
