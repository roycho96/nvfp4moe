# B200 benchmark report

Measured on August 11, 2026. Results were regenerated after the native-kernel
cleanup and passed the release CI described below.

## 📐 Primary workload

| item | value |
|---|---|
| GPU | NVIDIA B200 (`sm100`) |
| sequence | 8,192 tokens |
| model | Qwen3-30B-A3B layer 0 |
| expert geometry | `d=2048, I=768, E=128, top-k=8` |
| activation | SwiGLU |
| dataset | FineWeb-Edu |
| precision | NVFP4 expert operands, BF16 router and outputs |
| software | NGC PyTorch 26.07, CUDA 13, CUTLASS DSL 4.6 |

The trace uses the published tokenizer, layer-0 checkpoint weights, model
activations, and routing decisions. JIT compilation and calibration complete
before timing.

## ⏱️ Timing method

- Same-process CUDA-event timing
- Five warmups before each measurement group
- Median of 15 forward samples
- Median of seven forward-plus-backward samples
- A fixed GEMM canary before and after the suite; sessions with more than 5%
  drift are rejected
- Ratios are formed only within the same B200 session

The comparison matrix includes Transformer Engine BF16, Transformer Engine
NVFP4, TE dispatch with DeepGEMM BF16, TE dispatch with DeepGEMM FP8 × FP4,
and `nvfp4moe`.

## ⏱️ Qwen3-30B-A3B expert layer

| backend | precision | forward | forward + backward | output cosine / rel-L2 |
|---|---:|---:|---:|---:|
| Transformer Engine | BF16 | 5.256 ms | 16.488 ms | 0.999993 / 0.00383 |
| Transformer Engine 2 × 64 | NVFP4 | 12.900 ms | 33.306 ms | 0.995535 / 0.09734 |
| TE dispatch + DeepGEMM | BF16 | 2.244 ms | 11.536 ms | 0.999993 / 0.00383 |
| TE dispatch + DeepGEMM | FP8 × FP4 | 4.549 ms | — | 0.996663 / 0.08699 |
| `nvfp4moe` | NVFP4 × NVFP4 | **0.735 ms** | **3.328 ms** | 0.996526 / 0.08444 |

In this session, `nvfp4moe` was 17.55×/10.01× faster than TE NVFP4 and
3.05×/3.47× faster than TE + DeepGEMM BF16 for forward/training. The
forward-only DeepGEMM FP8 × FP4 adapter was 6.19× slower. The before/after
canary drift was 2.19%, within the 5% acceptance threshold.

A final native-only regression run after the SM100 scheduler and L2-policy
changes measured 0.552 ms forward and 3.492 ms forward + backward. The summed
GPU-kernel time was 0.439 ms and 2.275 ms, respectively; canary drift was
0.57%. In the full Qwen block with SDPA, the same run measured:

| expert path | compile mode | block forward | block forward + backward |
|---|---:|---:|---:|
| TE BF16 | eager | 9.479 ms | 22.385 ms |
| TE dispatch + DeepGEMM BF16 | eager | 4.243 ms | 17.826 ms |
| `nvfp4moe` | eager | **2.646 ms** | **10.012 ms** |
| TE dispatch + DeepGEMM BF16 | reduce-overhead | 3.425 ms | 14.575 ms |
| `nvfp4moe` | reduce-overhead | **1.706 ms** | **5.956 ms** |

TE 2.17 accepts at most 64 tensors in one grouped NVFP4 RHT call, so the
128-expert TE case uses two groups of 64. DeepGEMM BF16 uses M-grouped kernels
for forward and input gradients and K-grouped kernels for weight gradients.

## 🧪 Precision checks

Every backend receives the same checkpoint weights, routed inputs, routing
weights, and deterministic output gradient. The suite records cosine
similarity and relative L2 error against BF16 PyTorch for:

- expert-layer output
- input gradient
- router gradient
- gate/up weight gradient
- down-projection weight gradient

Packed E2M1 values and active E4M3 scale factors are also checked against the
project's independent PyTorch format reference. Empty experts, expert tails,
and scheduler boundary cases are included.

| `nvfp4moe` tensor | cosine similarity | relative L2 |
|---|---:|---:|
| output | 0.996526 | 0.08444 |
| input gradient | 0.975384 | 0.22193 |
| router gradient | 0.992358 | 0.12360 |
| gate/up weight gradient | 0.968857 | 0.25141 |
| down weight gradient | 0.986734 | 0.16351 |

The same real-data procedure on Gemma 4 26B A4B measured 0.655 ms forward and
3.251 ms forward + backward. Output cosine similarity to BF16 was 0.996787;
input- and gate/up-gradient cosine similarities were 0.971858 and 0.965768.
The 704-wide expert dimension is padded consistently to 768 and cropped back
at the model boundary. Canary drift was 0.87%.

## 🧭 Coverage

The standalone B200 matrix exercises practical local expert shards for:

| model family | local experts | hidden | expert intermediate | activation |
|---|---:|---:|---:|---|
| Qwen3-30B-A3B | 128 | 2,048 | 768 | SwiGLU |
| Qwen3-235B-A22B | 128 | 4,096 | 1,536 | SwiGLU |
| Gemma 4 26B A4B | 128 | 2,816 | 704→768 padded | GeGLU |
| DeepSeek V3.2 | 256 | 7,168 | 2,048 | SwiGLU |
| Kimi K2.7 EP3 shard | 128 / 384 | 7,168 | 2,048 | SwiGLU |
| MiniMax M2 | 256 | 3,072 | 1,536 | SwiGLU |
| OLMoE-1B-7B | 64 | 2,048 | 1,024 | SwiGLU |

Routing cases cover decode, 2K/8K/16K prefill, uniform, Zipf, hotspot,
single-expert concentration, empty experts, and non-multiple tails. CTA tiles
cover `tile_M={128,256}` and `tile_N={128,256}`. The scheduler edge suite
includes total work sizes from zero through four tiles.

Real-model routing captures use FineWeb-Edu, C4 English, FineMath 4+,
FineWeb2 Korean, and Stack v3 code. Qwen and Gemma use checkpoint-derived
routing. Larger published geometries use synthetic routing with the same
expert counts, top-k, dimensions, and routed-row distributions.

## 🔁 Fused dgrad2 matrix

Each row includes the FC2 input-gradient GEMM, gated derivative, packed BF16
gradient output, and saved post-activation output. CUDA-event results are the
median isolated kernel time from the final four-tile sweep. NCU duration is
shown separately for the long-K profiles; values from the two methods are not
mixed into ratios.

| model geometry | best tile | CUDA-event kernel | NCU duration |
|---|---:|---:|---:|
| Qwen3-30B-A3B | 256 × 128 | **0.146 ms** | — |
| Gemma 4 26B A4B, padded | 256 × 128 | **0.175 ms** | — |
| OLMoE-1B-7B | 256 × 128 | **0.181 ms** | — |
| Kimi K2.7 EP3 local shard | 256 × 256 | **0.316 ms** | **0.305 ms** |
| MiniMax M2 | 256 × 128 | **0.326 ms** | — |
| DeepSeek V3.2 | 256 × 256 | **0.681 ms** | **0.656 ms** |
| ReGLU synthetic | 128 × 128 | **0.019 ms** | — |

All rows passed sampled PyTorch activation-backward checks and repeated-launch
determinism. Every geometry exercised all four combinations of
`tile_M={128,256}` and `tile_N={128,256}`. The long-K kernels use 128-wide MMA
K tiles and operand-specific L2 eviction hints; shorter K shapes retain the
lower-overhead default path.

## ✅ Release validation

- 44 local pytest checks
- Ruff lint and format checks
- B200 forward/backward layer suite
- Same-session package-refactor A/B: Qwen GEMMs were 0.9–2.5% faster and
  fused dgrad2 was 1.6% faster than the preceding commit
- B200 scheduler edges with one through four tiles and empty experts
- TE row/column NVFP4 data and scale factors bitwise equal
- Qwen adapter gradients reach router and BF16 expert masters
- Default and delayed-amax training, stochastic rounding, decreasing routed
  batch size, and both `pre_permute` settings

## 🔁 Reproduce

Run local CPU/static checks:

```bash
python -m pytest -q
ruff check nvfp4moe benchmarks tests examples
ruff format --check nvfp4moe benchmarks tests examples
```

Run the B200 kernel suite:

```bash
modal run benchmarks/modal_ci.py
modal run benchmarks/modal_ci.py --frontier-matrix
```

Capture and benchmark real Qwen and Gemma expert layers:

```bash
modal run benchmarks/real_model_benchmark.py \
  --models qwen3_30b_a3b,gemma4_26b_a4b_local \
  --datasets fineweb_edu \
  --capture-only

modal run benchmarks/real_model_benchmark.py \
  --models qwen3_30b_a3b,gemma4_26b_a4b_local \
  --datasets fineweb_edu \
  --benchmark-only
```

Run the five-domain Qwen routing sweep:

```bash
modal run benchmarks/real_model_benchmark.py \
  --models qwen3_30b_a3b \
  --datasets fineweb_edu,c4_en,finemath_4plus,fineweb2_ko,stack_v3_code \
  --backends nvfp4moe \
  --no-stack-context \
  --benchmark-only
```

Machine result files are intentionally not committed. Keep raw JSON, NCU, and
NSYS artifacts with the run metadata when publishing new numbers.
