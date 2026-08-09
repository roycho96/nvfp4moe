# B200 real-model benchmark

Measured on August 10, 2026. This benchmark compares complete MoE expert
layers, including dispatch, routing-probability weighting, combine, and the
full backward pass. Both promoted runs passed the GPU health check.

## 📐 Workloads

| trace | geometry | activation | observed routing |
|---|---|---|---|
| Qwen3-30B-A3B | `T=8192, d=2048, I=768, E=128, k=8` | SwiGLU | 0-3,406 rows/expert, 6 empty |
| Gemma 4 26B-A4B | `T=8192, d=2816, I=704→768, E=128, k=8` | tanh-GeGLU | 0-4,877 rows/expert, 10 empty |

The inputs come from FineWeb-Edu `sample-10BT` revision `v1.0.0`, the official
model tokenizers, and published layer-0 checkpoint weights. Capture records
token hashes and validates each model geometry against its checkpoint config.
Gemma's intermediate width is padded from 704 to 768 for aligned kernels; the
extra work remains inside every measured latency.

Five expert backends are measured:

- Transformer Engine BF16 `GroupedLinear`, forward and backward.
- Transformer Engine NVFP4 `GroupedLinear`, forward and backward. TE 2.17 can
  apply grouped NVFP4 RHT to at most 64 tensors, so E=128 uses two launches.
- TE dispatch with DeepGEMM BF16 M-grouped forward/dgrad and K-grouped wgrad.
- TE dispatch with DeepGEMM FP8 x FP4, forward only.
- `nvfp4moe` NVFP4 x NVFP4 with `rht=True` and
  `delayed_col_amax=True`, forward and backward.

Attention is not part of the expert-only comparison. The contextual Qwen block
uses PyTorch SDPA, matching the stack used for these measurements.

## ⏱️ Expert latency

Each row is measured in a fresh B200 container. JIT compilation and BF16
reference calculations finish before timing. Reported times are CUDA-event p50
with `[p10, p90]` after warmup; memory is peak allocated CUDA memory. Ratios are
formed only within the same run because absolute B200 speed varies between
rental sessions.

### Qwen3-30B-A3B

| backend | precision | forward ms | fwd+bwd ms | output cosine / rel-L2 | peak GiB fwd/train |
|---|---:|---:|---:|---:|---:|
| TE | BF16 | 3.318 `[3.269, 3.335]` | 10.320 `[10.289, 10.328]` | 0.999993 / 0.00384 | 4.100 / 5.299 |
| TE 2 x 64 | NVFP4 | 7.845 `[7.835, 7.857]` | 21.424 `[21.387, 21.452]` | 0.995535 / 0.09734 | 6.033 / 6.926 |
| TE + DeepGEMM | BF16 | 1.959 `[1.953, 1.969]` | 11.163 `[11.109, 11.173]` | 0.999993 / 0.00384 | 4.286 / 8.304 |
| TE + DeepGEMM | FP8 x FP4 | 4.309 `[4.302, 4.313]` | — | 0.996663 / 0.08699 | 4.544 / — |
| **nvfp4moe** | **NVFP4 x NVFP4** | **0.558 `[0.550, 0.567]`** | **3.485 `[3.471, 3.489]`** | **0.996526 / 0.08444** | 4.952 / 7.480 |

Within this session, `nvfp4moe` was 14.07x/6.15x faster than TE NVFP4 and
3.51x/3.20x faster than TE + DeepGEMM BF16 for forward/training.

### Gemma 4 26B-A4B

| backend | precision | forward ms | fwd+bwd ms | output cosine / rel-L2 | peak GiB fwd/train |
|---|---:|---:|---:|---:|---:|
| TE | BF16 | 3.522 `[3.501, 3.558]` | 9.409 `[9.381, 9.439]` | 0.999993 / 0.00382 | 5.288 / 6.937 |
| TE 2 x 64 | NVFP4 | 8.044 `[8.020, 8.054]` | 18.294 `[18.275, 18.332]` | 0.995664 / 0.09537 | 7.800 / 9.068 |
| TE + DeepGEMM | BF16 | 2.481 `[2.474, 2.487]` | 15.792 `[15.768, 15.815]` | 0.999993 / 0.00382 | 7.024 / 12.329 |
| TE + DeepGEMM | FP8 x FP4 | 5.430 `[5.423, 5.433]` | — | 0.995732 / 0.09844 | 5.908 / — |
| **nvfp4moe** | **NVFP4 x NVFP4** | **0.607 `[0.604, 0.618]`** | **4.157 `[4.149, 4.163]`** | **0.996787 / 0.08189** | 6.461 / 9.703 |

Within this session, `nvfp4moe` was 13.26x/4.40x faster than TE NVFP4 and
4.09x/3.80x faster than TE + DeepGEMM BF16 for forward/training.

## 🧪 Gradient precision

All training backends receive the same deterministic `dY`. The resulting
input, routing-probability, gate/up-weight, and down-weight gradients are
compared elementwise with BF16 PyTorch autograd using the captured checkpoint
weights. Alignment-only Gemma channels are cropped before comparison.

### Cosine similarity

| model / backend | dX | router grad | gate/up dW | down dW |
|---|---:|---:|---:|---:|
| Qwen / TE BF16 | 0.999993 | 0.999997 | 1.000000 | 1.000000 |
| Qwen / TE NVFP4 | 0.971126 | 0.990131 | 0.965729 | 0.985860 |
| Qwen / TE + DeepGEMM BF16 | 0.999987 | 0.999997 | 0.999994 | 1.000000 |
| Qwen / **nvfp4moe** | **0.975384** | **0.992358** | **0.968857** | **0.986734** |
| Gemma / TE BF16 | 0.999993 | 0.999997 | 1.000000 | 1.000000 |
| Gemma / TE NVFP4 | 0.967544 | 0.992451 | 0.962373 | 0.987714 |
| Gemma / TE + DeepGEMM BF16 | 0.999988 | 0.999997 | 0.999994 | 1.000000 |
| Gemma / **nvfp4moe** | **0.971858** | **0.994196** | **0.965768** | **0.988153** |

### Relative L2 error

| model / backend | dX | router grad | gate/up dW | down dW |
|---|---:|---:|---:|---:|
| Qwen / TE NVFP4 | 0.23944 | 0.14102 | 0.26283 | 0.16798 |
| Qwen / **nvfp4moe** | **0.22193** | **0.12360** | **0.25141** | **0.16351** |
| Gemma / TE NVFP4 | 0.25464 | 0.12355 | 0.27498 | 0.15632 |
| Gemma / **nvfp4moe** | **0.23692** | **0.10783** | **0.26310** | **0.15383** |

`nvfp4moe` is closer to the BF16 reference than TE NVFP4 in all eight cosine
and all eight relative-L2 comparisons. Every training row also produced finite,
nonzero parameter gradients. This measures one layer from two checkpoints; it
is not a full-model convergence result.

## 🧱 Qwen block context

The reconstructed Qwen shell matches the capture at 0.999990 expert-input
cosine, 0.999985 routing-weight cosine, and 97.626% exact top-k slot match.
Because its inputs are reconstructed rather than exact, this table is context;
the expert-only tables above are the primary comparison.

| compile mode | expert | forward p50 / p90 ms | fwd+bwd p50 / p90 ms |
|---|---|---:|---:|
| eager | TE BF16 | 5.720 / 5.724 | 17.090 / 17.249 |
| eager | TE + DeepGEMM BF16 | 4.040 / 4.050 | 17.373 / 17.379 |
| eager | **nvfp4moe** | **2.599 / 2.601** | **9.493 / 9.497** |
| reduce-overhead | TE BF16 | 10.582 / 16.093 | 15.098 / 19.837 |
| reduce-overhead | TE + DeepGEMM BF16 | 2.985 / 5.304 | 14.153 / 14.185 |
| reduce-overhead | **nvfp4moe** | **1.538 / 4.168** | **6.285 / 6.295** |

Compiled forward rows retain tail outliers, so the compile p50 is not presented
as a tail-latency result.

## 🔁 Reproduce

The measured environment was NGC PyTorch 26.07, CUDA 13.3, Transformer Engine
2.17.0, CUTLASS DSL 4.6.0, and DeepGEMM 2.6.1 at commit
`559d79fb6994a58b8a15b4b93bf13ccc16edf247`. Qwen canary drift was 1.075% and
Gemma drift was 1.832%, below the 5% rejection threshold.

Run capture and measurement separately so B200 jobs remain serial:

```bash
modal run benchmarks/q32_real_stack.py \
  --models qwen3_30b_a3b --capture-only
modal run benchmarks/q32_real_stack.py \
  --models qwen3_30b_a3b --benchmark-only

modal run benchmarks/q32_real_stack.py \
  --models gemma4_26b_a4b_local --capture-only
modal run benchmarks/q32_real_stack.py \
  --models gemma4_26b_a4b_local --benchmark-only
```

Machine-readable results are saved as:

- `benchmarks/q32_real_stack_qwen3_30b_a3b.json`
- `benchmarks/q32_real_stack_gemma4_26b_a4b_local.json`

The source models and dataset are published by
[Qwen](https://huggingface.co/Qwen/Qwen3-30B-A3B),
[Google](https://huggingface.co/google/gemma-4-26B-A4B), and
[Hugging Face](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
