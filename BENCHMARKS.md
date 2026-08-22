# Benchmarks

Operator measurements use an NVIDIA B200 (`sm100`, 148 SMs),
`nvcr.io/nvidia/pytorch:26.07-py3`, PyTorch
`2.13.0a0+9186a08b2c.nv26.07`, CUDA 13.3, driver 580.95.05, CUTLASS DSL
4.7.0, FlashInfer 0.6.17, and Transformer Engine 2.17. The full-model result
uses eight B200s, vLLM 0.27.1, PyTorch 2.13.0+cu130, and FlashInfer
0.6.16.post3.

## Measurement

| Scope | Included | Excluded |
|---|---|---|
| Grouped GEMM | one prepacked NVFP4 grouped GEMM launch | quantization, packing, allocation, compilation, autotuning |
| Routed-expert training | dispatch, expert forward and backward, routing weights, combine | router logits, top-k selection, communication, optimizer |
| Full-model serving | one warmed `LLM.generate` call | model load, JIT compilation, dataset loading, tokenization |

Operator backends run on identical inputs in alternating order. Each row uses
21 CUDA-event samples after warmup and reports median `[IQR]`. LightMoE is run
again after its baselines; a row is rejected if the two LightMoE medians differ
by more than 5%. Runs are also rejected when host wall time divided by enclosed
CUDA-event time exceeds 1.5. An unavailable backend is marked unavailable, not
replaced by another implementation.

For expert `e`, `M_e` is its local row count. **Ragged** means that the `M_e`
values are unequal; a zero-row expert is valid. The stress matrix uses unequal
counts, a route distribution concentrated on one expert, and counts around
tile boundaries with one zero-row expert.

Grouped-GEMM useful work is `2 × sum(M_e) × N × K`, with one multiply-add
counted as two FLOPs. Peak percentages use the B200 dense FP4 specification of
9,000 TFLOP/s; the 18,000 TFLOP/s structured-sparse figure is not used. This is
logical throughput, so work added by tile padding is not counted as useful
FLOPs.

## Grouped NVFP4 GEMM

These are model-derived expert projection shapes with 8,192 input tokens per
EP rank and balanced routing. The local grouped GEMM contains
`sum(M_e) = tokens × top-k` rows across `experts / EP size` local experts.
LightMoE and FlashInfer write to preallocated outputs. PyTorch
`scaled_grouped_mm` allocates its output because its API has no `out=` argument.
The baseline column selects the faster runnable baseline for each row.

| Model / projection | Experts / top-k / EP size | `N × K` | LightMoE µs `[IQR]` | Fastest baseline µs | Baseline / LightMoE | TFLOP/s | Dense FP4 peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-35B-A3B gate/up | 256 / 8 / 16 | 1,024 × 2,048 | **77.0 [0.2]** | FlashInfer 95.8 | 1.243× | 3,569 | 39.7% |
| Qwen3.5-35B-A3B down | 256 / 8 / 16 | 2,048 × 512 | **67.8 [2.0]** | FlashInfer 68.0 | 1.003× | 2,027 | 22.5% |
| Qwen3.5-397B-A17B gate/up | 512 / 10 / 32 | 2,048 × 4,096 | **296.2 [9.3]** | PyTorch 342.2 | 1.155× | 4,640 | 51.6% |
| Qwen3.5-397B-A17B down | 512 / 10 / 32 | 4,096 × 1,024 | **214.3 [1.0]** | FlashInfer 248.4 | 1.159× | 3,207 | 35.6% |
| DeepSeek-V4-Flash gate/up | 256 / 6 / 32 | 4,096 × 4,096 | **353.4 [12.1]** | PyTorch 392.4 | 1.110× | 4,667 | 51.9% |
| DeepSeek-V4-Flash down | 256 / 6 / 32 | 4,096 × 2,048 | **195.9 [2.0]** | PyTorch 226.5 | 1.156× | 4,210 | 46.8% |
| Kimi-K3 gate/up | 896 / 16 / 32 | 6,144 × 3,584 | **1,191.0 [27.6]** | FlashInfer 1,777.7 | 1.493× | 4,847 | 53.9% |
| Kimi-K3 down | 896 / 16 / 32 | 3,584 × 3,072 | **644.4 [3.1]** | FlashInfer 893.9 | 1.387× | 4,479 | 49.8% |
| GLM-5.2 gate/up | 256 / 8 / 16 | 4,096 × 6,144 | **627.7 [2.3]** | PyTorch 691.4 | 1.101× | 5,255 | **58.4%** |
| GLM-5.2 down | 256 / 8 / 16 | 6,144 × 2,048 | **406.7 [3.9]** | PyTorch 437.4 | 1.075× | 4,055 | 45.1% |
| MiniMax-M3 gate/up | 128 / 4 / 16 | 6,144 × 6,144 | **486.6 [2.2]** | PyTorch 530.5 | 1.090× | 5,084 | 56.5% |
| MiniMax-M3 down | 128 / 4 / 16 | 6,144 × 3,072 | **275.6 [2.0]** | PyTorch 304.2 | 1.104× | 4,489 | 49.9% |
| Nemotron-3.5-Lightning-30B-A3B up | 128 / 6 / 16 | 1,920 × 2,688 | **156.1 [3.4]** | PyTorch 158.3 | 1.014× | 3,250 | 36.1% |
| Nemotron-3.5-Lightning-30B-A3B down | 128 / 6 / 16 | 2,688 × 1,920 | **167.2 [0.3]** | PyTorch 168.5 | 1.008× | 3,035 | 33.7% |

All 14 balanced rows pass the two validity gates. The maximum LightMoE repeat
deviation is 4.10%, the maximum host/CUDA ratio is 1.035, and sampled cosine
against FP32 GEMM from the same BF16 operands is at least 0.99086. Nemotron's
1,856-wide expert intermediate is padded to 1,920 for the NVFP4 kernel.
Kimi-K3 uses its 3,584-wide routed latent expert input; its 7,168-wide outer
projections are outside this grouped-GEMM boundary.

The ragged matrix covers imbalanced, single-expert-skew, and alignment-stress
routing, including a zero-row expert. PyTorch is unavailable for these rows
because its NVFP4 grouped GEMM requires every group to have a 128-row-aligned
size.

## Routed-expert training

This is a routed-expert layer with 8,192 input tokens. The table retains the
standard SwiGLU rows because the measured Transformer Engine baseline does not
implement bounded SwiGLU, SwiGLU-OAI, or ReLU² with the same math. LightMoE
supports those contracts, but does not relabel a different activation as an
equivalent baseline. SiTU-GLU and model-specific outer projections remain
outside this layer boundary.

| Model / routing | Shape `H × I`, experts / top-k / EP size | LightMoE ms `[IQR]` | Transformer Engine ms `[IQR]` | Latency reduction | Output / input-gradient cosine vs BF16 |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-35B-A3B balanced | 2,048 × 512, 256 / 8 / 16 | **1.707 [0.021]** | 2.448 [0.007] | 30.2% | 0.9730 / 0.9623 |
| Qwen3.5-35B-A3B imbalanced | 2,048 × 512, 256 / 8 / 16 | **1.688 [0.017]** | 2.459 [0.011] | 31.4% | 0.9727 / 0.9623 |
| Qwen3.5-397B-A17B balanced | 4,096 × 1,024, 512 / 10 / 32 | **4.021 [0.060]** | 5.557 [0.057] | 27.6% | 0.9734 / 0.9620 |
| Qwen3.5-397B-A17B imbalanced | 4,096 × 1,024, 512 / 10 / 32 | **4.001 [0.022]** | 5.546 [0.060] | 27.9% | 0.9731 / 0.9623 |
| GLM-5.2 balanced | 6,144 × 2,048, 256 / 8 / 16 | **6.353 [0.229]** | 8.299 [0.287] | 23.5% | 0.9732 / 0.9617 |
| GLM-5.2 imbalanced | 6,144 × 2,048, 256 / 8 / 16 | **6.337 [0.317]** | 8.239 [0.484] | 23.1% | 0.9732 / 0.9622 |

All six rows pass the validity gates. The maximum repeat deviation is 3.68%
and the maximum host/CUDA ratio is 1.012. Input, router, gate/up-weight, and
down-weight gradients are present and finite. These are precision checks, not
training-convergence results.

## Full-model vLLM serving

`nvidia/DeepSeek-V4-Flash-NVFP4` revision
`e3cd60e7de98e9867116860d522499a728de1cf9` runs on eight B200s with tensor
parallel size 8 and an FP8 KV cache. Attention, scheduling, collectives, and all
non-MoE operators are fixed; only the NVFP4 MoE backend changes. Prompts are
fixed SWE-bench Verified samples from revision
`78f471bf655a3137b2e8a75af1501690ec009ec3`.

| Workload | Fixed batch | LightMoE initial median `[IQR]` | FlashInfer TensorRT-LLM median `[IQR]` | LightMoE repeat median `[IQR]` | Baseline / LightMoE |
|---|---:|---:|---:|---:|---:|
| Prefill | 8 × 1,024 input → 1 output | 221.670 [2.869] ms | 250.464 [4.300] ms | 226.927 [1.803] ms | 1.10–1.13× |
| Decode | 32 × 256 input → 256 output | 2.4488 [0.0125] s | 2.5395 [0.0667] s | 2.4587 [0.0653] s | 1.033–1.037× |

Each backend has 21 warmed runs. Initial-to-repeat drift is 2.37% for prefill
and 0.40% for decode. Inputs, outputs, and generated token sequences match
across backends. This is the only full-model backend swap claimed here; an
operator result is never presented as end-to-end performance.

## Run

Pinned model sources and revisions live in `benchmarks/model_shapes.py`.

```bash
python -m benchmarks.nvfp4_gemm \
  --models all --tokens 8192 --routing balanced \
  --projections gate_up,down \
  --backends lightmoe,flashinfer_cutedsl,torch_scaled_grouped_mm \
  --mode prepacked --warmup 5 --iterations 21

python -m benchmarks.nvfp4_gemm \
  --models all --tokens 8192 \
  --routing imbalanced,single_expert_skew,alignment_stress \
  --projections gate_up,down \
  --backends lightmoe,flashinfer_cutedsl,torch_scaled_grouped_mm \
  --mode prepacked --warmup 5 --iterations 21

python -m benchmarks.moe \
  --models all --tokens 8192 --routing balanced,imbalanced \
  --backends lightmoe,transformer_engine_nvfp4_fused,pytorch_bf16 \
  --scope full-layer --pass fwd_bwd --warmup 15 --iterations 21 \
  --interleave-training
```
