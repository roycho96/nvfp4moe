# B200 benchmark results

Operator results use one NVIDIA B200 (`sm100`, 148 SMs). Expert-parallel
results use eight B200s in one node.

Single-GPU and expert-parallel runs use
`nvcr.io/nvidia/pytorch:26.07-py3`, PyTorch
`2.13.0a0+9186a08b2c.nv26.07`, CUDA 13.3, driver 580.95.05, CUTLASS DSL 4.7.0,
FlashInfer 0.6.17, and Transformer Engine 2.17. The full-model run uses vLLM
0.27.1, PyTorch 2.13.0+cu130, and FlashInfer 0.6.16.post3.

## Measurement contract

| Scope | Timed boundary | Excluded |
|---|---|---|
| Grouped GEMM | one prepacked GEMM launch | allocation, compilation, autotuning, packing |
| Inference MoE | dispatch, input quantization, gate/up projection, activation, requantization, down projection, routing weights, combine | router logits, top-k selection, calibration, compilation, weight packing |
| Full-model serving | one warmed `LLM.generate` call | model load, JIT compilation, dataset loading, tokenization |
| Training | complete MoE forward and backward | optimizer, master-weight refresh, communication unless marked expert parallelism |
| Expert parallelism | NCCL dispatch, local MoE, reverse dispatch, routing weights, combine | router logits, top-k selection, optimizer |

- Alternatives use identical inputs in the same GPU session. Operator backends
  are measured in alternating order. Full-model serving runs separate engine
  processes in the order
  LightMoE initial, baseline, LightMoE repeat.
- Operator tables report CUDA-event median latency. Full-model serving reports
  host wall-time median. Brackets contain the interquartile range (IQR).
- A result is rejected when the absolute initial-to-repeat median deviation is
  above 5%, or host wall time divided by enclosed CUDA-event time is above 1.5.
- An unavailable backend is skipped; it is never replaced by another
  implementation under the requested name.
- Published outputs are finite and pass the stated reference comparison.
  Precision checks are not convergence results.

`T` is the number of input tokens in one MoE call. The number of token–expert
assignments is `T × top-k`, and equals `sum(M_e)` over local experts. `balanced`
distributes assignments evenly; `imbalanced` uses unequal expert counts;
`single_expert_skew` concentrates routes on expert 0; `alignment_stress`
targets tile boundaries and includes a zero-assignment expert.

Useful grouped-GEMM work is `2 × sum(M_e) × N × K`. The B200 dense FP4
specification is 9,000 TFLOP/s per GPU; 18,000 TFLOP/s assumes structured
sparsity. Peak percentages below use the 9,000 TFLOP/s dense value and are not
clock-normalized.

## Full-model vLLM serving

The model is `nvidia/DeepSeek-V4-Flash-NVFP4` revision
`e3cd60e7de98e9867116860d522499a728de1cf9`: 284B total parameters, 13B active
parameters, 256 experts, top-k 6, and bounded SwiGLU. It runs on eight B200s
with tensor parallel size 8 and an FP8 KV cache. Only the NVFP4 MoE backend is
changed; attention, scheduling, collectives, and other operators are fixed.
Prompts are fixed SWE-bench Verified samples from revision
`78f471bf655a3137b2e8a75af1501690ec009ec3`.

| Workload | Fixed batch | LightMoE initial median [IQR] | FlashInfer TensorRT-LLM median [IQR] | LightMoE repeat median [IQR] | Baseline / LightMoE |
|---|---:|---:|---:|---:|---:|
| Prefill | 8 × 1,024 input → 1 output | 221.670 [2.869] ms | 250.464 [4.300] ms | 226.927 [1.803] ms | 1.10–1.13× |
| Decode | 32 × 256 input → 256 output | 2.5014 [0.0053] s | 2.6079 [0.1274] s | 2.4960 [0.0029] s | 1.043–1.045× |

Each backend has 21 measured runs after warmup. The initial-to-repeat median
deviations are 2.37% for prefill and 0.22% for decode; maximum relative IQR is
1.72% and 4.88%, respectively. Every sample records a 1,965 MHz SM clock.
Decode produces 8,192 output tokens per run: LightMoE reaches 3,275–3,282
tokens/s and 8.894–8.898 ms per output token; FlashInfer reaches 3,141 tokens/s
and 9.266 ms per output token. Input tokens, outputs, and generated token
sequences match across backends.

This table is end-to-end. The exact DeepSeek V4 MoE layer is not included in
the operator tables below.

## Inference MoE

This boundary includes the complete expert layer described above, not only the
grouped GEMMs. FlashInfer is its CuTe DSL fused MoE starting from the same BF16
input.

| Model case | Input tokens / local experts / routing | LightMoE (µs) | FlashInfer CuTe DSL (µs) | FlashInfer / LightMoE |
|---|---:|---:|---:|---:|
| DeepSeek-V3.2 decode | 1 / 32 / balanced with an empty expert | **91.78** | 212.43 | 2.31× |
| DeepSeek-V3.2 decode batch | 32 / 32 / balanced | **179.47** | 268.93 | 1.50× |
| Kimi-K2.7 decode | 1 / 48 / single-expert skew | **98.61** | 210.82 | 2.14× |
| DeepSeek-V3.2 prefill | 2,048 / 32 / balanced | **522.06** | 1,009.49 | 1.93× |
| Kimi-K2.7 prefill | 2,048 / 48 / single-expert skew | **633.39** | 870.61 | 1.37× |

The maximum absolute initial-to-repeat median deviation is 4.66%, and the
maximum host-wall-time-to-CUDA-event-time ratio is 1.29. Compilation, packing,
calibration, and output allocation are excluded.

### FlashInfer TensorRT-LLM-generated decode kernel

Both implementations start from BF16 and include dispatch, quantization, two
expert projections, SwiGLU, routing-weight application, and combine. The
48-local-expert cases use hidden size 7,168, intermediate size 2,048, and
top-k 8.

| Local experts | Routing | Input tokens | LightMoE (µs) | FlashInfer TensorRT-LLM (µs) | Baseline / LightMoE |
|---:|---|---:|---:|---:|---:|
| 32 | balanced | 1 | **28.757** | 43.039 | 1.50× |
| 32 | balanced | 8 | **86.319** | 127.638 | 1.48× |
| 32 | balanced | 32 | **130.431** | 132.395 | 1.015× |
| 48 | balanced | 32 | **123.064** | 186.443 | 1.52× |
| 48 | single-expert skew | 16 | **28.766** | 51.762 | 1.80× |
| 48 | single-expert skew | 32 | **41.039** | 54.152 | 1.32× |

Each implementation uses CUDA Graphs and 21 samples of 100 replays. The
maximum absolute initial-to-repeat median deviation is 0.068%, and the maximum
host-wall-time-to-CUDA-event-time ratio is 1.007.

## Grouped NVFP4 GEMM

These are prepacked, balanced cases with 8,192 input tokens. LightMoE and
FlashInfer write to preallocated outputs. PyTorch `scaled_grouped_mm` allocates
its result because it has no `out=` argument.

| Model / projection | Hidden × intermediate | LightMoE (µs) | FlashInfer (µs) | PyTorch (µs) | LightMoE TFLOP/s | Dense FP4 peak |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-235B gate/up | 4,096 × 1,536 | **347.5** | 475.4 | 383.4 | 4,746 | 52.7% |
| Qwen3-235B down | 4,096 × 1,536 | **210.3** | 282.8 | 265.5 | 3,922 | 43.6% |
| DeepSeek-V3.2 gate/up | 7,168 × 2,048 | **736.8** | 1,086.7 | 788.6 | 5,223 | 58.0% |
| DeepSeek-V3.2 down | 7,168 × 2,048 | **451.7** | 617.6 | 496.9 | 4,260 | 47.3% |
| Kimi-K2.7 gate/up | 7,168 × 2,048 | **738.0** | 1,117.5 | 788.7 | 5,214 | 57.9% |
| Kimi-K2.7 down | 7,168 × 2,048 | **451.8** | 619.2 | 499.5 | 4,259 | 47.3% |

LightMoE relative IQR is 0.1–1.4% across this matrix. Published rows pass both
validity criteria above and reach sampled reference cosine of at least 0.99078.
The `plan.run(...)` entry point differs from the direct call operator by at
most 0.078%, with bitwise-identical outputs. Dense, non-grouped GEMM is reported
separately because cuBLASLt is faster overall in the measured dense matrix.

## Expert parallelism

One node uses eight B200s with EP size 8. Latency is the slowest rank's
CUDA-event median. Each value is median `[IQR]` in milliseconds.

| Scope and case | LightMoE | Transformer Engine NVFP4 | PyTorch BF16 |
|---|---:|---:|---:|
| Inference, Qwen3-30B, 128 × 1 per rank, imbalanced | 1.373 [0.120] | 2.614 [0.119] | **1.173 [0.072]** |
| Inference, DeepSeek-V3.2, 1 × 2,048 per rank, imbalanced | **2.989 [0.016]** | 4.507 [0.077] | 5.076 [0.019] |
| Inference, Kimi-K2.7, 1 × 2,048 per rank, single-expert skew | **4.310 [0.020]** | 5.817 [0.054] | 8.073 [0.021] |
| Forward and backward, Qwen3-30B, 1 × 8,192 per rank, imbalanced | **7.623 [0.074]** | 9.743 [0.083] | 10.992 [0.044] |
| Forward and backward, DeepSeek-V3.2, 1 × 2,048 per rank, imbalanced | **7.912 [0.153]** | 11.381 [0.040] | 11.585 [0.014] |

The maximum absolute initial-to-repeat median deviation is 1.01%, and the
maximum host-wall-time-to-CUDA-event-time ratio is 1.055. The Qwen decode case
is communication-bound and remains slower than PyTorch BF16. No multi-node
result is reported because 16 B200s exceed the current workspace limit.

## Single-GPU training

These rows include the complete MoE forward and backward at 8,192 input tokens.
Ranges span balanced and imbalanced routing.

| Model | LightMoE NVFP4 (ms) | Transformer Engine NVFP4 (ms) | PyTorch BF16 (ms) | LightMoE latency vs Transformer Engine |
|---|---:|---:|---:|---:|
| Qwen3-30B | **2.482–2.520** | 3.888–3.921 | 4.814–4.865 | 35.2–36.7% lower |
| DeepSeek-V3.2 | **7.383–7.408** | 9.784–9.838 | 22.373–22.443 | 24.5–24.7% lower |
| Kimi-K2.7 | **6.980–7.018** | 8.891–8.916 | 22.112–22.359 | 21.1–21.7% lower |
| MiniMax-M2 | **3.564–3.584** | 5.508–5.538 | 9.076–9.116 | 35.3% lower |

The absolute initial-to-repeat median deviation is 0.19–4.27%. Against PyTorch
BF16, output cosine is 0.9727–0.9735 and input-gradient cosine is
0.9619–0.9626. Router and expert weight gradients are finite. A Qwen3-30B
FineWeb-Edu trace measured cosine 0.996526 for output, 0.975384 for input
gradient, 0.992358 for router gradient, 0.968857 for gate/up weight gradient,
and 0.986734 for down-weight gradient.

## Reproduce

```bash
python benchmarks/nvfp4_gemm.py \
  --models all --tokens 8192 --routing balanced,imbalanced \
  --backends lightmoe,flashinfer_cutedsl,torch_scaled_grouped_mm \
  --mode prepacked --warmup 3 --iterations 20

python benchmarks/moe.py \
  --models qwen3_30b_a3b,deepseek_v3_2,kimi_k2_7,minimax_m2 \
  --tokens 8192 --routing balanced,imbalanced \
  --backends lightmoe,transformer_engine_nvfp4_fused,pytorch_bf16 \
  --scope full-layer --pass fwd_bwd --interleave-training

modal run benchmarks/modal_ci.py --matrix gemm
modal run benchmarks/modal_ci.py --matrix moe-training
modal run benchmarks/modal_ci.py::benchmark_distributed --preset inference-extended
modal run benchmarks/modal_ci.py::benchmark_distributed --preset training
```

Keep the JSONL output, profiler artifacts, environment versions, and git
revision with every published result.
