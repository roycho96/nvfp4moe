# B200 benchmarks

This document reports the release results and the boundary measured by each
table. Operator benchmarks use one NVIDIA B200 (`sm100`, 148 SMs). Distributed
benchmarks use eight B200s in one node.

Single-GPU and EP runs use `nvcr.io/nvidia/pytorch:26.07-py3`, PyTorch
`2.13.0a0+9186a08b2c.nv26.07`, CUDA 13.3, driver 580.95.05, CUTLASS DSL 4.7.0,
FlashInfer 0.6.17, and Transformer Engine 2.17. The full-model vLLM run uses
vLLM 0.27.1, PyTorch 2.13.0+cu130, and FlashInfer 0.6.16.post3.

## Measurement rules

| Scope | Timed | Excluded |
|---|---|---|
| Grouped GEMM | one prepacked GEMM call | allocation, compilation, autotuning, packing |
| Inference MoE | dispatch, BF16 quantization, FC1, activation, requantization, FC2, weighting, combine | routing logits/top-k, calibration, compilation, weight packing |
| Full-model serving | one warmed `LLM.generate` call | model load, JIT, dataset loading, tokenization |
| Training | full MoE forward and backward | optimizer, master-weight refresh, communication unless marked EP |
| Distributed EP | NCCL dispatch, local MoE, reverse dispatch, weighting, combine | routing logits/top-k, optimizer |

- Alternatives are measured in the same GPU session with fixed inputs.
  Operator arms are interleaved; full-model serving uses native → baseline →
  native canary in separate engine processes.
- Tables report CUDA-event median latency, except full-model serving, which
  reports `LLM.generate` wall time. IQR and p10/p90 are retained in JSONL.
- A result is rejected when native canary drift exceeds 5% or wall time divided
  by enclosed GPU time exceeds 1.5.
- Missing backends are skipped. No fallback is reported under another name.
- Outputs must be finite and pass the stated FP32 or BF16 reference check.
  Precision results are operator checks, not convergence results.

`T` is the input-token count for one MoE call. Expert GEMMs receive
`T * top-k` routed rows. A grouped problem is jagged when expert row counts
differ; zero-row experts are valid and `sum(M_e) = T * top-k` is preserved.
Balanced routing spreads rows evenly, hotspot routing favors expert 0, and tail
routing exercises alignment boundaries and an empty expert.

Useful GEMM work is `2*M*N*K`, with `M = sum(M_e)`. B200 dense FP4 peak is
9,000 TFLOP/s per GPU; the 18,000 TFLOP/s figure assumes 2:4 sparsity. Reported
peak percentages use the 9,000 TFLOP/s dense specification and are not
clock-normalized.

## Full-model vLLM serving

This run serves `nvidia/DeepSeek-V4-Flash-NVFP4` revision
`e3cd60e7de98e9867116860d522499a728de1cf9` on eight B200s with TP8 and an FP8
KV cache. The model has 284B total and 13B active parameters, with E256/top-6
and bounded SwiGLU. The run replaces only the NVFP4 MoE backend; attention,
vLLM scheduling, collectives, and other operators are unchanged. Prompts are
fixed samples from SWE-bench Verified revision
`78f471bf655a3137b2e8a75af1501690ec009ec3`.

| Workload | Fixed batch | Native A p50 [IQR] | FlashInfer TRT p50 [IQR] | Native canary p50 [IQR] | Baseline/native |
|---|---:|---:|---:|---:|---:|
| Prefill | 8 × 1,024 input → 1 output | 221.670 [2.869] ms | 250.464 [4.300] ms | 226.927 [1.803] ms | 1.10–1.13× |
| Decode | 32 × 256 input → 256 output | 2.5014 [0.0053] s | 2.6079 [0.1274] s | 2.4960 [0.0029] s | 1.043–1.045× |

Each arm has 21 measured runs after a stable warmup. Prefill and decode canary
drift are 2.37% and 0.22%; maximum relative IQR is 1.72% and 4.88%. Every timed
sample records a 1,965 MHz SM clock. Decode produces 8,192 tokens per run:
native reaches 3,275–3,282 tokens/s and 8.894–8.898 ms TPOT versus 3,141
tokens/s and 9.266 ms for FlashInfer. Decode input, output, and the forced token
trajectory match across all arms.

This is a full-model result. The exact DeepSeek V4 E256/top-6 MoE layer has not
been isolated in the operator tables below.

## Inference MoE

These tables measure the complete inference MoE boundary defined above, not
standalone GEMM. `FI BF16` is FlashInfer's CuTeDSL fused MoE starting from the
same BF16 input as native.

| Model case | T / local E / routing | Native µs | FI BF16 µs | FI/native |
|---|---:|---:|---:|---:|
| DeepSeek-V3.2 decode | 1 / 32 / empty | **91.78** | 212.43 | 2.31× |
| DeepSeek-V3.2 decode batch | 32 / 32 / balanced | **179.47** | 268.93 | 1.50× |
| Kimi-K2.7 decode | 1 / 48 / hotspot | **98.61** | 210.82 | 2.14× |
| DeepSeek-V3.2 prefill | 2,048 / 32 / balanced | **522.06** | 1,009.49 | 1.93× |
| Kimi-K2.7 prefill | 2,048 / 48 / hotspot | **633.39** | 870.61 | 1.37× |

Maximum accepted canary drift is 4.66% and maximum wall/GPU ratio is 1.29.
Compilation, packing, calibration, and output allocation are excluded.

### TRT-LLM generated decode

This is the stronger FlashInfer TRT-LLM generated baseline. Both columns start
from BF16 and include dispatch, quantization, two expert projections, SwiGLU,
weighting, and combine. The E48 cases use `H=7,168`, `I=2,048`, and top-k 8.

| Shape | Routing | T | Native µs | TRT µs | TRT/native |
|---|---|---:|---:|---:|---:|
| local E32 | balanced | 1 | **28.757** | 43.039 | 1.50× |
| local E32 | balanced | 8 | **86.319** | 127.638 | 1.48× |
| local E32 | balanced | 32 | **130.431** | 132.395 | 1.015× |
| local E48 | balanced | 32 | **123.064** | 186.443 | 1.52× |
| local E48 | hotspot | 16 | **28.766** | 51.762 | 1.80× |
| local E48 | hotspot | 32 | **41.039** | 54.152 | 1.32× |

The arms use CUDA Graphs and 21 samples of 100 replays. Maximum canary drift is
0.068% and maximum wall/GPU ratio is 1.007.

## Grouped NVFP4 GEMM

These are prepacked balanced cases at 8,192 tokens. Native and FlashInfer write
a preallocated output. PyTorch `scaled_grouped_mm` allocates its result because
it has no `out=` API.

| Model / projection | H × I | Native µs | FlashInfer µs | PyTorch µs | Native TFLOP/s | Dense peak |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-235B FC1 | 4,096 × 1,536 | **347.5** | 475.4 | 383.4 | 4,746 | 52.7% |
| Qwen3-235B FC2 | 4,096 × 1,536 | **210.3** | 282.8 | 265.5 | 3,922 | 43.6% |
| DeepSeek-V3.2 FC1 | 7,168 × 2,048 | **736.8** | 1,086.7 | 788.6 | 5,223 | 58.0% |
| DeepSeek-V3.2 FC2 | 7,168 × 2,048 | **451.7** | 617.6 | 496.9 | 4,260 | 47.3% |
| Kimi-K2.7 FC1 | 7,168 × 2,048 | **738.0** | 1,117.5 | 788.7 | 5,214 | 57.9% |
| Kimi-K2.7 FC2 | 7,168 × 2,048 | **451.8** | 619.2 | 499.5 | 4,259 | 47.3% |

Native IQR is 0.1–1.4% across the release matrix. Published rows pass the
canary and wall/GPU gates and reach sampled reference cosine of at least
0.99078. Dense, non-grouped GEMM is not presented as SOTA: cuBLASLt is faster
overall in the measured matrix. Public `plan.run(...)` overhead is at most
0.078%, with bitwise-identical outputs.

## Distributed EP

This benchmark uses one node with eight B200s and EP size 8. Latency is the
slowest rank's CUDA-event median. `B/rank × S` is batch and sequence length per
rank. Values are median `[IQR]` in ms.

| Scope and case | Native | TE NVFP4 | Torch BF16 |
|---|---:|---:|---:|
| Inference, Qwen3-30B, 128 × 1, jagged | 1.373 [0.120] | 2.614 [0.119] | **1.173 [0.072]** |
| Inference, DeepSeek-V3.2, 1 × 2,048, jagged | **2.989 [0.016]** | 4.507 [0.077] | 5.076 [0.019] |
| Inference, Kimi-K2.7, 1 × 2,048, hotspot | **4.310 [0.020]** | 5.817 [0.054] | 8.073 [0.021] |
| Forward/backward, Qwen3-30B, 1 × 8,192, jagged | **7.623 [0.074]** | 9.743 [0.083] | 10.992 [0.044] |
| Forward/backward, DeepSeek-V3.2, 1 × 2,048, jagged | **7.912 [0.153]** | 11.381 [0.040] | 11.585 [0.014] |

Maximum canary drift is 1.01% and maximum wall/GPU ratio is 1.055. The small
Qwen decode case is communication-bound and remains slower than Torch BF16.
No multi-node result is reported because 16 B200s exceed the current workspace
limit.

## Single-GPU training

These rows include the complete MoE forward and backward at 8,192 tokens.
Ranges cover balanced and jagged routing.

| Model | Native NVFP4 ms | TE NVFP4 ms | Torch BF16 ms | Native latency vs TE |
|---|---:|---:|---:|---:|
| Qwen3-30B | **2.482–2.520** | 3.888–3.921 | 4.814–4.865 | 35.2–36.7% lower |
| DeepSeek-V3.2 | **7.383–7.408** | 9.784–9.838 | 22.373–22.443 | 24.5–24.7% lower |
| Kimi-K2.7 | **6.980–7.018** | 8.891–8.916 | 22.112–22.359 | 21.1–21.7% lower |
| MiniMax-M2 | **3.564–3.584** | 5.508–5.538 | 9.076–9.116 | 35.3% lower |

Canary drift is 0.19–4.27%. Against Torch BF16, output cosine is
0.9727–0.9735 and input-gradient cosine is 0.9619–0.9626. Router and expert
weight gradients are finite. A Qwen3-30B FineWeb-Edu trace measured cosine
0.996526 for output, 0.975384 for input gradient, 0.992358 for router gradient,
0.968857 for gate/up weight gradient, and 0.986734 for down-weight gradient.

## Reproduce

```bash
python benchmarks/nvfp4_gemm.py \
  --models all --tokens 8192 --routing balanced,jagged \
  --backends native,flashinfer_cutedsl,torch_scaled_grouped_mm \
  --mode prepacked --warmup 3 --iterations 20

python benchmarks/nvfp4_moe.py \
  --models qwen3_30b_a3b,deepseek_v3_2,kimi_k2_7,minimax_m2 \
  --tokens 8192 --routing balanced,jagged \
  --backends native,te_nvfp4_fused,torch_bf16 \
  --scope full-layer --pass fwd_bwd --interleave-training

modal run benchmarks/modal_ci.py --matrix gemm
modal run benchmarks/modal_ci.py --matrix moe-training
modal run benchmarks/modal_ci.py::benchmark_distributed --preset inference-extended
modal run benchmarks/modal_ci.py::benchmark_distributed --preset training
```

Keep the JSONL output, profiler artifacts, environment versions, and git
revision with newly published results.
