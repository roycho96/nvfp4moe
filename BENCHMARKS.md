# B200 benchmarks

Single-GPU results were measured on one NVIDIA B200 (`sm100`, 148 SMs); the
distributed section uses eight B200s in one node. Both use
`nvcr.io/nvidia/pytorch:26.07-py3` with PyTorch
`2.13.0a0+9186a08b2c.nv26.07`, CUDA 13.3, and driver 580.95.05. The grouped
GEMM comparison uses CUTLASS DSL 4.7.0 and FlashInfer 0.6.17. The training
comparison uses Transformer Engine 2.17.

Four entry points cover the release:

1. `benchmarks/nvfp4_gemm.py` measures dense and grouped NVFP4 GEMM.
2. `benchmarks/nvfp4_moe.py` measures the complete MoE layer.
3. `benchmarks/api_overhead.py` isolates the public GEMM call boundary.
4. `benchmarks/distributed_ep.py` measures the end-to-end EP expert boundary.

Missing backends are reported as skipped. No fallback is timed under another
backend's name.

## Measurement contract

| Scope | Timed | Not timed |
|---|---|---|
| Prepacked GEMM | one GEMM call writing the stated output contract | input creation, compilation, autotuning, NVFP4 packing |
| Dynamic GEMM | BF16 activation quantization and GEMM | compilation, weight packing |
| Full MoE training | dispatch, FC1, activation, FC2, probability weighting, combine, input/router/expert-weight gradients | optimizer, master-weight refresh, EP/TP communication |
| Distributed EP | gather, NCCL dispatch, local expert layer, reverse NCCL dispatch, probability weighting, combine; backward when selected | router logits/top-k, optimizer, master-weight refresh |
| Public API | `plan.run(...)` on the same plan, tensors, and output as the direct call | construction, JIT, packing, allocation |

Prepacked native and FlashInfer grouped calls write a preallocated output.
PyTorch `scaled_grouped_mm` allocates its result because it has no `out=` API.
The tables therefore compare measured API latency, not identical allocation
contracts.

Every backend starts from the same seeded BF16 sources. Compilation, input
creation, calibration, and weight packing finish before timing unless a case
is explicitly marked `dynamic`.

### Timing and rejection rules

- Runnable arms are shuffled deterministically and interleaved in one GPU
  session.
- Grouped release rows use 20 samples after a 1,000 ms stabilization phase.
  Training rows use 10 samples after three warmups.
- Tables report CUDA-event median latency. JSONL also records IQR, p10/p90,
  versions, GPU identity, row counts, FLOPs, TFLOP/s, and peak percentage.
- The first arm is rerun as a canary. A case is rejected above 5% drift.
- A run is rejected when wall time divided by the enclosing CUDA-event time
  exceeds 1.5.
- Fixed random inputs and routing seeds are shared by every arm.

Accuracy checks require a finite full output and compare up to two rows per
nonempty expert with an FP32 GEMM formed from the same BF16 sources. This is a
sampled kernel check, not a convergence claim.

## Public API overhead

`DenseGemm` and `GroupedGemm` are aliases of the native runtime classes.
`plan.run(...)` and `plan(...)` execute the same Python function body and the
same compiled kernel. The benchmark isolates the spelling of that call by
reusing one plan, one set of inputs, and one output allocation.

The accepted B200 session used 30 interleaved samples after a 1,000 ms
stabilization phase. Each CUDA event enclosed eight identical calls and was
divided by eight; the canary used the same batching. All outputs were bitwise
identical. Maximum absolute canary drift was 0.297%, and the maximum
wall-time/GPU-time ratio was 1.031.

| Case | Direct µs | Public `run` µs | Difference | IQR direct / public µs |
|---|---:|---:|---:|---:|
| Qwen3-30B FC2 dense, M=8,192 | 12.620 | 12.624 | +0.032% | 0.020 / 0.015 |
| DeepSeek-V3 FC1 dense, M=8,192 | 92.570 | 92.568 | -0.002% | 0.178 / 0.153 |
| Qwen3-30B FC2 grouped, jagged M=65,536 | 84.150 | 84.216 | +0.078% | 0.147 / 0.150 |
| DeepSeek-V3 FC2 grouped, jagged M=65,536 | 468.652 | 468.384 | -0.057% | 0.481 / 0.329 |
| DeepSeek-V3 FC2 grouped, one empty expert | 466.716 | 466.598 | -0.025% | 6.189 / 3.206 |

The largest positive difference was 0.078%, below the 2% API regression gate.
These figures validate the API boundary only; the GEMM performance tables below
compare the kernels against external implementations.

## FLOPs and peak

Useful GEMM work is `2*M*N*K`; one multiply-add counts as two FLOPs. For
grouped GEMM, `M = sum(M_e)`, where `M_e` is the row count of expert `e`.
Padding and fused activation work are not counted.

B200 dense FP4 peak is **9,000 TFLOP/s per GPU**. NVIDIA's 144 PFLOP/s HGX
B200 figure uses 2:4 structured sparsity; the corresponding eight-GPU dense
figure is 72 PFLOP/s. Peak utilization in this document is therefore
`logical TFLOP/s / 9,000`.

This percentage uses the product specification, not a clock-normalized
roofline. Modal clocks are not locked. Dynamic and full-layer rows include
non-GEMM work and do not report peak utilization.

## Jagged routing

A grouped problem is **jagged** when expert row counts differ. Expert-major
rows remain packed in one contiguous tensor, with cumulative offsets marking
boundaries. A zero row count is a valid empty expert. Every distribution keeps
`sum(M_e) = tokens * top-k` exactly.

| Routing | Definition |
|---|---|
| balanced | every expert has equal weight |
| jagged | expert `e` has weight `1 + ((17*e + 11) mod 31)` |
| hotspot | expert 0 has weight `max(1, E/2)`; all others have weight 1 |
| tail | weights repeat `1,15,16,127,128,129,255,256,257`; the last expert is empty |

Integer row counts use largest-remainder allocation. For the common
`M=65,536, E=8` shard:

| Routing | Expert row counts | CV | Empty experts |
|---|---|---:|---:|
| balanced | 8,192 × 8 | 0.000 | 0 |
| jagged | 7,350, 17,762, 9,187, 613, 11,025, 2,450, 12,862, 4,287 | 0.653 | 0 |
| hotspot | 23,831, 5,958, 5,958, 5,958, 5,958, 5,958, 5,958, 5,957 | 0.722 | 0 |
| tail | 98, 1,465, 1,563, 12,404, 12,502, 12,599, 24,905, 0 | 1.018 | 1 |

Synthetic top-k assignments are sampled without replacement. Every JSONL row
stores the complete row-count array, min/max, coefficient of variation, empty
expert count, and 128-row alignment count.

## Model shapes

| Model | Hidden | Intermediate | Global experts | Top-k | GEMM EP size / local E | Training EP size / local E |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | 2,048 | 768 | 128 | 8 | 16 / 8 | 8 / 16 |
| Qwen3-235B-A22B | 4,096 | 1,536 | 128 | 8 | 16 / 8 | 8 / 16 |
| Gemma 4 26B A4B | 2,816 | 704→768 | 128 | 8 | 16 / 8 | 8 / 16 |
| DeepSeek-V3.2 | 7,168 | 2,048 | 256 | 8 | 32 / 8 | 8 / 32 |
| Kimi-K2.7 | 7,168 | 2,048 | 384 | 8 | 48 / 8 | 24 / 16 |
| MiniMax-M2 | 3,072 | 1,536 | 256 | 8 | 32 / 8 | 8 / 32 |
| Llama 4 Scout | 5,120 | 8,192 | 16 | 1 | 2 / 8 | 8 / 2 |

These are model-derived operator shapes with synthetic activations and
routing. They are not full checkpoint or dataset runs.

## Distributed expert parallelism

This benchmark uses one 8×B200 node with EP size 8. `B/rank` is the batch size
on each rank, and each rank contributes `B/rank * S` input tokens. Global input
tokens multiply that value by eight; routed rows additionally multiply by the
model's top-k.

Latency is the slowest rank's enclosing CUDA-event time. Token gather, NCCL
dispatch, expert IDs, the local expert layer, reverse dispatch, probability
weighting, and combine are timed. Routing logits and top-k selection are not.
Uneven messages use one fixed-capacity `all_to_all_single`; the table reports
the padding overhead rather than hiding it. Values are `median [IQR]` in ms.

| Model and case | Native | TE NVFP4 | Torch BF16 | TE/native | BF16/native | Transport / padding |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-30B, B/rank 128 × S1, jagged | 1.373 [0.120] | 2.614 [0.119] | **1.173 [0.072]** | 1.90× | 0.85× | 0.513 / 25.0% |
| Qwen3-30B, B/rank 1 × S8192, jagged | **3.236 [0.019]** | 4.459 [0.141] | 5.065 [0.015] | 1.38× | 1.56× | 1.727 / 13.5% |
| Qwen3-235B, B/rank 1 × S2048, jagged | **1.834 [0.006]** | 2.546 [0.059] | 2.968 [0.011] | 1.39× | 1.62× | 0.966 / 13.0% |
| DeepSeek-V3.2, B/rank 1 × S2048, jagged | **2.989 [0.016]** | 4.507 [0.077] | 5.076 [0.019] | 1.51× | 1.70× | 1.476 / 4.8% |
| Kimi-K2.7, B/rank 1 × S2048, hotspot | **4.310 [0.020]** | 5.817 [0.054] | 8.073 [0.021] | 1.35× | 1.87× | 2.265 / 87.5% |
| MiniMax-M2, B/rank 1 × S2048, jagged | **1.492 [0.007]** | 2.426 [0.101] | 2.375 [0.012] | 1.63× | 1.59× | 0.790 / 4.8% |
| Llama 4 Scout, B/rank 1 × S2048, tail | **1.082 [0.020]** | 1.657 [0.033] | 2.146 [0.017] | 1.53× | 1.98× | 0.613 / 162.1% |

The small Qwen decode case is communication and launch dominated, so native
does not beat BF16 there. The larger cases amortize that fixed cost and native
leads both comparison arms.

Forward/backward includes the autograd path and the two reverse collectives.

| Model and case | Native | TE NVFP4 | Torch BF16 | TE/native | BF16/native |
|---|---:|---:|---:|---:|---:|
| Qwen3-30B, B/rank 1 × S8192, balanced | **7.225 [0.021]** | 9.125 [0.468] | 10.122 [0.021] | 1.26× | 1.40× |
| Qwen3-30B, B/rank 1 × S8192, jagged | **7.623 [0.074]** | 9.743 [0.083] | 10.992 [0.044] | 1.28× | 1.44× |
| DeepSeek-V3.2, B/rank 1 × S2048, jagged | **7.912 [0.153]** | 11.381 [0.040] | 11.585 [0.014] | 1.44× | 1.46× |

Inference rows use 20 samples; training rows use 10. Arms are deterministically
interleaved in one session. Maximum accepted canary drift was 1.01%, and the
largest wall/event ratio was 1.055. All eight ranks produced finite outputs and
gradients. Across the three training rows, native's worst-rank minimum cosine
against BF16 was 0.97247 for output, 0.96144 for input gradient, 0.97262 for
router-probability gradient, 0.95992 for gate/up weight gradient, and 0.96600
for down-weight gradient.

These results cover one node. A two-node B200 run requires 16 GPUs, above the
current Modal workspace limit of 10 concurrent GPUs, so no multi-node number is
reported.

## Grouped NVFP4 GEMM

The table reports balanced routing at 8,192 tokens. Llama 4 has fewer routed
rows because its top-k is 1.

| Model | Routed M | FC1 us / TFLOP/s / peak | FC2 us / TFLOP/s / peak |
|---|---:|---:|---:|
| Qwen3-30B | 65,536 | 111.8 / 3,687 / 41.0% | 84.6 / 2,438 / 27.1% |
| Qwen3-235B | 65,536 | 347.5 / 4,746 / 52.7% | 210.3 / 3,922 / 43.6% |
| Gemma 4 26B | 65,536 | 146.7 / 3,866 / 43.0% | 113.2 / 2,503 / 27.8% |
| DeepSeek-V3.2 | 65,536 | 736.8 / 5,223 / 58.0% | 451.7 / 4,260 / 47.3% |
| Kimi-K2.7 | 65,536 | 738.0 / 5,214 / 57.9% | 451.8 / 4,259 / 47.3% |
| MiniMax-M2 | 65,536 | 278.6 / 4,440 / 49.3% | 163.1 / 3,792 / 42.1% |
| Llama 4 Scout | 8,192 | 314.8 / 4,367 / 48.5% | 156.9 / 4,379 / 48.7% |

Backend latency is CUDA-event median microseconds for the same balanced cases.
PyTorch uses `scaled_grouped_mm` and is measured only when every group meets
its 128-row alignment requirement.

| Model | FC1 native / FlashInfer / PyTorch | FC2 native / FlashInfer / PyTorch |
|---|---:|---:|
| Qwen3-30B | **111.8** / 142.4 / 134.6 | **84.6** / 90.5 / 125.4 |
| Qwen3-235B | **347.5** / 475.4 / 383.4 | **210.3** / 282.8 / 265.5 |
| Gemma 4 26B | **146.7** / 180.7 / 169.2 | **113.2** / 123.3 / 170.5 |
| DeepSeek-V3.2 | **736.8** / 1,086.7 / 788.6 | **451.7** / 617.6 / 496.9 |
| Kimi-K2.7 | **738.0** / 1,117.5 / 788.7 | **451.8** / 619.2 / 499.5 |
| MiniMax-M2 | **278.6** / 368.9 / 312.7 | **163.1** / 215.5 / 206.2 |
| Llama 4 Scout | **314.8** / 437.4 / 357.6 | **156.9** / 205.1 / 179.4 |

Native IQR was 0.1–1.4% of the median across these 14 rows. All shown cases
passed the canary and wall/GPU gates and reached sampled reference cosine of at
least 0.99078. A Llama 4 jagged FC1 run drifted 10.9% and was rejected; it is
not included in either table.

## Dense NVFP4 GEMM

Dense mode measures `C = A @ B.T` without an expert dimension. It compares the
native kernel with cuBLASLt through `torch.nn.functional.scaled_mm` at
`M={128,512,2048,8192}` for Qwen3-30B, Qwen3-235B, DeepSeek-V3.2, and Llama 4
FC1/FC2 shapes.

cuBLASLt was faster overall in the measured matrix. Native led a small subset
of large-K cases, so dense performance is not presented as a universal SOTA
claim.

## Single-GPU full MoE training

Times include the complete single-GPU forward and backward boundary defined
above. Results use 8,192 tokens and top-k sampling without replacement.

| Model / routing | Native NVFP4 | TE fused NVFP4 | Torch BF16 | Native vs TE |
|---|---:|---:|---:|---:|
| Qwen3-30B / balanced | **2.520 ms** | 3.888 ms | 4.814 ms | 35.2% faster |
| Qwen3-30B / jagged | **2.482 ms** | 3.921 ms | 4.865 ms | 36.7% faster |
| DeepSeek-V3.2 / balanced | **7.383 ms** | 9.784 ms | 22.373 ms | 24.5% faster |
| DeepSeek-V3.2 / jagged | **7.408 ms** | 9.838 ms | 22.443 ms | 24.7% faster |
| Kimi-K2.7 / balanced | **6.980 ms** | 8.916 ms | 22.112 ms | 21.7% faster |
| Kimi-K2.7 / jagged | **7.018 ms** | 8.891 ms | 22.359 ms | 21.1% faster |
| MiniMax-M2 / balanced | **3.584 ms** | 5.538 ms | 9.116 ms | 35.3% faster |
| MiniMax-M2 / jagged | **3.564 ms** | 5.508 ms | 9.076 ms | 35.3% faster |

Canary drift was 0.19–4.27%, so every published training row passed the 5%
gate.

## Precision

Against the Torch BF16 layer, native output cosine was 0.9727–0.9735 and input
gradient cosine was 0.9619–0.9626. Router and expert weight gradients were
finite in every case. A separate Qwen3-30B FineWeb-Edu trace measured:

| Tensor | Cosine similarity | Relative L2 |
|---|---:|---:|
| Output | 0.996526 | 0.08444 |
| Input gradient | 0.975384 | 0.22193 |
| Router gradient | 0.992358 | 0.12360 |
| Gate/up weight gradient | 0.968857 | 0.25141 |
| Down weight gradient | 0.986734 | 0.16351 |

These are layer checks, not convergence results.

## Reproduce

```bash
python benchmarks/nvfp4_gemm.py --list --suite full
python benchmarks/nvfp4_moe.py --list --suite full

python benchmarks/nvfp4_gemm.py \
  --models all --tokens 8192 --routing balanced,jagged \
  --backends native,flashinfer_cutedsl,torch_scaled_grouped_mm \
  --mode prepacked --warmup 3 --iterations 20

python benchmarks/nvfp4_moe.py \
  --models qwen3_30b_a3b,deepseek_v3_2,kimi_k2_7,minimax_m2 \
  --tokens 8192 --routing balanced,jagged \
  --backends native,te_nvfp4_fused,torch_bf16 \
  --scope full-layer --pass fwd_bwd --interleave-training

modal run benchmarks/modal_ci.py
modal run benchmarks/modal_ci.py --benchmark-smoke
modal run benchmarks/modal_ci.py --matrix gemm
modal run benchmarks/modal_ci.py --matrix moe-training
modal run benchmarks/modal_ci.py --grouped api
modal run benchmarks/modal_ci.py::benchmark_distributed --preset inference
modal run benchmarks/modal_ci.py::benchmark_distributed --preset inference-extended
modal run benchmarks/modal_ci.py::benchmark_distributed --preset training
```

Keep the JSONL output, profiler artifacts, environment versions, and git
revision with newly published results.

## Methodology references

- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/testing/bench.py) separates kernel timing from setup and uses CUDA/Kineto measurements.
- [TileLink](https://arxiv.org/abs/2503.20313) reports operator, layer, and end-to-end scopes separately with model-derived shapes.
- [Comet](https://arxiv.org/abs/2502.19811) varies token count, expert count, top-k, EP size, and token imbalance in MoE evaluation.
- [FLUX](https://arxiv.org/abs/2406.06858) separates operation-level measurements from full training and inference results.
- [NVIDIA DGX B200 specifications](https://www.nvidia.com/en-gb/data-center/dgx-b200/) provide the dense and structured-sparse FP4 peak values.

No timing from these sources is copied into the result tables. A baseline is
reported only when its implementation ran in the same session and measured
the stated boundary.
