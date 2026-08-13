# B200 benchmark results

Results were measured on NVIDIA B200 (`sm100`) with NGC PyTorch 26.07 and
CUDA 13. The standalone grouped comparison uses CUTLASS DSL 4.7, FlashInfer
0.6.17, and PyTorch 2.13 nightly. The training comparison uses Transformer
Engine 2.17.

Two public entry points cover the project:

1. `benchmarks/nvfp4_gemm.py` — dense and grouped NVFP4 GEMM
2. `benchmarks/nvfp4_moe.py` — expert-core and complete MoE layer

Both scripts emit JSON with `--list` and JSONL while running. Missing backends
are reported as skipped; no fallback is timed under another backend's name.

## ⚙️ Method

- Inputs are fixed for every backend in a case.
- Timed arms are interleaved in one GPU session.
- Tables report CUDA-event median latency. Raw JSONL also includes IQR and
  p10/p90.
- The first arm is rerun as a canary. A case is rejected above 5% drift.
- A run is rejected when wall time divided by summed GPU time exceeds 1.5.
- Compilation, input creation, calibration, and weight packing happen before
  timing unless the case is marked `dynamic`.

Synthetic top-k assignments are sampled without replacement. The training
quick suite also keeps more local experts than top-k, so balanced and skewed
routing remain distinct. Standalone GEMM retains the common eight-expert
shard:

| model | hidden | intermediate | global experts | top-k | GEMM EP size / local E | training EP size / local E |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | 2,048 | 768 | 128 | 8 | 16 / 8 | 8 / 16 |
| Qwen3-235B-A22B | 4,096 | 1,536 | 128 | 8 | 16 / 8 | 8 / 16 |
| Gemma 4 26B A4B | 2,816 | 704→768 | 128 | 8 | 16 / 8 | 8 / 16 |
| DeepSeek-V3.2 | 7,168 | 2,048 | 256 | 8 | 32 / 8 | 8 / 32 |
| Kimi-K2.7 | 7,168 | 2,048 | 384 | 8 | 48 / 8 | 24 / 16 |
| MiniMax-M2 | 3,072 | 1,536 | 256 | 8 | 32 / 8 | 8 / 32 |
| Llama 4 Scout | 5,120 | 8,192 | 16 | 1 | 2 / 8 | 8 / 2 |

## 🧱 Grouped NVFP4 GEMM

The prepacked test excludes activation quantization and interleaves native,
FlashInfer `grouped_gemm_nt_masked`, and PyTorch `scaled_grouped_mm`. PyTorch
is measured only for balanced routing because its NVFP4 grouped path requires
128-row-aligned groups.

Latency below is in microseconds for 65,536 routed rows:

| model | FC1 native / FlashInfer / PyTorch | FC2 native / FlashInfer / PyTorch |
|---|---:|---:|
| Qwen3-30B | **123.7** / 138.9 / 131.5 | 92.5 / **90.5** / 124.0 |
| Qwen3-235B | **367.8** / 455.5 / **367.8** | 258.4 / 271.1 / **253.6** |
| Gemma 4 26B | **154.2** / 174.8 / 164.2 | 123.6 / **118.9** / 162.4 |
| DeepSeek-V3.2 | **711.0** / 1,096.4 / 771.9 | **432.0** / 595.4 / 479.1 |
| Kimi-K2.7 | **738.0** / 1,131.9 / 792.4 | **448.4** / 619.9 / 497.6 |
| MiniMax-M2 | 311.6 / 356.3 / **305.6** | 197.0 / 208.1 / **194.9** |
| Llama 4 Scout | **331.0** / 424.6 / 349.3 | **162.5** / 202.2 / 179.7 |

The DeepSeek and Kimi rows were remeasured on August 13 after the long-hidden
scheduler update. Native led FlashInfer in all eight balanced/jagged cases and
PyTorch in all four aligned balanced cases. The balanced advantage was
6.9–9.9% over PyTorch and 27.5–35.2% over FlashInfer. PyTorch remains skipped
for jagged routing because it requires 128-row-aligned groups.

For K=2,048 FC2, a direct scheduler A/B measured static at 394.0 us versus
dynamic at 396.9 us on balanced routing, and 394.0 versus 396.6 us on jagged
routing. The standalone runtime now selects static scheduling for K≤2,048 and
keeps dynamic scheduling for longer reductions. Explicit scheduler selection
remains available to the expert-layer runtime.

All retained rows used 20 samples after 200 ms stabilization, had a valid
native canary, and reached sampled BF16-reference cosine of at least 0.99078.

## 🔹 Dense NVFP4 GEMM

Dense mode measures `C = A @ B.T` without an expert dimension. It compares the
native kernel with the NGC image's cuBLASLt path through
`torch.nn.functional.scaled_mm`. The native runner autotunes CTA tiles before
the timed region and exposes both allocating and preallocated output APIs.

The full suite covers FC1 and FC2 from Qwen3-30B, Qwen3-235B, DeepSeek-V3.2,
and Llama 4 at M=128, 512, 2,048, and 8,192. cuBLASLt was faster overall in the
measured matrix; native led a small subset of large-K cases. Dense results are
therefore reported as a comparison, not a universal SOTA claim.

## 🔁 Full MoE training

The timed region contains dispatch, FC1, activation, FC2, probability
weighting, combine, input gradient, router-weight gradient, and both expert
weight gradients. Transformer Engine uses its fused NVFP4
`GroupedLinear → SwiGLU → GroupedLinear` training graph. Torch BF16 is included
as a precision and performance reference.

Corrected 8K results from August 12, 2026:

| model / routing | native NVFP4 | TE fused NVFP4 | Torch BF16 | native vs TE |
|---|---:|---:|---:|---:|
| Qwen3-30B / balanced | **2.520 ms** | 3.888 ms | 4.814 ms | 35.2% faster |
| Qwen3-30B / jagged | **2.482 ms** | 3.921 ms | 4.865 ms | 36.7% faster |
| DeepSeek-V3.2 / balanced | **7.383 ms** | 9.784 ms | 22.373 ms | 24.5% faster |
| DeepSeek-V3.2 / jagged | **7.408 ms** | 9.838 ms | 22.443 ms | 24.7% faster |
| Kimi-K2.7 / balanced | **6.980 ms** | 8.916 ms | 22.112 ms | 21.7% faster |
| Kimi-K2.7 / jagged | **7.018 ms** | 8.891 ms | 22.359 ms | 21.1% faster |
| MiniMax-M2 / balanced | **3.584 ms** | 5.538 ms | 9.116 ms | 35.3% faster |
| MiniMax-M2 / jagged | **3.564 ms** | 5.508 ms | 9.076 ms | 35.3% faster |

Each row uses 10 interleaved samples after three warmups. Canary drift was
0.19–4.27%, so every row passed the 5% gate.

An earlier matrix allowed duplicate expert ids inside a token. The native arm
processed all 65,536 routes while TE's mask collapsed duplicates to 43,019
unique routes. Those results were invalid and are not used above. The corrected
generator samples top-k experts without replacement and records actual counts
in every result.

For EP shards with at most 32 local experts, a warp-ballot scatter replaced one
thread-per-expert serial scans. At local E=8, direct same-session measurement
reduced dispatch from 111 to 17 microseconds. Parallelizing the histogram then
reduced local-E=32 dispatch from 25.73 to 23.09 microseconds. Both comparisons
were bitwise identical. The remaining backward control cost is primarily
router-gradient dot reduction; a packed implementation increased latency and
was rejected.

## 🧪 Precision

Against the Torch BF16 layer in the final run, native output cosine was
0.9727–0.9735 and input-gradient cosine was 0.9619–0.9626. Router and expert
weight gradients were finite in every case. A separate Qwen3-30B FineWeb-Edu
trace measured:

| tensor | cosine similarity | relative L2 |
|---|---:|---:|
| output | 0.996526 | 0.08444 |
| input gradient | 0.975384 | 0.22193 |
| router gradient | 0.992358 | 0.12360 |
| gate/up weight gradient | 0.968857 | 0.25141 |
| down weight gradient | 0.986734 | 0.16351 |

These are layer-level numerical checks, not convergence results.

## 🧭 Coverage

The full suites cover:

- tokens: 1, 128, 512, 2,048, 8,192, and 16,384
- balanced, jagged, hotspot, boundary-tail, and empty-expert routing
- every registered EP size
- FC1, FC2, dgrad, and K-grouped wgrad
- prepacked and dynamic activation quantization
- expert-core and full-layer forward/forward-backward timing

`--source trace` replays a local EP shard from a `.pt` file containing
`expert_input`, `topk_index`, `topk_weight`, `gate_up_weight`, and
`down_weight`.

## 🚀 Reproduce

Inspect the matrices without a GPU:

```bash
python benchmarks/nvfp4_gemm.py --list --suite full
python benchmarks/nvfp4_moe.py --list --suite full
```

Run standalone grouped and dense comparisons:

```bash
python benchmarks/nvfp4_gemm.py \
  --models all --tokens 8192 --routing balanced,jagged \
  --backends native,flashinfer_cutedsl,torch_scaled_grouped_mm \
  --mode prepacked --warmup 3 --iterations 20

python benchmarks/nvfp4_gemm.py \
  --workload dense --suite full \
  --models qwen3_30b_a3b,qwen3_235b_a22b,deepseek_v3_2,llama4_scout \
  --tokens 128,512,2048,8192 --projections fc1,fc2 \
  --backends native,cublaslt --mode prepacked
```

Run the full training comparison:

```bash
python benchmarks/nvfp4_moe.py \
  --models qwen3_30b_a3b,deepseek_v3_2,kimi_k2_7,minimax_m2 \
  --tokens 8192 --routing balanced,jagged \
  --backends native,te_nvfp4_fused,torch_bf16 \
  --scope full-layer --pass fwd_bwd --interleave-training
```

Run release checks on Modal:

```bash
modal run benchmarks/modal_ci.py
modal run benchmarks/modal_ci.py --benchmark-smoke
modal run benchmarks/modal_ci.py --matrix gemm
modal run benchmarks/modal_ci.py --matrix moe-training
modal run benchmarks/modal_ci.py --grouped focused
modal run benchmarks/modal_ci.py --dense full
```

Keep raw JSONL, profiler artifacts, environment versions, and the git revision
with any newly published results.
