# B200 benchmark results

Results were measured on NVIDIA B200 (`sm100`, 148 SMs) with NGC PyTorch 26.07,
PyTorch `2.13.0a0+9186a08b2c.nv26.07`, container CUDA 13.3, and driver
580.95.05 in CUDA minor-version compatibility mode. The standalone grouped
comparison uses CUTLASS DSL 4.7.0 and FlashInfer 0.6.17. The training comparison
uses Transformer Engine 2.17.

Two public entry points cover the project:

1. `benchmarks/nvfp4_gemm.py` — dense and grouped NVFP4 GEMM
2. `benchmarks/nvfp4_moe.py` — expert-core and complete MoE layer

Both scripts emit JSON with `--list` and JSONL while running. Missing backends
are reported as skipped; no fallback is timed under another backend's name.

## ⚙️ Method

- GEMM work is `2*M*N*K`; one multiply-add counts as two FLOPs. Grouped GEMM
  uses `M = sum(expert rows)`. This is useful, or logical, work before tile
  padding.
- B200 peak utilization uses 9,000 dense FP4 TFLOP/s per GPU. NVIDIA specifies
  72 dense FP4 PFLOP/s for the eight-GPU DGX B200; the 144 PFLOP/s headline is
  the structured-sparse figure. See the
  [DGX B200 specification](https://www.nvidia.com/en-gb/data-center/dgx-b200/).
- Peak utilization is `logical TFLOP/s / 9,000`. It is a product-spec ceiling,
  not a clock-normalized roofline. Modal clocks are not locked.
- The JSONL also records native tile-rounded FLOPs and padding overhead. These
  are not used in the headline peak percentage because padded work is not model
  work.
- Every backend starts from the same seeded BF16 source tensors. Backend-native
  NVFP4 packing and scale-layout conversion happen before timing; packed byte
  identity across different layouts is not assumed.
- Runnable arms are deterministically shuffled per iteration in one GPU
  session. The release table used a 1,000 ms stabilization phase before both
  the first measurement and canary; the CLI default is 200 ms.
- Tables report CUDA-event median latency. JSONL also contains IQR, p10/p90,
  package versions, GPU identity, logical FLOPs, TFLOP/s, and peak percentage.
- The first arm is rerun as a canary. A case is rejected above 5% drift.
- A run is rejected when wall time divided by summed GPU time exceeds 1.5.
- Compilation, input creation, calibration, and weight packing happen before
  timing unless the case is marked `dynamic`.

For prepacked grouped forward, the native and FlashInfer calls write a
preallocated output. PyTorch `scaled_grouped_mm` has no `out=` API, so its timed
call allocates the return tensor. The output contract is recorded on every
result. The accuracy check verifies the complete output is finite and compares
up to two rows per nonempty expert with an FP32 GEMM formed from the same BF16
sources. It is a sampled kernel check, not a model-accuracy claim.

Dynamic rows include BF16 activation quantization and report equivalent logical
TFLOP/s, but no peak utilization. Full MoE rows include non-GEMM work and are
reported only as end-to-end latency.

### Routing definitions

The grouped table uses model dimensions and synthetic routing; it is not a
captured production trace. Counts are deterministic and preserve exactly
`tokens * top-k` rows. Integer allocation takes
`floor(rows * weight / sum(weights))`, then assigns the remainder by largest
fractional remainder.

- `balanced`: every expert has weight 1.
- `jagged`: expert `e` has weight `1 + ((17*e + 11) mod 31)`.
- `hotspot`: expert 0 has weight `max(1, E/2)` and all others have weight 1.
- `tail`: weights repeat `1, 15, 16, 127, 128, 129, 255, 256, 257`; the last
  expert is forced empty.

For the common `M=65,536, E=8` shard, the exact distributions are:

| routing | expert row counts | min / max | CV | empty E |
|---|---|---:|---:|---:|
| balanced | 8,192 × 8 | 8,192 / 8,192 | 0.000 | 0 |
| jagged | 7,350, 17,762, 9,187, 613, 11,025, 2,450, 12,862, 4,287 | 613 / 17,762 | 0.653 | 0 |
| hotspot | 23,831, 5,958, 5,958, 5,958, 5,958, 5,958, 5,958, 5,957 | 5,957 / 23,831 | 0.722 | 0 |
| tail | 98, 1,465, 1,563, 12,404, 12,502, 12,599, 24,905, 0 | 0 / 24,905 | 1.018 | 1 |

Every result embeds the full `expert_row_counts` array plus min, max,
coefficient of variation, empty-expert count, and number of 128-row-aligned
experts.

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
128-row-aligned groups. In the tested NGC PyTorch build,
`scaled_grouped_mm` uses the MSLK CUTLASS FP4 grouped implementation; this row
is not cuBLASLt. The dense benchmark below uses cuBLASLt through `scaled_mm`.

The first table shows native useful throughput for balanced routing at 8,192
tokens. `M` differs for Llama 4 because its top-k is 1 rather than 8. Percentages
use the 9,000 TFLOP/s dense FP4 specification above.

| model | routed M | FC1 `(N,K)` | FC1 us / TFLOP/s / peak | FC2 `(N,K)` | FC2 us / TFLOP/s / peak |
|---|---:|---:|---:|---:|---:|
| Qwen3-30B | 65,536 | 1,536 × 2,048 | 111.8 / 3,687 / 41.0% | 2,048 × 768 | 84.6 / 2,438 / 27.1% |
| Qwen3-235B | 65,536 | 3,072 × 4,096 | 347.5 / 4,746 / 52.7% | 4,096 × 1,536 | 210.3 / 3,922 / 43.6% |
| Gemma 4 26B | 65,536 | 1,536 × 2,816 | 146.7 / 3,866 / 43.0% | 2,816 × 768 | 113.2 / 2,503 / 27.8% |
| DeepSeek-V3.2 | 65,536 | 4,096 × 7,168 | 736.8 / 5,223 / 58.0% | 7,168 × 2,048 | 451.7 / 4,260 / 47.3% |
| Kimi-K2.7 | 65,536 | 4,096 × 7,168 | 738.0 / 5,214 / 57.9% | 7,168 × 2,048 | 451.8 / 4,259 / 47.3% |
| MiniMax-M2 | 65,536 | 3,072 × 3,072 | 278.6 / 4,440 / 49.3% | 3,072 × 1,536 | 163.1 / 3,792 / 42.1% |
| Llama 4 Scout | 8,192 | 16,384 × 5,120 | 314.8 / 4,367 / 48.5% | 5,120 × 8,192 | 156.9 / 4,379 / 48.7% |

For example, DeepSeek FC1 performs 3,848,290,697,216 logical FLOPs. Dividing by
the 736.784 us median gives 5,223 TFLOP/s, or 58.0% of 9,000 TFLOP/s. Native
IQR was 0.1–1.4% of the median across the 14 balanced rows shown.

The backend comparison below is CUDA-event median latency in microseconds for
the same balanced cases. Bold marks the lowest measured API latency in that
row; it does not imply identical output-allocation contracts.

| model | FC1 native / FlashInfer / PyTorch | FC2 native / FlashInfer / PyTorch |
|---|---:|---:|
| Qwen3-30B | **111.8** / 142.4 / 134.6 | **84.6** / 90.5 / 125.4 |
| Qwen3-235B | **347.5** / 475.4 / 383.4 | **210.3** / 282.8 / 265.5 |
| Gemma 4 26B | **146.7** / 180.7 / 169.2 | **113.2** / 123.3 / 170.5 |
| DeepSeek-V3.2 | **736.8** / 1,086.7 / 788.6 | **451.7** / 617.6 / 496.9 |
| Kimi-K2.7 | **738.0** / 1,117.5 / 788.7 | **451.8** / 619.2 / 499.5 |
| MiniMax-M2 | **278.6** / 368.9 / 312.7 | **163.1** / 215.5 / 206.2 |
| Llama 4 Scout | **314.8** / 437.4 / 357.6 | **156.9** / 205.1 / 179.4 |

The matrix was remeasured on August 13 after the long-hidden scheduler update.
Native led FlashInfer in all eight DeepSeek/Kimi balanced and jagged cases and
PyTorch in all four aligned balanced cases. The balanced advantage was
6.4–9.6% over PyTorch and 26.9–34.0% over FlashInfer. PyTorch remains skipped
for jagged routing because it requires 128-row-aligned groups.

For K=2,048 FC2, a direct scheduler A/B measured static at 394.0 us versus
dynamic at 396.9 us on balanced routing, and 394.0 versus 396.6 us on jagged
routing. The standalone runtime now selects static scheduling for K≤2,048 and
keeps dynamic scheduling for longer reductions. Explicit scheduler selection
remains available to the expert-layer runtime.

All rows shown above used 20 samples after 1,000 ms stabilization, had a valid
native canary, passed the 1.5 wall/GPU gate, and reached sampled reference cosine
of at least 0.99078. The Llama 4 FC1 jagged arm drifted 10.9% and was rejected;
no number from that arm is used above. The published table is balanced routing;
other jagged rows are deterministic stress arms with the counts above.

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
