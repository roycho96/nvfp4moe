# B200 benchmarks

Measured on August 11, 2026, on one NVIDIA B200 (`sm100`) with NGC PyTorch
26.07, CUDA 13, CUTLASS DSL 4.6, Transformer Engine 2.17, and DeepGEMM at
`559d79f`. JIT compilation, weight packing, calibration, and input creation
complete before timing.

There are two public benchmark entry points:

1. `benchmarks/nvfp4_gemm.py` for standalone grouped GEMM
2. `benchmarks/nvfp4_moe.py` for expert-core and complete MoE layer timing

Both emit JSON for `--list` and JSONL while running. A missing backend is
reported as `skipped`; the scripts never record another implementation under
its name.

## ⚙️ Workload

The tables below use 8,192 tokens and deterministic jagged routing. Top-k 8
models therefore execute 65,536 routed rows. Each case uses the practical
quick-suite expert-parallel shard shown here.

| model | D | I | global E | top-k | EP | local E | activation |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3-30B-A3B | 2,048 | 768 | 128 | 8 | 16 | 8 | SwiGLU |
| Qwen3-235B-A22B | 4,096 | 1,536 | 128 | 8 | 16 | 8 | SwiGLU |
| Gemma 4 26B A4B | 2,816 | 704→768 | 128 | 8 | 16 | 8 | GeGLU |
| DeepSeek-V3.2 | 7,168 | 2,048 | 256 | 8 | 32 | 8 | SwiGLU |
| Kimi-K2.7 | 7,168 | 2,048 | 384 | 8 | 48 | 8 | SwiGLU |
| MiniMax-M2 | 3,072 | 1,536 | 256 | 8 | 32 | 8 | SwiGLU |
| Llama 4 Scout | 5,120 | 8,192 | 16 | 1 | 2 | 8 | SwiGLU |

Forward tables are CUDA-event medians of 10 samples after three warmups.
Training tables are medians of five forward-plus-backward samples after two
warmups. All comparisons are same-session. TE pads each local expert to 64
rows; both logical and executed throughput are emitted in the raw result.

## 🧱 Standalone NVFP4 grouped GEMM

`dynamic` includes BF16 activation quantization and grouped GEMM. Expert
weights remain resident and prepacked for both implementations.

| model | native FC1 | TE FC1 | native FC2 | TE FC2 | native speedup range |
|---|---:|---:|---:|---:|---:|
| Qwen3-30B | 0.188 ms | 0.527 ms | 0.114 ms | 0.445 ms | 2.80×–3.89× |
| Qwen3-235B | 0.490 ms | 0.949 ms | 0.284 ms | 0.687 ms | 1.93×–2.42× |
| Gemma 4 26B | 0.238 ms | 0.615 ms | 0.143 ms | 0.497 ms | 2.58×–3.47× |
| DeepSeek-V3.2 | 0.989 ms | 1.632 ms | 0.544 ms | 1.028 ms | 1.65×–1.89× |
| Kimi-K2.7 | 0.990 ms | 1.702 ms | 0.572 ms | 1.063 ms | 1.72×–1.86× |
| MiniMax-M2 | 0.383 ms | 0.810 ms | 0.231 ms | 0.597 ms | 2.12×–2.58× |
| Llama 4 Scout | 0.361 ms | 0.637 ms | 0.206 ms | 0.458 ms | 1.76×–2.23× |

Native won all 14 dynamic cases. The CLI also measures prepacked GEMM-only,
dgrad, and the K-grouped wgrad contract. Qwen FC2 smoke tests exercised all
six native mode/direction combinations successfully.

## ⚡ Full MoE forward

`full-layer` includes dispatch, FC1, activation, FC2, probability weighting,
and combine. Router logits and top-k selection are excluded. These rows use
synthetic tensors with the exact model geometry and deterministic routing;
they are performance cases, not checkpoint-quality comparisons.

| model | native NVFP4 | TE NVFP4 | DeepGEMM BF16 | DeepGEMM FP8×FP4 | Torch BF16 |
|---|---:|---:|---:|---:|---:|
| Qwen3-30B | **0.527 ms** | 2.532 ms | 1.252 ms | 2.417 ms | 1.899 ms |
| Qwen3-235B | **0.963 ms** | 2.749 ms | 2.476 ms | 4.495 ms | 4.415 ms |
| Gemma 4 26B | **0.532 ms** | 1.708 ms | 1.270 ms | 2.698 ms | 2.391 ms |
| DeepSeek-V3.2 | **1.713 ms** | 3.695 ms | 4.381 ms | 7.062 ms | 8.809 ms |
| Kimi-K2.7 | **1.714 ms** | 3.714 ms | 4.358 ms | 7.091 ms | 8.958 ms |
| MiniMax-M2 | **0.782 ms** | 2.475 ms | 2.001 ms | 3.704 ms | 3.480 ms |
| Llama 4 Scout | **0.763 ms** | 2.311 ms | 2.719 ms | 3.504 ms | 2.429 ms |

## 🔁 Full MoE training

The timed region contains forward, output-gradient backward, input gradient,
router-weight gradient, and both expert weight gradients.

| model | native NVFP4 | TE NVFP4 | DeepGEMM BF16 |
|---|---:|---:|---:|
| Qwen3-30B | **2.071 ms** | 3.843 ms | 3.573 ms |
| DeepSeek-V3.2 | **8.897 ms** | 9.939 ms | 16.135 ms |
| Kimi-K2.7 | **8.830 ms** | 10.063 ms | 16.244 ms |
| MiniMax-M2 | **3.790 ms** | 5.366 ms | 6.831 ms |

All input and router gradients were finite. The native path led both
comparators in every measured training case, including the long-K DeepSeek and
Kimi geometries.

## 🧪 Precision

A separate 8,192-token FineWeb-Edu trace uses Qwen3-30B-A3B layer-0 checkpoint
weights, activations, and routing. Against a BF16 PyTorch layer reference:

| tensor | cosine similarity | relative L2 |
|---|---:|---:|
| output | 0.996526 | 0.08444 |
| input gradient | 0.975384 | 0.22193 |
| router gradient | 0.992358 | 0.12360 |
| gate/up weight gradient | 0.968857 | 0.25141 |
| down weight gradient | 0.986734 | 0.16351 |

The routed-core B200 test measured 0.97357 output cosine to the FP32 master
reference and verified finite input, gate, up, and down gradients. Packed E2M1
data and active E4M3 scale factors are checked against the independent PyTorch
format reference; TE row/column quantization is also checked bitwise.

These are layer-level checks, not a convergence result.

## 🧭 Matrix controls

The quick suite uses token counts 128 and 8,192, one practical EP shard, and
balanced/jagged routing. The full suite covers:

- tokens: 1, 128, 512, 2,048, 8,192, 16,384
- routing: balanced, jagged, hotspot, and boundary-tail distributions
- every registered EP size
- FC1 and FC2
- standalone fwd, dgrad, and wgrad
- prepacked and dynamic quantization scopes
- expert-core and full-layer
- forward and forward-plus-backward

`--source trace` replays a local EP shard. The `.pt` file must contain
`expert_input`, `topk_index`, `topk_weight`, `gate_up_weight`, and
`down_weight`. The same timing rules apply to synthetic and captured inputs.

## 🔁 Reproduce

Inspect matrices without a GPU:

```bash
python benchmarks/nvfp4_gemm.py --list --suite full
python benchmarks/nvfp4_moe.py --list --suite full
```

Run the 8K standalone comparison:

```bash
python benchmarks/nvfp4_gemm.py \
  --models all --tokens 8192 --routing jagged \
  --backends native,te_nvfp4 --mode dynamic
```

Run the full expert comparison and training pass:

```bash
python benchmarks/nvfp4_moe.py \
  --models all --tokens 8192 --routing jagged \
  --scope both --pass fwd

python benchmarks/nvfp4_moe.py \
  --models qwen3_30b_a3b,deepseek_v3_2,kimi_k2_7,minimax_m2 \
  --tokens 8192 --routing jagged --scope full-layer --pass fwd_bwd
```

Run release checks on a Modal B200:

```bash
modal run benchmarks/modal_ci.py
modal run benchmarks/modal_ci.py --benchmark-smoke
modal run benchmarks/modal_ci.py --matrix gemm
modal run benchmarks/modal_ci.py --matrix moe-forward
modal run benchmarks/modal_ci.py --matrix moe-training
```

Keep raw JSONL, NCU, and NSYS artifacts with the environment and git revision
when publishing new results.
