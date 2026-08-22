# LightMoE

> Fast NVFP4 GEMM and Mixture-of-Experts kernels for NVIDIA Blackwell SM100,
> written in CuTe DSL.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
![GPU](https://img.shields.io/badge/GPU-Blackwell%20SM100-76B900.svg)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://github.com/roycho96/lightmoe/blob/main/LICENSE)

LightMoE provides standalone dense and grouped GEMM, a reusable-workspace MoE
inference layer, and deterministic training kernels for dispatch, combine,
input gradients, weight gradients, and router gradients.

| API | Purpose |
|---|---|
| `lightmoe.gemm` | Standalone dense and grouped NVFP4 GEMM |
| `lightmoe.InferenceMoE` | Prefill and decode with reusable workspaces |
| `lightmoe.MoEDispatch` | Deterministic expert-major token dispatch |
| `lightmoe.MoEExpertLayer` | Expert training layer |

Exact timing boundaries, model shapes, and measured results are documented in
[BENCHMARKS.md](https://github.com/roycho96/lightmoe/blob/main/BENCHMARKS.md).

## 🧩 Compatibility

LightMoE is shape-configurable, not tied to a model allowlist. Standard
SwiGLU-based MoE models can use the same kernels without model-specific code,
subject to the alignment and capacity limits below. Complete inference also
supports bounded SwiGLU, SwiGLU-OAI, GeGLU, ReGLU, and ReLU²; training supports
the same set except ReGLU. Standalone GEMM is independent of the activation.

The following model contracts are included as measured benchmark cases, not as
the complete list of compatible models. LightMoE exposes kernels and routed
expert layers; it does not load or patch model checkpoints.

| Model | Hidden / intermediate | Experts / top-k | Activation | Coverage |
|---|---:|---:|---|---|
| Qwen3.5-35B-A3B | 2048 / 512 | 256 / 8 | SwiGLU | Grouped GEMM and complete MoE layer |
| Qwen3.5-397B-A17B | 4096 / 1024 | 512 / 10 | SwiGLU | Grouped GEMM and complete MoE layer |
| DeepSeek-V4-Flash | 4096 / 2048 | 256 / 6 | Bounded SwiGLU | Grouped GEMM and complete MoE layer |
| GLM-5.2 | 6144 / 2048 | 256 / 8 | SwiGLU | Grouped GEMM and complete MoE layer |
| MiniMax-M3 | 6144 / 3072 | 128 / 4 | SwiGLU-OAI | Grouped GEMM and complete MoE layer |
| Nemotron-3.5-Lightning-30B-A3B | 2688 / 1856 | 128 / 6 | ReLU² | Grouped GEMM and complete MoE layer |
| Kimi-K3 | 3584 / 3072 | 896 / 16 | SiTU-GLU | Grouped latent-expert GEMM |

Kimi-K3's outer latent projections and SiTU-GLU activation are outside the
current complete-layer API.

## 📈 Performance

In same-session B200 measurements, LightMoE wins all 14 reported grouped-GEMM
projection cases, reaching up to **1.49×** the fastest runnable baseline. At
8,192 input tokens, the complete MoE layer is **1.18–1.52×** faster than
FlashInfer across six model contracts, while exact-contract SwiGLU training is
**1.37–1.91×** faster than Transformer Engine. The eight-B200 DeepSeek-V4
full-model swap improves prefill by **1.10–1.13×** and decode by
**1.033–1.037×**.

[![Complete MoE inference latency](https://raw.githubusercontent.com/roycho96/lightmoe/main/assets/moe-inference-latency.svg)](https://github.com/roycho96/lightmoe/blob/main/BENCHMARKS.md#complete-moe-inference)

[![Complete MoE inference speedup](https://raw.githubusercontent.com/roycho96/lightmoe/main/assets/moe-inference-speedup.svg)](https://github.com/roycho96/lightmoe/blob/main/BENCHMARKS.md#complete-moe-inference)

[![Complete MoE training latency](https://raw.githubusercontent.com/roycho96/lightmoe/main/assets/moe-training-latency.svg)](https://github.com/roycho96/lightmoe/blob/main/BENCHMARKS.md#routed-expert-training)

[![Complete MoE training speedup](https://raw.githubusercontent.com/roycho96/lightmoe/main/assets/moe-training-speedup.svg)](https://github.com/roycho96/lightmoe/blob/main/BENCHMARKS.md#routed-expert-training)

## ⚡ Highlights

- NVFP4 × NVFP4 GEMM with BF16 or FP32 output
- Dynamic per-expert row counts, imbalanced routing, and zero-assignment experts
- Compile-time SwiGLU-OAI, bounded SwiGLU, ReLU², GeGLU, and ReGLU epilogues
- Decode-specific and persistent prefill launch paths
- Reusable inference storage with CUDA Graph support
- Deterministic dispatch, combine, input-gradient, weight-gradient, and
  router-gradient kernels
- Direct kernel runtime API without a dispatcher or `torch.library` launch hop

## 🚀 Installation

| Requirement | Version |
|---|---|
| GPU | NVIDIA Blackwell SM100; validated on B200 |
| Python | 3.12 |
| CUDA | 13 |
| PyTorch | 2.11 or newer |
| NVIDIA CUTLASS DSL | 4.6 or 4.7 |

The reference environment is `nvcr.io/nvidia/pytorch:26.07-py3`.

```bash
python -m pip install lightmoe
```

The first call compiles the selected kernel geometry. Keep construction,
packing, and compilation outside latency measurements.

## 🧮 Standalone GEMM

### Dense GEMM

`DenseGemm` computes `out = A @ B.T` from packed NVFP4 operands. The caller
owns the output, so a warmed call does not allocate.

```python
import torch

from lightmoe.gemm import DenseGemm, quantize

# a: [M, K] BF16, b: [N, K] BF16
qa, sfa, scale_a = quantize(a)
qb, sfb, scale_b = quantize(b)

out = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
gemm = DenseGemm(n=N, k=K, tile_m=256, tile_n=256)
gemm.run(qa, qb, out, sfa, sfb, scale_a * scale_b)
```

`quantize` returns packed E2M1 values, blocked E4M3 scale factors, and a
per-tensor dequantization scale. Packing is a separate operation.

### Grouped GEMM

Grouped inputs use contiguous expert-major rows. `m_indptr` is an `int32` CUDA
tensor of length `num_experts + 1`; zero-assignment experts are valid.

```python
import torch

from lightmoe.gemm import GroupedGemm, quantize_grouped

# a: [sum(M_e), K], b: [num_experts, N, K]
qa, qb, sfa, sfb, alpha = quantize_grouped(a, b, m_indptr)
out = torch.empty(a.shape[0], N, dtype=torch.bfloat16, device=a.device)

gemm = GroupedGemm(
    experts=num_experts,
    n=N,
    k=K,
    tile_m=256,
    tile_n=256,
)
gemm.run(qa, qb, out, m_indptr, sfa, sfb, alpha)
```

`run` and the call operator execute the same launch path. Tile choices remain
explicit because the fastest geometry depends on the matrix shape and expert
assignment distribution.

## 🔀 MoE inference

`InferenceMoE` owns packed weights and fixed workspaces. One call includes
dispatch, BF16 input quantization, gate/up projection, gated activation, NVFP4
requantization, down projection, routing-weight application, and deterministic
combine.

```python
from lightmoe import InferenceMoE

moe = InferenceMoE(
    hidden_size=2048,
    intermediate_size=768,
    num_experts=16,
    top_k=8,
    max_tokens=8192,
)
moe.load_weights(gate_weight, up_weight, down_weight)
moe.calibrate(calibration_input, top_k_ids, top_k_weights)
moe.warmup(calibration_input, top_k_ids, top_k_weights)

y = moe(x, top_k_ids, top_k_weights)
```

`top_k_ids` must be contiguous CUDA `int32` with distinct expert indices per
token. `top_k_weights` must be contiguous CUDA `float32`. Pass `out=` for a
caller-owned BF16 output. Frameworks with expert-major input can call
`run_routed(x, m_indptr, padded_offsets, out=...)`.

Checkpoint scales can replace calibration through
`set_activation_scales(input_scale, hidden_scale)`. Use one plan per concurrent
CUDA stream because its workspaces are reused.

Set `activation="swiglu_oai"` for MiniMax-M3, `activation="relu2"` for
Nemotron 3.5, or `activation="swiglu", activation_clamp=10` for DeepSeek-V4.
ReLU² experts load two weights with `load_weights(up_weight, down_weight)`;
gated experts retain the three-weight form shown above.

## 🧠 MoE training

The training layer keeps routing outputs in BF16 and supports deterministic
expert and router gradients.

```python
from lightmoe import MoEDispatch, MoEExpertLayer

dispatch = MoEDispatch(num_tokens=8192, num_experts=128, top_k=8)
experts = MoEExpertLayer(
    hidden_size=2048,
    intermediate_size=768,
    num_experts=128,
    top_k=8,
).cuda()
experts.refresh_weights()

gather_indices, m_indptr, routing_weights, inverse_slots = dispatch(
    top_k_ids,
    top_k_weights,
)
experts.calibrate(
    x,
    gather_indices,
    m_indptr,
    routing_weights,
    off_pad=dispatch.off_pad,
)
routing_weights = dispatch.differentiable_probs(top_k_weights)
y = experts(
    x,
    gather_indices,
    m_indptr,
    routing_weights,
    inverse_slots,
    off_pad=dispatch.off_pad,
)
```

Call `experts.refresh_weights()` after each optimizer step. Expert-parallel
communication and routing policy remain the host framework's responsibility.
The inference activation values above select the same compile-time forward and
backward specializations in `MoEExpertLayer`.

## 🗂️ Package layout

```text
lightmoe/
├── gemm.py          # Standalone GEMM API
├── inference.py     # Reusable-workspace inference layer
├── training.py      # Routed training layer
└── kernels/
    ├── dense/       # Dense GEMM
    ├── grouped/     # Grouped GEMM and gradients
    ├── quantize/    # NVFP4 packing and scaling
    └── routing/     # Dispatch and combine
```

Low-level modules are available for profiling and kernel development but are
not exported from the package root.

## 📐 Limits

- NVIDIA Blackwell SM100; validated on B200
- `K` aligned to 64
- At most 256 local experts
- At most 131,072 token–expert assignments in the complete layer
- Precision tables are operator and layer checks, not convergence results

## 📊 Benchmarks

From a source checkout:

```bash
python -m benchmarks.nvfp4_gemm --list --suite full
python -m benchmarks.moe --list --suite full
```

Comparison backends are used only when their packages are already installed.
See [BENCHMARKS.md](https://github.com/roycho96/lightmoe/blob/main/BENCHMARKS.md)
for measurement rules and results.

## 🧪 Development

```bash
git clone https://github.com/roycho96/lightmoe.git
cd lightmoe
python -m pip install -e '.[test]'
ruff check lightmoe benchmarks tests
ruff format --check lightmoe benchmarks tests
python -m pytest -q
```

## 📄 License

LightMoE is Apache-2.0 licensed. A small set of kernel source files retains
upstream BSD-3-Clause or Apache-2.0 notices; see
[NOTICE](https://github.com/roycho96/lightmoe/blob/main/NOTICE).
