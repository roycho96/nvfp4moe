# LightMoE

> Fast NVFP4 GEMM and Mixture-of-Experts kernels for NVIDIA B200, written in
> CuTe DSL.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
![GPU](https://img.shields.io/badge/GPU-B200%20(sm100)-76B900.svg)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

LightMoE provides standalone dense and grouped GEMM, a reusable-workspace MoE
inference layer, and deterministic training kernels for dispatch, combine,
input gradients, weight gradients, and router gradients.

| API | Purpose |
|---|---|
| `lightmoe.gemm` | Standalone dense and grouped NVFP4 GEMM |
| `lightmoe.InferenceMoE` | Prefill and decode with reusable workspaces |
| `lightmoe.MoEDispatch` | Deterministic expert-major token dispatch |
| `lightmoe.MoEExpertLayer` | Single-GPU expert training layer |

Exact timing boundaries, model shapes, and measured results are documented in
[BENCHMARKS.md](BENCHMARKS.md).

## ⚡ Highlights

- NVFP4 × NVFP4 GEMM with BF16 or FP32 output
- Dynamic per-expert row counts, imbalanced routing, and zero-assignment experts
- Fused SwiGLU, GeGLU, and ReGLU gate/up epilogues
- Decode-specific and persistent prefill launch paths
- Reusable inference storage with CUDA Graph support
- Deterministic dispatch, combine, input-gradient, weight-gradient, and
  router-gradient kernels
- Direct kernel runtime API without a dispatcher or `torch.library` launch hop

## 🚀 Installation

| Requirement | Version |
|---|---|
| GPU | NVIDIA B200 (`sm100`) |
| Python | 3.12 |
| CUDA | 13 |
| PyTorch | 2.11 or newer |
| NVIDIA CUTLASS DSL | 4.6 or 4.7 |

The reference environment is `nvcr.io/nvidia/pytorch:26.07-py3`.

```bash
python -m pip install .
python -m pip install '.[test]'
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

- B200 (`sm100`) only
- `K` aligned to 64
- At most 256 local experts
- At most 131,072 token–expert assignments in the complete layer
- Public kernels and benchmark runners are single-GPU
- Precision tables are operator and layer checks, not convergence results

## 📊 Benchmarks

```bash
python -m benchmarks.nvfp4_gemm --list --suite full
python -m benchmarks.moe --list --suite full
```

Comparison backends are used only when their packages are already installed.
See [BENCHMARKS.md](BENCHMARKS.md) for measurement rules and results.

## 🧪 Development

```bash
ruff check lightmoe benchmarks tests
ruff format --check lightmoe benchmarks tests
python -m pytest -q
```

## 📄 License

LightMoE is Apache-2.0 licensed. A small set of kernel source files retains
upstream BSD-3-Clause or Apache-2.0 notices; see [NOTICE](NOTICE).
