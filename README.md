# LightMoE

LightMoE is a CuTe DSL kernel library for NVFP4 GEMM and Mixture-of-Experts
workloads on NVIDIA B200 (`sm100`). It provides standalone dense and grouped
GEMM, a reusable-workspace inference layer, and deterministic training kernels
for dispatch, combine, input gradients, weight gradients, and router gradients.

The public API has four entry points:

- `lightmoe.gemm`: standalone dense and grouped GEMM
- `lightmoe.InferenceMoE`: allocation-free inference with reusable workspaces
- `lightmoe.MoEDispatch`: deterministic expert-major token permutation
- `lightmoe.MoEExpertLayer`: single-GPU expert training layer

Measured results and exact timing boundaries are in
[BENCHMARKS.md](BENCHMARKS.md).

## Requirements

- NVIDIA B200 (`sm100`)
- Python 3.12
- CUDA 13
- PyTorch 2.11 or newer
- NVIDIA CUTLASS DSL 4.6 or 4.7

The reference container is `nvcr.io/nvidia/pytorch:26.07-py3`.

```bash
python -m pip install .
python -m pip install '.[test,benchmark]'
```

The first call compiles the selected kernel geometry. Keep construction,
packing, and compilation outside latency measurements.

## Dense GEMM

`DenseGemm` computes `out = A @ B.T` from packed NVFP4 operands. The caller owns
the output, so a warmed call does not allocate.

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

## Grouped GEMM

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

`DenseGemm` and `GroupedGemm` expose the launch runtime directly. `run` and the
call operator execute the same path, without a public dispatcher or output
allocation. Tile choices remain explicit because the fastest geometry depends
on matrix shape and the per-expert assignment distribution.

## MoE inference

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

y = moe(input, top_k_ids, top_k_weights)
```

`top_k_ids` is contiguous CUDA `int32` with distinct expert indices per token;
`top_k_weights` is contiguous CUDA `float32`. The returned workspace is reused
by the next call. Pass `out=` for a caller-owned BF16 output. Frameworks with
expert-major input can call
`run_routed(x, m_indptr, padded_offsets, out=...)`.

Checkpoint activation scales can replace calibration through
`set_activation_scales(input_scale, hidden_scale)`. A checkpoint with bounded
SwiGLU can pass its bound as `activation_clamp=`. Use one plan per concurrent
CUDA stream because workspaces are reused. A warmed plan can be captured in a
CUDA Graph.

Batch size 1 and small-batch decode use decode-specific dispatch and launch
geometry. Larger decode batches and prefill use the persistent path.

## MoE training

The training layer keeps routing outputs in BF16 and provides deterministic
dispatch, combine, router gradients, input gradients, and expert weight
gradients.

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

## Kernel surface

- Dense and grouped NVFP4 × NVFP4 GEMM with BF16 or FP32 output
- Grouped GEMM with dynamic per-expert assignment counts
- Fused SwiGLU, GeGLU, and ReGLU gate/up epilogues
- Static-scale inference with reusable storage and CUDA Graph replay
- Grouped input-gradient and weight-gradient kernels
- Deterministic dispatch, combine, and router gradients
- Imbalanced routing, alignment-stress cases, and zero-assignment experts

Low-level code is grouped under `lightmoe.kernels.dense`, `grouped`,
`quantize`, and `routing`. These modules are available for profiling and kernel
development but are not exported from the package root.

## Limits

- B200 (`sm100`) only
- `K` aligned to 64
- At most 256 local experts
- At most 131,072 token–expert assignments in the complete layer
- The public package contains single-GPU kernels; the benchmark suite contains
  a reference NCCL expert-parallel pipeline
- Precision tables are operator and layer checks, not convergence results

## Development

```bash
ruff check lightmoe benchmarks tests
ruff format --check lightmoe benchmarks tests
python -m pytest -q

python benchmarks/nvfp4_gemm.py --list --suite full
python benchmarks/moe.py --list --suite full
modal run benchmarks/modal_ci.py --grouped api
modal run benchmarks/modal_ci.py::benchmark_inference --preset full
modal run benchmarks/modal_ci.py::benchmark_distributed --preset inference
modal run benchmarks/modal_ci.py::benchmark_distributed --preset training
```

## License

LightMoE is Apache-2.0 licensed. A small set of kernel source files retains
upstream BSD-3-Clause or Apache-2.0 notices; see [NOTICE](NOTICE).
