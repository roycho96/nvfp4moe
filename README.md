# nvfp4moe

CuTe DSL NVFP4 GEMM and Mixture-of-Experts kernels for NVIDIA B200 (`sm100`).
The package includes dense and grouped GEMM, an inference-only MoE plan, and a
training layer with input, weight, and router gradients.

The public surface is deliberately small:

- `nvfp4moe.gemm`: standalone dense and grouped GEMM
- `nvfp4moe.InferenceMoE`: allocation-free inference plan with static scales
- `nvfp4moe.MoEDispatch`: deterministic token permutation and combine metadata
- `nvfp4moe.MoEExpertLayer`: complete single-GPU expert training layer

Current results and measurement rules are in [BENCHMARKS.md](BENCHMARKS.md).

## Requirements

- NVIDIA B200 (`sm100`)
- Python 3.12
- CUDA 13
- PyTorch 2.11 or newer
- NVIDIA CUTLASS DSL 4.6 or 4.7

The reference container is `nvcr.io/nvidia/pytorch:26.07-py3`.

```bash
python -m pip install .

# Tests and benchmark dependencies
python -m pip install '.[test,benchmark]'
```

The first call compiles the selected kernel geometry. Keep construction,
packing, and the first call outside latency measurements.

## Dense GEMM

`DenseGemm` computes `out = A @ B.T`. Inputs are packed NVFP4 and the output is
caller-owned, so steady-state calls do not allocate.

```python
import torch
from nvfp4moe.gemm import DenseGemm, quantize

# a: [M, K] BF16, b: [N, K] BF16
qa, sfa, scale_a = quantize(a)
qb, sfb, scale_b = quantize(b)

out = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
gemm = DenseGemm(n=N, k=K, tile_m=256, tile_n=256)

# Warm up once, then reuse the same plan and output.
gemm.run(qa, qb, out, sfa, sfb, scale_a * scale_b)
```

`quantize` returns packed E2M1 values, blocked E4M3 scale factors, and the
per-tensor dequantization scale. Packing is separate from GEMM timing.

## Grouped GEMM

Grouped inputs use contiguous expert-major rows. `m_indptr` is an `int32` CUDA
tensor of length `E + 1`; empty experts are valid.

```python
import torch
from nvfp4moe.gemm import GroupedGemm, quantize_grouped

# a: [sum(M_e), K], b: [E, N, K]
qa, qb, sfa, sfb, alpha = quantize_grouped(a, b, m_indptr)
out = torch.empty(a.shape[0], N, dtype=torch.bfloat16, device=a.device)

gemm = GroupedGemm(
    experts=E,
    n=N,
    k=K,
    tile_m=256,
    tile_n=256,
)
gemm.run(qa, qb, out, m_indptr, sfa, sfb, alpha)
```

`DenseGemm` and `GroupedGemm` are direct aliases of the native runtime classes;
`run` and the runtime call operator are the same function. There is no public
dispatcher, wrapper allocation, or `torch.library` hop in the launch path.
Same-session B200 measurements found at most +0.078% CUDA-event difference
between the two call spellings across the release gate.

Tile choices are explicit because the best geometry depends on shape and expert
row distribution. The benchmark runner tests the supported candidates before
recording a result.

## MoE layer

`InferenceMoE` owns fixed workspaces and packed weights. Its timed path includes
dispatch, BF16 activation quantization, fused FC1 + gated activation + NVFP4
requantization, FC2, probability weighting, and deterministic combine.

```python
from nvfp4moe import InferenceMoE

moe = InferenceMoE(
    hidden_size=2048,
    intermediate_size=768,
    experts=16,
    topk=8,
    max_tokens=8192,
)
moe.load_weights(gate_weight, up_weight, down_weight)
moe.calibrate(calibration_input, topk_index, topk_weight)
moe.warmup(calibration_input, topk_index, topk_weight)

y = moe(input, topk_index, topk_weight)
```

`topk_index` is contiguous CUDA `int32` with distinct expert indices in each
token row; `topk_weight` is contiguous CUDA `float32`. The returned workspace
is overwritten by the next call. Pass
`out=` to use a caller-owned BF16 output. Frameworks that already have
expert-major rows can call `run_routed(x, m_indptr, padded_offsets, out=...)`.
Checkpoint activation scales can replace calibration through
`set_activation_scales(input_scale, hidden_scale)`.
Use one plan per concurrent CUDA stream because its workspaces are reused.
Batch-one decode fuses dispatch and input quantization when top-k is at most
32. Long-hidden decode uses a launch grid selected for batch-one and small
batches; larger decode batches and prefill use the persistent path.

Construction, weight packing, calibration, and `warmup()` stay outside latency
measurements. A warmed plan does not allocate CUDA memory and can be captured
in a CUDA Graph.

### Training

The routed layer keeps the router in BF16 and provides deterministic dispatch,
combine, router gradients, input gradients, and expert weight gradients.

```python
from nvfp4moe import MoEDispatch, MoEExpertLayer

dispatch = MoEDispatch(T=8192, E=128, k=8)
experts = MoEExpertLayer(d=2048, I=768, E=128, topk=8).cuda()
experts.refresh_weights()

gather, cu, probs, slots = dispatch(topk_index, topk_weight)
experts.calibrate(x, gather, cu, probs, off_pad=dispatch.off_pad)
probs = dispatch.differentiable_probs(topk_weight)
y = experts(x, gather, cu, probs, slots, off_pad=dispatch.off_pad)
```

Call `experts.refresh_weights()` after each optimizer step. Expert-parallel
communication and routing policy remain the host framework's responsibility.
Frameworks can call the standalone plans with ordinary CUDA tensors; no
framework-specific adapter is required.

## Supported kernel surface

- Dense NVFP4 × NVFP4 GEMM with BF16 or FP32 output
- Grouped NVFP4 × NVFP4 GEMM with dynamic expert row counts
- Fused SwiGLU, GeGLU, and ReGLU FC1 epilogues
- Static-scale, allocation-free inference with CUDA Graph replay
- Grouped input-gradient and weight-gradient kernels
- Deterministic dispatch, combine, and router gradients
- Dynamic routed-row shapes, skewed routing, and empty experts

Low-level quantizers, epilogues, schedulers, and launch runtimes live under
`nvfp4moe.kernels` for profiling and kernel development. They are not exported
from the package root.

## Limits

- B200 (`sm100`) only
- `K` aligned to 64
- At most 256 local experts
- At most 131,072 routed rows in the complete layer
- The public package exposes single-GPU kernels; the benchmark includes a
  reference NCCL expert-parallel pipeline
- Precision tables are operator and layer checks, not convergence results

## Development

```bash
ruff check nvfp4moe benchmarks tests
ruff format --check nvfp4moe benchmarks tests
pytest -q

python benchmarks/nvfp4_gemm.py --list --suite full
python benchmarks/nvfp4_moe.py --list --suite full
modal run benchmarks/modal_ci.py --grouped api
modal run benchmarks/modal_ci.py::benchmark_inference --preset decode
modal run benchmarks/modal_ci.py::benchmark_inference --preset full
modal run benchmarks/modal_ci.py::benchmark_distributed --preset inference
modal run benchmarks/modal_ci.py::benchmark_distributed --preset training
```

## License

Apache-2.0. NVIDIA-derived files retain their upstream copyright, license, and
modification notices. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
