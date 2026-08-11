# nvfp4moe

Training-capable NVFP4 Mixture-of-Experts kernels for NVIDIA Blackwell B200
(`sm100`). The package covers grouped GEMM, fused gated epilogues, dispatch,
combine, input gradients, weight gradients, and router gradients.

This is an alpha release for single-GPU kernel work and expert-parallel
integration. Communication and routing policy stay with the host framework.

## ⚡ What is included

- Native CuTe DSL grouped NVFP4 GEMM for forward and input gradients
- Fused SwiGLU, GeGLU, and quantized FC1 epilogues
- Grouped NVFP4 weight-gradient kernel
- Full training expert layer with deterministic dispatch/combine
- Parameter-free expert core for framework-owned BF16 master weights
- `torch.compile`-visible custom op for prepacked grouped GEMM
- Small adapters for Transformer Engine and TorchTitan expert boundaries

Current B200 results, model shapes, precision checks, and exact commands are in
[BENCHMARKS.md](BENCHMARKS.md).

## 🚀 Install

Tested with NVIDIA B200, Python 3.12, CUDA 13, PyTorch 2.11 or newer, and
CUTLASS DSL 4.6.x. The reference image is
`nvcr.io/nvidia/pytorch:26.07-py3`.

```bash
python -m pip install .

# Development and benchmark tools
python -m pip install '.[test,benchmark]'
```

The first call compiles kernels for the selected geometry. Keep the JIT cache
between runs.

## 🔧 Framework-owned experts

`NVFP4ExpertCore` starts after token permutation. Inputs are contiguous
expert-major rows; `w1` and `w3` are `[E, I, D]`, and `w2` is `[E, D, I]`.
The caller retains ownership of parameters, checkpoints, optimizer state, and
expert-parallel communication.

```python
import torch
from nvfp4moe import NVFP4ExpertCore

core = NVFP4ExpertCore(D, I, E, activation="swiglu").cuda()

counts = num_tokens_per_local_expert.to(torch.int32)
cu = torch.cat((counts.new_zeros(1), counts.cumsum(0, dtype=torch.int32)))
off_pad = (((counts + 127) // 128) * 128).cumsum(0, dtype=torch.int32)

core.refresh_weights(w1, w3, w2)
core.calibrate(expert_input, cu, off_pad)
output = core(expert_input, w1, w3, w2, cu, off_pad)
```

The core notices in-place optimizer updates and refreshes its resident NVFP4
weights on the next call. Calling `refresh_weights()` immediately after
`optimizer.step()` keeps that work outside the following forward measurement.
`off_pad` is required; there is no backward-time Torch fallback.

## 🧱 Complete routed layer

Use `MoEDispatch` when the package also owns permutation and combine.

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

Call `experts.refresh_weights()` after each optimizer step. The router remains
BF16 and receives gradients through `differentiable_probs()`.

## 🧩 Standalone GEMM and `torch.compile`

The functional op accepts packed E2M1 operands and E4M3 scale factors. It
allocates the output, keeps the kernel opaque to Dynamo, and supports a dynamic
total routed-row count.

```python
import torch
from nvfp4moe import grouped_nvfp4_gemm

compiled_gemm = torch.compile(grouped_nvfp4_gemm)
y = compiled_gemm(qa, qb, sfa, sfb, cu, alpha)
```

Packed operands are deliberately non-differentiable. Use `NVFP4ExpertCore` or
`MoEExpertLayer` for training.

For launch control or custom epilogues, use the lower-level classes directly:

```python
from nvfp4moe import GatedEpilogue, GroupedNvfp4Gemm

gemm = GroupedNvfp4Gemm(
    experts=8,
    n=1536,
    k=2048,
    tile_m=256,
    tile_n=128,
    output_dtype=torch.float4_e2m1fn_x2,
    epilogue=GatedEpilogue("swiglu"),
)
```

`nvfp4moe.kernels` also exports the quantizers, grouped wgrad, gated fragment
helpers, scheduler kernel, and finalize kernels independently.

## 🔌 Transformer Engine

`TEExpertAdapter` follows the common `forward(inp, m_splits,
is_first_microbatch=None)` convention without importing or patching
Transformer Engine. TE can continue to own attention, normalization, dense
layers, and token transport.

```python
from nvfp4moe import NVFP4ExpertCore, TEExpertAdapter

core = NVFP4ExpertCore(D, I, E).cuda()
experts = TEExpertAdapter(core, w1, w3, w2)
y = experts(expert_input, num_tokens_per_local_expert, is_first_microbatch=True)
```

Pass `is_first_microbatch=True` after an optimizer update to refresh the
resident weights before the first microbatch.

## 🔌 TorchTitan

The converter keeps `GroupedExperts` parameters and DTensor local shards in
place and changes only expert computation.

```python
from nvfp4moe import TorchTitanExpertsConfig

config = TorchTitanExpertsConfig(fqns=["layers.*.experts"])
converter = config.build(parallel_dims=parallel_dims)
converter.convert(model)

# after optimizer.step()
converter.post_optimizer_hook(model)
```

Conversion is lazy, so meta-device model construction and sharding can finish
before the native core is created.

## 📁 Layout

```text
nvfp4moe/
├── layer.py          full expert layer and framework-owned core
├── ops.py            PyTorch custom operators
├── te.py             Transformer Engine calling adapter
├── torchtitan.py     TorchTitan GroupedExperts converter
├── recipe.py         tensor-scale state
├── reference.py      readable format reference
└── kernels/
    ├── gemm.py       validation, JIT cache, and launch API
    ├── gemm_kernel.py
    ├── epilogue.py
    ├── scheduler.py
    ├── quantize.py
    ├── dispatch.py
    ├── finalize.py
    └── wgrad/
```

## ⚠️ Limits

- B200 `sm100` only for the native kernels
- `D` divisible by 256 and `I` divisible by 128
- At most 256 local experts and 131,072 routed rows
- Global expert counts above 256 require external expert parallelism
- Single-GPU kernels; no all-to-all or distributed router is included
- Precision tables are layer checks, not a convergence claim

## 📄 License

Apache-2.0. NVIDIA-derived kernel files retain their upstream copyright,
license, and modification notices. See `LICENSE` and `NOTICE`.
