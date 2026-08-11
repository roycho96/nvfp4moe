# nvfp4moe

Training-capable NVFP4 Mixture-of-Experts kernels for NVIDIA Blackwell B200
(`sm100`). The package implements dispatch, fused gated FC1, FC2,
deterministic combine, input gradients, weight gradients, and router gradients.

This is an alpha release for single-GPU kernel research and model integration.

## ⚡ Highlights

- Native CuTe DSL kernels for routed NVFP4 expert computation
- Fused SwiGLU and GeGLU FC1 with NVFP4 output quantization
- Fused FC2 input gradient and gated-activation backward
- Grouped NVFP4 weight gradients with optional accumulation
- Direct gather-and-quantize input path; no BF16 expert buffer is required
- Hugging Face adapter and real-model training example for Qwen3 MoE

Fresh B200 measurements, precision checks, model shapes, and reproduction
commands are in [BENCHMARKS.md](BENCHMARKS.md).

The release benchmark uses one 8,192-token FineWeb-Edu sequence with real
Qwen3-30B-A3B layer-0 weights, activations, and routing. Times are same-session
CUDA-event medians on one B200; JIT compilation is excluded.

| expert backend | precision | forward | forward + backward |
|---|---:|---:|---:|
| Transformer Engine | BF16 | 5.256 ms | 16.488 ms |
| Transformer Engine 2 × 64 | NVFP4 | 12.900 ms | 33.306 ms |
| TE dispatch + DeepGEMM | BF16 | 2.244 ms | 11.536 ms |
| TE dispatch + DeepGEMM | FP8 × FP4 | 4.549 ms | — |
| `nvfp4moe` | NVFP4 × NVFP4 | **0.735 ms** | **3.328 ms** |

Against BF16 PyTorch, the Qwen layer output measured 0.996526 cosine
similarity and 0.08444 relative L2 error. The input, router, gate/up-weight,
and down-weight gradients measured cosine similarities of 0.975384, 0.992358,
0.968857, and 0.986734. These are one-layer checks, not a convergence claim.

## 🚀 Install

Tested configuration:

- NVIDIA B200 (`sm100`)
- Linux and Python 3.12
- CUDA 13
- PyTorch 2.11 or newer
- NVIDIA CUTLASS DSL 4.6.x

The reference container is `nvcr.io/nvidia/pytorch:26.07-py3`.

```bash
python -m pip install .

# Tests and benchmark dependencies
python -m pip install '.[test,benchmark]'
```

## 🔧 Use

`MoEExpertLayer` accepts expert-major routing metadata from `MoEDispatch`.
Weights are kept in BF16 for optimization and refreshed into resident NVFP4
buffers after an optimizer step.

```python
from nvfp4moe import MoEDispatch, MoEExpertLayer

dispatch = MoEDispatch(T=8192, E=128, k=8)
experts = MoEExpertLayer(
    d=2048,
    I=768,
    E=128,
    topk=8,
    activation="swiglu",
).cuda()

experts.refresh_weights()
```

The short training example loads Qwen3-30B-A3B, replaces one real MoE expert
group, streams a FineWeb-Edu batch, and runs optimizer steps on the model's
causal-language-modeling loss. The rest of the model is frozen.

```bash
python examples/train_qwen.py --tokens 256 --steps 2
python examples/train_qwen.py --tokens 8192 --steps 1
```

The adapter preserves the model's BF16 router and differentiable top-k routing
weights. Call `refresh_weights()` after every optimizer step.

### Kernel API

The grouped GEMM can be used without the expert-layer wrapper. Epilogues are
compile-time policies, so selecting one does not add a separate launch.

```python
import torch
from nvfp4moe import GatedEpilogue, GroupedNvfp4Gemm

gemm = GroupedNvfp4Gemm(
    experts=128,
    n=1536,
    k=2048,
    tile_m=256,
    tile_n=128,
    output_dtype=torch.float4_e2m1fn_x2,
    epilogue=GatedEpilogue("swiglu"),
)
```

`GroupedNvfp4Gemm` without an epilogue is the plain grouped GEMM. CuTe DSL
fragment helpers such as `gated_postact_fragment`, `gated_backward_values`,
and `quantize_postact_fragment` are exported from `nvfp4moe.kernels` for use in
custom kernels.

### Transformer Engine integration

Use `nvfp4moe` at the complete expert boundary. Keep Transformer Engine for
attention, normalization, dense layers, and FP8 state, while replacing expert
permutation, FC1/activation, FC2, and unpermutation together. Replacing only a
single GEMM leaves intermediate buffers and launches in the critical path.

## 🧩 Layout

```text
nvfp4moe/
├── hf.py                         Hugging Face adapter
├── layer.py                      training expert layer
├── reference.py                  readable PyTorch format reference
└── kernels/
    ├── dispatch.py               expert-major routing
    ├── epilogue.py               reusable gated and quantized epilogues
    ├── finalize.py               deterministic combine and gather
    ├── gemm.py                   public validation, JIT, and launch API
    ├── gemm_kernel.py            persistent SM100 grouped GEMM
    ├── quantize.py               NVFP4 quantization and RHT
    ├── scheduler.py              shared persistent tile scheduler
    └── wgrad/                    weight-gradient kernel implementation
```

## ⚠️ Scope

- Single GPU; expert/data/tensor parallel communication is not included.
- `d` must be divisible by 256 and `I` by 128.
- At most 256 local experts and 131,072 routed rows are supported.
- Larger global expert counts require an external expert-parallel shard.
- Router computation remains BF16.
- The first call compiles shape-specialized kernels.
- Precision results are one-layer checks, not a convergence claim.

## 📄 License

Apache-2.0. NVIDIA-derived grouped GEMM files retain their upstream copyright,
license, and modification notices. See `LICENSE` and `NOTICE`.
