# nvfp4moe

Training-capable NVFP4 Mixture-of-Experts kernels for NVIDIA Blackwell B200
(`sm100`). `nvfp4moe` covers the full expert path: dispatch, FC1 with
SwiGLU/GeGLU, FC2, combine, input gradients, weight gradients, and routing
probability gradients.

This is an alpha release with a deliberately narrow hardware and shape target.
The current implementation is single-GPU and intended for kernel research and
early model integration.

## ⚡ Results

The main benchmark uses real 8,192-token FineWeb-Edu sequences and layer-0
activations, routing, and expert weights from Qwen3-30B-A3B and Gemma 4
26B-A4B. Times are same-session CUDA-event p50 on one B200, after warmup and
outside JIT compilation. Training is forward plus the complete backward pass.

| model / expert backend | precision | forward | training |
|---|---:|---:|---:|
| Qwen3-30B-A3B / **nvfp4moe** | NVFP4 x NVFP4 | **0.558 ms** | **3.485 ms** |
| Qwen3-30B-A3B / TE GroupedLinear 2 x 64 | NVFP4 | 7.845 ms | 21.424 ms |
| Qwen3-30B-A3B / TE dispatch + DeepGEMM | BF16 | 1.959 ms | 11.163 ms |
| Gemma 4 26B-A4B / **nvfp4moe** | NVFP4 x NVFP4 | **0.607 ms** | **4.157 ms** |
| Gemma 4 26B-A4B / TE GroupedLinear 2 x 64 | NVFP4 | 8.044 ms | 18.294 ms |
| Gemma 4 26B-A4B / TE dispatch + DeepGEMM | BF16 | 2.481 ms | 15.792 ms |

TE 2.17 limits grouped NVFP4 RHT to 64 tensors, so the E=128 comparison uses
two E=64 launches. The DeepGEMM BF16 path includes M-grouped forward and dgrad
plus K-grouped wgrad; it is a training comparison, not an inference-only row.

Full distributions, memory use, numerical results, and the exact measurement
method are in [BENCHMARKS.md](BENCHMARKS.md).

## 🧪 Training precision

FP4 training is approximate. With identical deterministic output gradients,
the custom path produced better cosine similarity than TE NVFP4 for all eight
gradient comparisons across the two captured layers.

| model / backend | dX | router grad | gate/up dW | down dW |
|---|---:|---:|---:|---:|
| Qwen / **nvfp4moe** | **0.97538** | **0.99236** | **0.96886** | **0.98673** |
| Qwen / TE NVFP4 | 0.97113 | 0.99013 | 0.96573 | 0.98586 |
| Gemma / **nvfp4moe** | **0.97186** | **0.99420** | **0.96577** | **0.98815** |
| Gemma / TE NVFP4 | 0.96754 | 0.99245 | 0.96237 | 0.98771 |

These are one-layer comparisons against BF16 PyTorch autograd. They do not by
themselves establish full-model convergence equivalence to BF16.

## 🚀 Install

The tested setup is:

- Linux with NVIDIA B200 (`sm100`)
- Python 3.12
- CUDA 13
- PyTorch 2.11 or newer
- NVIDIA CUTLASS DSL 4.6.x

The reference container is `nvcr.io/nvidia/pytorch:26.07-py3`.

```bash
python -m pip install .

# Tests and benchmark tooling
python -m pip install '.[test,benchmark]'
```

The patched QuACK sources required by the kernels are included in the package.

## 🔧 Training example

Routing selection remains in BF16/FP32. Dispatch receives detached routing
weights for its index path, while `differentiable_probs()` preserves the
values on the router's autograd graph.

```python
import torch
from nvfp4moe import MoEDispatch, MoEExpertLayer

T, d, I, E, k = 8192, 2048, 768, 128, 8

layer = MoEExpertLayer(
    d, I, E, k,
    rht=True,
    delayed_col_amax=True,
).cuda()
dispatch = MoEDispatch(T, E, k)

# BF16 master weights become resident NVFP4 formats.
layer.refresh_weights()

x = torch.randn(T, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
logits = torch.randn(T, E, device="cuda", dtype=torch.float32, requires_grad=True)
topv, topi = torch.topk(torch.softmax(logits, dim=-1), k, dim=-1)
topv = topv / topv.sum(dim=-1, keepdim=True)

gather_idx, cu, probs, slots = dispatch(topi.to(torch.int32), topv.detach())
probs_diff = dispatch.differentiable_probs(topv)

# Calibrate once for a representative shape before training.
layer.calibrate(x.detach(), gather_idx, cu, probs)
layer.sr_seed = 1234

y = layer(
    x, gather_idx, cu, probs_diff, slots,
    off_pad=dispatch.off_pad,
)
y.float().square().mean().backward()

# Requantize after the optimizer updates the BF16 master weights.
# optimizer.step()
# layer.refresh_weights()
```

`examples/train_toy_moe.py` includes router training, deterministic stochastic
rounding, and fused microbatch wgrad accumulation.

## ⚠️ Scope

- Single GPU only; EP, TP, DP, and communication are not included.
- Dimensions require `d % 256 == 0`, `I % 128 == 0`, `E <= 256`, and
  `T * topk <= 131072`.
- Optimizers, embeddings, LM heads, and distributed checkpoints stay outside
  this package.
- `delayed_col_amax=True` uses a one-step lag for columnwise wgrad scales.
  Disable it when same-step scale semantics are required.
- The first call compiles kernels and stores them in the QuACK JIT cache.

## 📄 License

Apache-2.0. Modified QuACK sources and NVIDIA-derived grouped-wgrad files keep
their upstream copyright and license headers. See `LICENSE` and `NOTICE` for
details.
