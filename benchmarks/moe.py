"""NVFP4 routed-expert benchmark with synthetic routing or captured trace replay.

The timed boundary includes dispatch, routed expert compute, routing-weight
application, and combine. Every backend receives the same assignments.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import random
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .model_shapes import (
    FULL_ROUTINGS,
    MODEL_SHAPES,
    QUICK_ROUTINGS,
    ModelShape,
    MoeCase,
    generate_moe_cases,
    parse_ints,
    parse_models,
    parse_names,
    routing_counts,
)

os.environ.setdefault("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")


BACKEND_NAMES = (
    "lightmoe",
    "transformer_engine_nvfp4_fused",
    "transformer_engine_nvfp4",
    "deepgemm_bf16",
    "deepgemm_fp8_fp4",
    "pytorch_bf16",
    "torchao_mxfp8",
)
SCOPES = ("full-layer",)
PASSES = ("fwd", "fwd_bwd")


@dataclass(frozen=True)
class BackendStatus:
    name: str
    discovered: bool
    runnable: bool
    scopes: tuple[str, ...]
    passes: tuple[str, ...]
    precision: str
    reason: str


@dataclass
class _TrainingArm:
    name: str
    call: Callable[[], object]
    x: object
    topv: object
    output: dict[str, object]
    weight_grads: Callable[[], dict[str, object]]


def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def detect_backends() -> dict[str, BackendStatus]:
    try:
        import torch
    except ImportError as exc:
        return {
            name: BackendStatus(name, False, False, (), (), "unknown", str(exc))
            for name in BACKEND_NAMES
        }
    cuda_ready = torch.cuda.is_available()
    blackwell = False
    sm100 = False
    if cuda_ready:
        try:
            capability = torch.cuda.get_device_capability()
            blackwell = capability[0] >= 10
            sm100 = capability == (10, 0)
        except (AssertionError, RuntimeError):
            pass
    base_reason = ""
    if not cuda_ready:
        base_reason = "CUDA is not available"
    elif not blackwell:
        base_reason = "compute capability 10.x or newer is required for NVFP4"

    lightmoe_discovered = False
    lightmoe_reason = base_reason or ("" if sm100 else "LightMoE requires SM100")
    if cuda_ready and sm100:
        try:
            package = importlib.import_module("lightmoe")
            lightmoe_discovered = callable(getattr(package, "MoEDispatch", None)) and callable(
                getattr(package, "MoEExpertLayer", None)
            )
            lightmoe_reason = (
                "LightMoE fused NVFP4 expert layer"
                if lightmoe_discovered
                else "MoEDispatch or MoEExpertLayer is missing"
            )
        except Exception as exc:  # noqa: BLE001
            lightmoe_reason = f"LightMoE import failed: {type(exc).__name__}: {exc}"

    grouped_mm = getattr(torch, "_grouped_mm", None)
    torch_discovered = callable(grouped_mm)
    torch_reason = base_reason or (
        "torch._grouped_mm BF16 reference"
        if torch_discovered
        else "torch._grouped_mm is not present"
    )

    te_discovered = False
    te_fused_discovered = False
    te_reason = base_reason
    if _module_exists("transformer_engine"):
        try:
            te = importlib.import_module("transformer_engine.pytorch")
            recipe = importlib.import_module("transformer_engine.common.recipe")
            te_discovered = callable(getattr(te, "GroupedLinear", None)) and callable(
                getattr(recipe, "NVFP4BlockScaling", None)
            )
            te_ops = importlib.import_module("transformer_engine.pytorch.ops")
            te_fused_discovered = te_discovered and all(
                callable(getattr(te_ops, name, None))
                for name in ("GroupedLinear", "ScaledSwiGLU", "Sequential")
            )
            te_reason = (
                "Transformer Engine NVFP4 GroupedLinear with permutation/combine"
                if te_discovered
                else "Transformer Engine GroupedLinear or NVFP4BlockScaling is missing"
            )
        except Exception as exc:  # noqa: BLE001
            te_reason = f"Transformer Engine import failed: {type(exc).__name__}: {exc}"
    elif not te_reason:
        te_reason = "transformer_engine is not installed"

    torchao_discovered = False
    torchao_reason = "TorchAO MXFP8 grouped GEMM is not installed"
    if _module_exists("torchao"):
        try:
            try:
                from .moe_backends import _torchao_mxfp8_grouped_mm
            except ImportError:
                from moe_backends import _torchao_mxfp8_grouped_mm

            torchao_discovered = callable(_torchao_mxfp8_grouped_mm())
            torchao_reason = "TorchAO dynamic MXFP8 scaled grouped GEMM"
        except Exception as exc:  # noqa: BLE001
            torchao_reason = f"TorchAO MXFP8 import failed: {type(exc).__name__}: {exc}"

    deepgemm_discovered = False
    deepgemm_reason = base_reason
    if _module_exists("deep_gemm"):
        try:
            deep_gemm = importlib.import_module("deep_gemm")
            deepgemm_discovered = all(
                callable(getattr(deep_gemm, name, None))
                for name in (
                    "m_grouped_bf16_gemm_nt_contiguous",
                    "k_grouped_bf16_gemm_tn_contiguous",
                    "m_grouped_fp8_fp4_gemm_nt_contiguous",
                )
            )
            deepgemm_reason = (
                "DeepGEMM grouped training and FP8 x FP4 forward"
                if deepgemm_discovered
                else "required grouped DeepGEMM callables are missing"
            )
        except Exception as exc:  # noqa: BLE001
            deepgemm_reason = f"DeepGEMM import failed: {type(exc).__name__}: {exc}"
    elif not deepgemm_reason:
        deepgemm_reason = "deep_gemm is not installed"

    return {
        "lightmoe": BackendStatus(
            "lightmoe",
            lightmoe_discovered,
            lightmoe_discovered and cuda_ready and sm100,
            SCOPES,
            PASSES,
            "NVFP4 x NVFP4",
            lightmoe_reason,
        ),
        "transformer_engine_nvfp4_fused": BackendStatus(
            "transformer_engine_nvfp4_fused",
            te_fused_discovered,
            te_fused_discovered and cuda_ready and sm100,
            SCOPES,
            PASSES,
            "Transformer Engine fused NVFP4",
            (
                "Transformer Engine 2.17 fused CUTLASS DSL grouped MLP"
                if te_fused_discovered
                else te_reason
            ),
        ),
        "transformer_engine_nvfp4": BackendStatus(
            "transformer_engine_nvfp4",
            te_discovered,
            te_discovered and cuda_ready and blackwell,
            SCOPES,
            PASSES,
            "Transformer Engine NVFP4",
            te_reason,
        ),
        "deepgemm_bf16": BackendStatus(
            "deepgemm_bf16",
            deepgemm_discovered,
            deepgemm_discovered and cuda_ready and blackwell,
            ("full-layer",),
            PASSES,
            "BF16",
            deepgemm_reason,
        ),
        "deepgemm_fp8_fp4": BackendStatus(
            "deepgemm_fp8_fp4",
            deepgemm_discovered,
            deepgemm_discovered and cuda_ready and blackwell,
            ("full-layer",),
            ("fwd",),
            "FP8 x FP4",
            deepgemm_reason,
        ),
        "pytorch_bf16": BackendStatus(
            "pytorch_bf16",
            torch_discovered,
            torch_discovered and cuda_ready,
            SCOPES,
            PASSES,
            "BF16",
            torch_reason,
        ),
        "torchao_mxfp8": BackendStatus(
            "torchao_mxfp8",
            torchao_discovered,
            torchao_discovered and te_discovered and cuda_ready and sm100,
            SCOPES,
            PASSES,
            "MXFP8",
            torchao_reason,
        ),
    }


def _measure_cuda(fn: Callable[[], object], warmup: int, iterations: int) -> dict[str, float | int]:
    import torch

    fn()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    wall_samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        wall_samples.append(1000 * (time.perf_counter() - wall_start))
    ordered = sorted(samples)
    q25 = ordered[int(0.25 * (len(ordered) - 1))]
    q75 = ordered[int(0.75 * (len(ordered) - 1))]
    health_ratio = sum(wall_samples) / sum(samples)
    return {
        "event_ms_p50": statistics.median(samples),
        "event_ms_p10": ordered[int(0.10 * (len(ordered) - 1))],
        "event_ms_p90": ordered[int(0.90 * (len(ordered) - 1))],
        "event_ms_p95": ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))],
        "event_ms_iqr": q75 - q25,
        "event_ms_min": min(samples),
        "host_wall_to_cuda_event_ratio": health_ratio,
        "health_valid": health_ratio <= 1.5,
        "iterations": iterations,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


def _synthetic_inputs(case: MoeCase) -> dict[str, object]:
    import torch

    torch.manual_seed(20260811)
    if case.topk > case.local_experts:
        raise ValueError(
            f"top-k {case.topk} exceeds {case.local_experts} local experts; "
            "synthetic complete-MoE-layer cases require unique local expert ids"
        )
    requested_counts = routing_counts(
        case.token_expert_assignments, case.local_experts, case.routing
    )
    weights = torch.tensor(requested_counts, dtype=torch.float64).clamp_min_(1)
    generator = torch.Generator().manual_seed(20260811)
    uniforms = torch.rand(
        case.tokens,
        case.local_experts,
        dtype=torch.float64,
        generator=generator,
    ).clamp_min_(torch.finfo(torch.float64).tiny)
    # Exponential keys sample without replacement while retaining routing skew.
    scores = -uniforms.log() / weights
    topi_cpu = scores.topk(case.topk, dim=1, largest=False).indices
    sorted_topi = topi_cpu.sort(1).values
    if case.topk > 1 and bool((sorted_topi[:, 1:] == sorted_topi[:, :-1]).any()):
        raise RuntimeError("synthetic router produced duplicate expert ids")
    actual_counts = torch.bincount(topi_cpu.reshape(-1), minlength=case.local_experts)
    topi = topi_cpu.to(torch.int32).cuda().contiguous()
    topv = torch.rand(case.tokens, case.topk, dtype=torch.float32, device="cuda")
    topv /= topv.sum(-1, keepdim=True)
    x = torch.randn(case.tokens, case.hidden, dtype=torch.bfloat16, device="cuda")
    first_projection = case.intermediate if case.activation == "relu2" else 2 * case.intermediate
    gate_up = (
        torch.randn(
            case.local_experts,
            first_projection,
            case.hidden,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * case.hidden**-0.5
    )
    down = (
        torch.randn(
            case.local_experts,
            case.hidden,
            case.intermediate,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * case.intermediate**-0.5
    )
    return {
        "expert_input": x,
        "topk_index": topi,
        "topk_weight": topv,
        "gate_up_weight": gate_up,
        "down_weight": down,
        "metadata": {
            "source": "synthetic",
            "routing": case.routing,
            "requested_counts": list(requested_counts),
            "actual_counts": actual_counts.tolist(),
            "unique_topk": True,
        },
    }


def _trace_inputs(case: MoeCase, path: str) -> dict[str, object]:
    import torch

    trace_path = Path(path).expanduser().resolve()
    if not trace_path.is_file():
        raise FileNotFoundError(trace_path)
    trace = torch.load(trace_path, map_location="cpu", weights_only=False)
    required = (
        "expert_input",
        "topk_index",
        "topk_weight",
        "gate_up_weight",
        "down_weight",
    )
    missing = [key for key in required if key not in trace]
    if missing:
        raise ValueError(f"trace is missing tensors: {missing}")
    x, topi = trace["expert_input"], trace["topk_index"]
    if tuple(x.shape) != (case.tokens, case.hidden):
        raise ValueError(f"trace input is {tuple(x.shape)}, expected {(case.tokens, case.hidden)}")
    if tuple(topi.shape) != (case.tokens, case.topk):
        raise ValueError(f"trace top-k is {tuple(topi.shape)}, expected {(case.tokens, case.topk)}")
    if trace["gate_up_weight"].shape[0] != case.local_experts:
        raise ValueError("trace must contain the selected EP shard's local experts")
    moved = {key: trace[key].cuda().contiguous() for key in required}
    moved["topk_index"] = moved["topk_index"].to(torch.int32)
    moved["topk_weight"] = moved["topk_weight"].to(torch.float32)
    moved["metadata"] = {
        "source": "trace",
        "path": str(trace_path),
        "capture": trace.get("metadata", {}),
    }
    return moved


def _interleave_gate_up(weight):
    import torch

    gate, up = weight.chunk(2, dim=1)
    return torch.stack((gate, up), dim=2).flatten(1, 2).contiguous()


class _TorchGroupedExperts:
    def __init__(self, spec: ModelShape, inputs: dict[str, object]):
        import torch

        self.torch = torch
        self.w1 = inputs["gate_up_weight"].transpose(1, 2).contiguous().requires_grad_(True)
        self.w2 = inputs["down_weight"].transpose(1, 2).contiguous().requires_grad_(True)
        self.experts = spec.experts
        self.topk = spec.topk
        self.activation = spec.activation
        self.activation_clamp = spec.activation_clamp

    def zero_grad(self):
        self.w1.grad = None
        self.w2.grad = None

    def prepare(self, topi, topv):
        torch = self.torch
        flat = topi.reshape(-1).long()
        order = torch.argsort(flat, stable=True)
        token = order // self.topk
        expert = flat[order]
        counts = torch.bincount(expert, minlength=self.experts)
        offsets = counts.cumsum(0).to(torch.int32)
        probs = topv.reshape(-1)[order]
        return token, offsets, probs, order

    def expert_core(self, x, offsets):
        torch = self.torch
        h = torch._grouped_mm(x, self.w1, offs=offsets)
        if self.activation == "relu2":
            hh = torch.relu(h).square()
            return torch._grouped_mm(hh, self.w2, offs=offsets)
        gate, up = h.chunk(2, dim=-1)
        if self.activation == "swiglu_oai":
            gate = gate.clamp(max=7)
            up = up.clamp(min=-7, max=7)
            hh = gate * torch.sigmoid(1.702 * gate) * (up + 1)
        elif self.activation_clamp is not None:
            gate = gate.clamp(max=self.activation_clamp)
            up = up.clamp(min=-self.activation_clamp, max=self.activation_clamp)
            hh = torch.nn.functional.silu(gate) * up
        else:
            hh = torch.nn.functional.silu(gate) * up
        return torch._grouped_mm(hh, self.w2, offs=offsets)

    def full_layer(self, x, topi, topv):
        token, offsets, probs, _ = self.prepare(topi, topv)
        yp = self.expert_core(x[token], offsets).float() * probs[:, None]
        y = self.torch.zeros_like(x, dtype=self.torch.float32)
        y.index_add_(0, token, yp)
        return y.to(x.dtype)


def _lightmoe_backend(spec: ModelShape, inputs: dict[str, object]):
    try:
        from .moe_backends import LightMoEExpert
    except ImportError:
        from moe_backends import LightMoEExpert

    backend = LightMoEExpert(spec, inputs)
    return backend


def _te_backend(spec: ModelShape, inputs: dict[str, object]):
    try:
        from .moe_backends import TEExpert
    except ImportError:
        from moe_backends import TEExpert

    return TEExpert(spec, inputs, nvfp4=True)


def _backend_matches_activation(backend_name: str, case: MoeCase) -> bool:
    if backend_name in ("lightmoe", "pytorch_bf16"):
        return True
    spec = MODEL_SHAPES[case.model]
    return spec.activation == "swiglu" and spec.activation_clamp is None


def _crop_training_grads(grads, intermediate, unary=False):
    gate_up = grads["gate_up"]
    if unary:
        gate_up = gate_up[:, :intermediate]
    else:
        half = gate_up.shape[1] // 2
        gate_up = __import__("torch").cat(
            (gate_up[:, :intermediate], gate_up[:, half : half + intermediate]), dim=1
        )
    return {
        "gate_up": gate_up.contiguous(),
        "down": grads["down"][:, :, :intermediate].contiguous(),
    }


def _make_training_arm(backend_name, case, inputs, dout):
    global_spec = MODEL_SHAPES[case.model]
    spec = replace(global_spec, experts=case.local_experts, ep_sizes=(1,), quick_ep=1)
    x = inputs["expert_input"].detach().clone().requires_grad_(True)
    topv = inputs["topk_weight"].detach().clone().requires_grad_(True)

    if backend_name == "lightmoe":
        backend = _lightmoe_backend(spec, inputs)
    elif backend_name == "transformer_engine_nvfp4_fused":
        try:
            from .moe_backends import TEFusedExpert
        except ImportError:
            from moe_backends import TEFusedExpert

        backend = TEFusedExpert(spec, inputs)
    elif backend_name == "transformer_engine_nvfp4":
        backend = _te_backend(spec, inputs)
    elif backend_name == "deepgemm_bf16":
        try:
            from .moe_backends import TEDeepGEMMTrainExpert
        except ImportError:
            from moe_backends import TEDeepGEMMTrainExpert

        backend = TEDeepGEMMTrainExpert(spec, inputs)
    elif backend_name == "torchao_mxfp8":
        try:
            from .moe_backends import TorchAOMXFP8Expert
        except ImportError:
            from moe_backends import TorchAOMXFP8Expert

        backend = TorchAOMXFP8Expert(spec, inputs)
    elif backend_name == "pytorch_bf16":
        backend = _TorchGroupedExperts(spec, inputs)
    else:
        raise ValueError(f"{backend_name} does not provide full training")

    if backend_name == "pytorch_bf16":

        def forward(step):
            del step
            return backend.full_layer(x, inputs["topk_index"], topv)

        zero_grad = backend.zero_grad

        def raw_weight_grads():
            return {
                "gate_up": backend.w1.grad.transpose(1, 2),
                "down": backend.w2.grad.transpose(1, 2),
            }

    else:

        def forward(step):
            return backend(x, inputs["topk_index"], topv, step)

        zero_grad = lambda: backend.zero_grad(set_to_none=True)
        raw_weight_grads = backend.training_gradients

    output = {}
    step = [0]

    def call():
        zero_grad()
        x.grad = None
        topv.grad = None
        step[0] += 1
        y = forward(step[0])
        y.backward(dout)
        output["y"] = y.detach()
        return y

    def weight_grads():
        return _crop_training_grads(
            raw_weight_grads(), case.intermediate, unary=case.activation == "relu2"
        )

    return _TrainingArm(backend_name, call, x, topv, output, weight_grads)


def _sample_error(actual, reference, limit=65_536):
    import torch

    if actual is None or reference is None or actual.shape != reference.shape:
        return {"comparable": False}
    a = actual.detach().float().reshape(-1)
    b = reference.detach().float().reshape(-1)
    stride = max(1, (a.numel() + limit - 1) // limit)
    a = a[::stride]
    b = b[::stride]
    diff = a - b
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    cosine = torch.dot(a, b) / denom if denom else a.new_tensor(1.0)
    ref_norm = torch.linalg.vector_norm(b).clamp_min(1e-12)
    return {
        "comparable": True,
        "sample_count": a.numel(),
        "cosine": float(cosine),
        "relative_l2": float(torch.linalg.vector_norm(diff) / ref_norm),
        "max_abs": float(diff.abs().max()),
    }


def _timing_summary(samples, walls, case, peak_allocated_gib):
    ordered = sorted(samples)
    q25 = ordered[int(0.25 * (len(ordered) - 1))]
    q75 = ordered[int(0.75 * (len(ordered) - 1))]
    health_ratio = sum(walls) / sum(samples)
    p50 = statistics.median(samples)
    return {
        "event_ms_p50": p50,
        "event_ms_p10": ordered[int(0.10 * (len(ordered) - 1))],
        "event_ms_p90": ordered[int(0.90 * (len(ordered) - 1))],
        "event_ms_iqr": q75 - q25,
        "event_ms_min": min(samples),
        "host_wall_to_cuda_event_ratio": health_ratio,
        "health_valid": health_ratio <= 1.5,
        "iterations": len(samples),
        "peak_allocated_gib": peak_allocated_gib,
        "tokens_per_second": case.tokens * 1000 / p50,
        "token_expert_assignments_per_second": case.token_expert_assignments * 1000 / p50,
    }


def _profile_training_arm(arm):
    import torch

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        arm.call()
    torch.cuda.synchronize()
    kernels = []
    for event in prof.key_averages():
        device_us = getattr(event, "self_device_time_total", 0.0)
        if device_us <= 0:
            device_us = getattr(event, "self_cuda_time_total", 0.0)
        if device_us > 0:
            kernels.append(
                {
                    "name": event.key,
                    "device_us": float(device_us),
                    "count": int(event.count),
                }
            )
    kernels.sort(key=lambda item: item["device_us"], reverse=True)
    return kernels


def _run_nvtx_training_arm(backend_name, case, inputs):
    import torch

    torch.manual_seed(20260812)
    arm = _make_training_arm(backend_name, case, inputs, torch.randn_like(inputs["expert_input"]))
    arm.call()
    arm.call()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("lightmoe_training_profile")
    arm.call()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


def _run_interleaved_training(backends, case, inputs, warmup, iterations, profile_kernels=False):
    import torch

    torch.manual_seed(20260812)
    dout = torch.randn_like(inputs["expert_input"])
    arms = {}
    results = {}
    for backend_name in backends:
        if not _backend_matches_activation(backend_name, case):
            results[backend_name] = {
                "status": "skipped",
                "reason": "backend does not implement the model activation contract",
            }
            continue
        try:
            arms[backend_name] = _make_training_arm(backend_name, case, inputs, dout)
        except Exception as exc:  # noqa: BLE001
            results[backend_name] = {
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
            }
    if not arms:
        return results

    for name in list(arms):
        arm = arms[name]
        try:
            arm.call()
            for _ in range(warmup):
                arm.call()
        except Exception as exc:  # noqa: BLE001
            results[name] = {
                "status": "error",
                "reason": f"warmup failed: {type(exc).__name__}: {exc}",
            }
            del arms[name]
    if not arms:
        return results
    torch.cuda.synchronize()
    names = list(arms)

    def stabilize():
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            for name in names:
                arms[name].call()
            torch.cuda.synchronize()

    def measure():
        samples = {name: [] for name in arms}
        walls = {name: [] for name in arms}
        rng = random.Random(20260812)
        for _ in range(iterations):
            order = names.copy()
            rng.shuffle(order)
            for name in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                wall_start = time.perf_counter()
                start.record()
                arms[name].call()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end))
                walls[name].append(1000 * (time.perf_counter() - wall_start))

        repeat_samples = []
        for _ in range(iterations):
            order = names.copy()
            rng.shuffle(order)
            for name in order:
                if name != names[0]:
                    arms[name].call()
                    torch.cuda.synchronize()
                    continue
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                arms[name].call()
                end.record()
                end.synchronize()
                repeat_samples.append(start.elapsed_time(end))
        return samples, walls, repeat_samples

    stabilize()
    torch.cuda.reset_peak_memory_stats()
    for measurement_attempts in range(1, 4):
        samples, walls, repeat_samples = measure()
        initial_p50 = statistics.median(samples[names[0]])
        repeat_p50 = statistics.median(repeat_samples)
        initial_to_repeat_median_deviation = abs(repeat_p50 / initial_p50 - 1.0)
        repeat_deviation_valid = initial_to_repeat_median_deviation <= 0.05
        if repeat_deviation_valid:
            break
        stabilize()

    first = names[0]
    peak = torch.cuda.max_memory_allocated() / 2**30

    kernel_profiles = {}
    if profile_kernels:
        for name, arm in arms.items():
            kernel_profiles[name] = _profile_training_arm(arm)
            for item in kernel_profiles[name][:24]:
                print(
                    json.dumps(
                        {
                            "event": "kernel_profile",
                            "backend": name,
                            "device_us": item["device_us"],
                            "count": item["count"],
                            "name": item["name"][:240],
                        }
                    ),
                    flush=True,
                )

    reference = arms.get("pytorch_bf16")
    reference_grads = reference.weight_grads() if reference is not None else None
    for name, arm in arms.items():
        timing = _timing_summary(samples[name], walls[name], case, peak)
        accuracy = None
        if reference is not None:
            grads = arm.weight_grads()
            accuracy = {
                "output": _sample_error(arm.output.get("y"), reference.output.get("y")),
                "input_grad": _sample_error(arm.x.grad, reference.x.grad),
                "router_grad": _sample_error(arm.topv.grad, reference.topv.grad),
                "gate_up_grad": _sample_error(grads["gate_up"], reference_grads["gate_up"]),
                "down_grad": _sample_error(grads["down"], reference_grads["down"]),
            }
        results[name] = {
            "status": "ok",
            "timing": timing,
            "accuracy_vs_pytorch_bf16": accuracy,
            **({"kernels": kernel_profiles[name]} if profile_kernels else {}),
            "gradient_status": {
                "input": arm.x.grad is not None and bool(torch.isfinite(arm.x.grad).all()),
                "router_weight": arm.topv.grad is not None
                and bool(torch.isfinite(arm.topv.grad).all()),
            },
            "session": {
                "interleaved": True,
                "first_arm": first,
                "repeat_p50_ms": repeat_p50,
                "initial_to_repeat_median_deviation": (initial_to_repeat_median_deviation),
                "repeat_deviation_valid": repeat_deviation_valid,
                "measurement_attempts": measurement_attempts,
                "valid": repeat_deviation_valid
                and all(sum(walls[key]) / sum(samples[key]) <= 1.5 for key in arms),
            },
        }
    return results


def _run_case(
    backend_name: str,
    case: MoeCase,
    scope: str,
    benchmark_pass: str,
    inputs: dict[str, object],
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    import torch

    if not _backend_matches_activation(backend_name, case):
        return {
            "status": "skipped",
            "reason": "backend does not implement the model activation contract",
        }
    global_spec = MODEL_SHAPES[case.model]
    spec = replace(global_spec, experts=case.local_experts, ep_sizes=(1,), quick_ep=1)
    if backend_name == "transformer_engine_nvfp4_fused" and case.activation != "swiglu":
        return {
            "status": "skipped",
            "reason": "Transformer Engine fused grouped MLP supports SwiGLU only",
        }
    if scope != "full-layer":
        raise ValueError(scope)
    if backend_name == "lightmoe":
        backend = _lightmoe_backend(spec, inputs)

        def forward(x, tv, step):
            return backend(x, inputs["topk_index"], tv, step)

        zero_grad = lambda: backend.zero_grad(set_to_none=True)
    elif backend_name == "transformer_engine_nvfp4":
        backend = _te_backend(spec, inputs)

        def forward(x, tv, step):
            return backend(x, inputs["topk_index"], tv, step)

        zero_grad = lambda: backend.zero_grad(set_to_none=True)
    elif backend_name == "transformer_engine_nvfp4_fused":
        try:
            from .moe_backends import TEFusedExpert
        except ImportError:
            from moe_backends import TEFusedExpert

        backend = TEFusedExpert(spec, inputs)

        def forward(x, tv, step):
            return backend(x, inputs["topk_index"], tv, step)

        zero_grad = lambda: backend.zero_grad(set_to_none=True)
    elif backend_name == "torchao_mxfp8":
        try:
            from .moe_backends import TorchAOMXFP8Expert
        except ImportError:
            from moe_backends import TorchAOMXFP8Expert

        backend = TorchAOMXFP8Expert(spec, inputs)

        def forward(x, tv, step):
            return backend(x, inputs["topk_index"], tv, step)

        zero_grad = lambda: backend.zero_grad(set_to_none=True)
    elif backend_name in ("deepgemm_bf16", "deepgemm_fp8_fp4"):
        try:
            from .moe_backends import (
                TEDeepGEMMFP8FP4Expert,
                TEDeepGEMMTrainExpert,
            )
        except ImportError:
            from moe_backends import TEDeepGEMMFP8FP4Expert, TEDeepGEMMTrainExpert

        backend_type = (
            TEDeepGEMMTrainExpert if backend_name == "deepgemm_bf16" else TEDeepGEMMFP8FP4Expert
        )
        backend = backend_type(spec, inputs)

        def forward(x, tv, step):
            return backend(x, inputs["topk_index"], tv, step)

        zero_grad = lambda: backend.zero_grad(set_to_none=True)
    elif backend_name == "pytorch_bf16":
        backend = _TorchGroupedExperts(spec, inputs)

        def forward(x, tv, step):
            del step
            return backend.full_layer(x, inputs["topk_index"], tv)

        zero_grad = backend.zero_grad
    else:
        raise ValueError(backend_name)

    needs_grad = benchmark_pass == "fwd_bwd"
    x = inputs["expert_input"].detach().clone().requires_grad_(needs_grad)
    topv = inputs["topk_weight"].detach().clone().requires_grad_(needs_grad)
    dout = torch.randn_like(x) if needs_grad else None
    step = [0]

    def fwd():
        step[0] += 1
        return forward(x, topv, step[0])

    def fwd_bwd():
        zero_grad()
        x.grad = None
        topv.grad = None
        y = fwd()
        y.backward(dout)
        return y

    timing = _measure_cuda(fwd_bwd if needs_grad else fwd, warmup, iterations)
    p50 = float(timing["event_ms_p50"])
    timing["tokens_per_second"] = case.tokens * 1000 / p50
    timing["token_expert_assignments_per_second"] = case.token_expert_assignments * 1000 / p50
    result = {"status": "ok", "timing": timing}
    if needs_grad:
        result["gradient_status"] = {
            "input": x.grad is not None and bool(torch.isfinite(x.grad).all()),
            "router_weight": topv.grad is not None and bool(torch.isfinite(topv.grad).all()),
        }
    return result


def _case_dict(case: MoeCase) -> dict[str, object]:
    return {**asdict(case), "label": case.label}


def listing_payload(args: argparse.Namespace) -> dict[str, object]:
    models = parse_models(args.models, layer_only=True)
    if args.source == "trace" and len(models) != 1:
        raise ValueError("trace replay accepts exactly one --models key")
    tokens = (
        (8192,)
        if args.source == "trace" and args.tokens is None
        else parse_ints(args.tokens, args.suite)
    )
    routings = parse_names(
        args.routing,
        QUICK_ROUTINGS if args.suite == "quick" else FULL_ROUTINGS,
        "routing",
    )
    cases = generate_moe_cases(models, tokens, args.suite, routings, args.source)
    return {
        "benchmark": "routed_expert_layer",
        "suite": args.suite,
        "source": {
            "kind": args.source,
            "trace": str(Path(args.trace).expanduser()) if args.trace else None,
        },
        "boundaries": {
            "full-layer": (
                "dispatch, routed expert compute, routing-weight application, and combine"
            ),
            "excluded": (
                "router logits, top-k selection, shared experts, and model-specific outer projections"
            ),
        },
        "models": {key: asdict(MODEL_SHAPES[key]) for key in models},
        "backends": {name: asdict(status) for name, status in detect_backends().items()},
        "case_count": len(cases),
        "cases": [_case_dict(case) for case in cases],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print matrix and availability only")
    parser.add_argument("--suite", choices=("quick", "full"), default="quick")
    parser.add_argument("--models", default="all", help="comma-separated registry keys")
    parser.add_argument("--tokens", default=None, help="comma-separated token counts")
    parser.add_argument(
        "--backends",
        default=(
            "lightmoe,transformer_engine_nvfp4_fused,transformer_engine_nvfp4,"
            "deepgemm_bf16,deepgemm_fp8_fp4,pytorch_bf16,torchao_mxfp8"
        ),
    )
    parser.add_argument("--scope", choices=SCOPES, default="full-layer")
    parser.add_argument("--pass", dest="benchmark_pass", choices=(*PASSES, "both"), default="fwd")
    parser.add_argument("--source", choices=("synthetic", "trace"), default="synthetic")
    parser.add_argument(
        "--trace", help="captured .pt trace; required with --source trace when running"
    )
    parser.add_argument("--routing", default="all")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--interleave-training",
        action="store_true",
        help=(
            "interleave complete MoE forward/backward arms and repeat the first arm "
            "to check timing stability"
        ),
    )
    parser.add_argument("--profile-kernels", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--profile-nvtx-arm", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-cases", type=int, default=0, help="0 runs the complete selected matrix"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = listing_payload(args)
        selected = parse_names(args.backends, BACKEND_NAMES, "backend")
    except ValueError as exc:
        parser.error(str(exc))
    if args.list:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.source == "trace" and not args.trace:
        parser.error("--trace is required when running --source trace")
    if args.warmup < 0 or args.iterations <= 0 or args.max_cases < 0:
        parser.error("warmup/max-cases must be non-negative and iterations must be positive")
    if args.interleave_training and not (
        args.scope == "full-layer" and args.benchmark_pass == "fwd_bwd"
    ):
        parser.error("--interleave-training requires --scope full-layer --pass fwd_bwd")

    cases = [
        MoeCase(**{key: value for key, value in row.items() if key != "label"})
        for row in payload["cases"]
    ]
    if args.max_cases:
        cases = cases[: args.max_cases]
    scopes = (args.scope,)
    passes = PASSES if args.benchmark_pass == "both" else (args.benchmark_pass,)
    statuses = detect_backends()
    print(
        json.dumps(
            {
                "event": "start",
                "benchmark": payload["benchmark"],
                "suite": args.suite,
                "source": payload["source"],
                "boundaries": payload["boundaries"],
                "selected_backends": selected,
            }
        ),
        flush=True,
    )
    started = time.time()
    for case in cases:
        has_runnable = any(
            statuses[backend].runnable
            and any(scope in statuses[backend].scopes for scope in scopes)
            and any(benchmark_pass in statuses[backend].passes for benchmark_pass in passes)
            for backend in selected
        )
        if not has_runnable:
            for scope in scopes:
                for benchmark_pass in passes:
                    for backend in selected:
                        status = statuses[backend]
                        reason = (
                            status.reason if not status.runnable else "scope or pass is unsupported"
                        )
                        print(
                            json.dumps(
                                {
                                    "event": "result",
                                    "backend": backend,
                                    "precision": status.precision,
                                    "scope": scope,
                                    "pass": benchmark_pass,
                                    "source_metadata": {
                                        "source": args.source,
                                        "setup": "not_needed_for_skipped_backends",
                                    },
                                    "case": _case_dict(case),
                                    "status": "skipped",
                                    "reason": reason,
                                }
                            ),
                            flush=True,
                        )
            continue
        try:
            inputs = (
                _trace_inputs(case, args.trace)
                if args.source == "trace"
                else _synthetic_inputs(case)
            )
            source_metadata = inputs["metadata"]
        except Exception as exc:  # noqa: BLE001
            for backend in selected:
                print(
                    json.dumps(
                        {
                            "event": "result",
                            "backend": backend,
                            "case": _case_dict(case),
                            "status": "error",
                            "reason": f"input setup failed: {type(exc).__name__}: {exc}",
                        }
                    ),
                    flush=True,
                )
            continue
        if args.interleave_training:
            runnable = [
                backend
                for backend in selected
                if statuses[backend].runnable
                and "full-layer" in statuses[backend].scopes
                and "fwd_bwd" in statuses[backend].passes
            ]
            if args.profile_nvtx_arm is not None:
                if args.profile_nvtx_arm not in runnable:
                    raise RuntimeError(f"profile backend is unavailable: {args.profile_nvtx_arm}")
                _run_nvtx_training_arm(args.profile_nvtx_arm, case, inputs)
                return 0
            interleaved = _run_interleaved_training(
                runnable,
                case,
                inputs,
                args.warmup,
                args.iterations,
                args.profile_kernels,
            )
            for backend in selected:
                status = statuses[backend]
                base = {
                    "event": "result",
                    "backend": backend,
                    "precision": status.precision,
                    "scope": "full-layer",
                    "pass": "fwd_bwd",
                    "source_metadata": source_metadata,
                    "case": _case_dict(case),
                }
                if backend in interleaved:
                    result = interleaved[backend]
                elif not status.runnable:
                    result = {"status": "skipped", "reason": status.reason}
                else:
                    result = {"status": "skipped", "reason": "full training is unsupported"}
                print(json.dumps({**base, **result}, default=str), flush=True)
            continue
        for scope in scopes:
            for benchmark_pass in passes:
                for backend in selected:
                    status = statuses[backend]
                    base = {
                        "event": "result",
                        "backend": backend,
                        "precision": status.precision,
                        "scope": scope,
                        "pass": benchmark_pass,
                        "source_metadata": source_metadata,
                        "case": _case_dict(case),
                    }
                    if not status.runnable:
                        result = {"status": "skipped", "reason": status.reason}
                    elif scope not in status.scopes or benchmark_pass not in status.passes:
                        result = {"status": "skipped", "reason": "scope or pass is unsupported"}
                    else:
                        try:
                            result = _run_case(
                                backend,
                                case,
                                scope,
                                benchmark_pass,
                                inputs,
                                args.warmup,
                                args.iterations,
                            )
                        except Exception as exc:  # noqa: BLE001
                            result = {
                                "status": "error",
                                "reason": f"{type(exc).__name__}: {exc}",
                            }
                    print(json.dumps({**base, **result}, default=str), flush=True)
    print(json.dumps({"event": "done", "wall_seconds": time.time() - started}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
