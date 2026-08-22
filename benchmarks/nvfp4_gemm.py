"""Standalone grouped NVFP4 GEMM benchmark.

The prepacked mode measures only GEMM launch and execution. Dynamic mode adds
activation quantization while keeping resident expert weights prepacked. Every
row is either produced by the named backend or reported as skipped; this script
never substitutes another implementation for a missing backend.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import random
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from lightmoe.gemm import quantize

from .model_shapes import (
    FULL_ROUTINGS,
    MODEL_SHAPES,
    QUICK_ROUTINGS,
    GemmCase,
    generate_gemm_cases,
    parse_ints,
    parse_models,
    parse_names,
    routing_counts,
)

BACKEND_NAMES = (
    "lightmoe",
    "flashinfer_cutedsl",
    "torch_scaled_grouped_mm",
    "transformer_engine_nvfp4",
    "cutlass",
    "cublaslt",
)
DENSE_BACKEND_NAMES = ("lightmoe", "lightmoe_grouped", "cublaslt")
DIRECTIONS = ("fwd", "dgrad", "wgrad")
MODES = ("prepacked", "dynamic")
B200_DENSE_FP4_PEAK_TFLOPS = 9_000.0
B200_PEAK_SOURCE = "NVIDIA DGX B200: 72 dense FP4 PFLOP/s per 8-GPU system"


@dataclass(frozen=True)
class BackendStatus:
    name: str
    discovered: bool
    runnable: bool
    modes: tuple[str, ...]
    capabilities: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class DenseCase:
    model: str
    projection: str
    m: int
    n: int
    k: int

    @property
    def label(self) -> str:
        return f"{self.model}:{self.projection}:m{self.m}:n{self.n}:k{self.k}"

    @property
    def flops(self) -> int:
        return 2 * self.m * self.n * self.k


def _runtime_metadata(peak_override: float | None) -> dict[str, object]:
    import torch

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    device_name = properties.name
    if peak_override is not None:
        peak_tflops = peak_override
        peak_source = "--theoretical-peak-tflops override"
    elif "B200" in device_name.upper():
        peak_tflops = B200_DENSE_FP4_PEAK_TFLOPS
        peak_source = B200_PEAK_SOURCE
    else:
        peak_tflops = None
        peak_source = "unknown GPU; pass --theoretical-peak-tflops"

    package_versions = {}
    for package in ("nvidia-cutlass-dsl", "flashinfer-python", "transformer-engine"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "gpu_name": device_name,
        "compute_capability": list(torch.cuda.get_device_capability()),
        "sm_count": properties.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "packages": package_versions,
        "theoretical_peak_tflops": peak_tflops,
        "theoretical_peak_basis": peak_source,
        "peak_uses_structured_sparsity": False,
        "flop_convention": "one multiply-add is two FLOPs",
    }


def _annotate_throughput(
    timing: dict[str, object],
    logical_flops: int,
    theoretical_peak_tflops: float | None,
    *,
    gemm_only: bool,
) -> None:
    tflops = logical_flops / (float(timing["event_ms_p50"]) * 1e9)
    timing["logical_flops"] = logical_flops
    if gemm_only:
        timing["logical_tflops"] = tflops
        if theoretical_peak_tflops is not None:
            timing["theoretical_peak_tflops"] = theoretical_peak_tflops
            timing["dense_fp4_spec_peak_pct"] = 100.0 * tflops / theoretical_peak_tflops
    else:
        timing["equivalent_logical_tflops"] = tflops


def _tile_rounded_flops(
    counts: tuple[int, ...],
    n: int,
    k: int,
    tile_m: int,
    tile_n: int,
    tile_k: int = 256,
) -> int:
    padded_m = sum(-(-rows // tile_m) * tile_m for rows in counts)
    padded_n = -(-n // tile_n) * tile_n
    padded_k = -(-k // tile_k) * tile_k
    return 2 * padded_m * padded_n * padded_k


def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _dense_cublas_available() -> bool:
    try:
        from torch.nn.functional import ScalingType, SwizzleType, scaled_mm
    except (ImportError, AttributeError):
        return False
    return callable(scaled_mm) and ScalingType is not None and SwizzleType is not None


def detect_backends() -> dict[str, BackendStatus]:
    try:
        import torch
    except ImportError as exc:
        reason = f"PyTorch import failed: {exc}"
        return {name: BackendStatus(name, False, False, (), (), reason) for name in BACKEND_NAMES}

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
    cuda_reason = "" if cuda_ready else "CUDA is not available"
    arch_reason = "" if blackwell else "compute capability 10.x or newer is required"

    lightmoe_callable = False
    lightmoe_reason = cuda_reason or ("" if sm100 else "LightMoE requires SM100")
    if cuda_ready and sm100:
        try:
            module = importlib.import_module("lightmoe.kernels.grouped.runtime")
            quant = importlib.import_module("lightmoe.kernels.quantize")
            lightmoe_callable = callable(getattr(module, "grouped_nvfp4_gemm", None)) and callable(
                getattr(quant, "nvfp4_quantize_rowwise", None)
            )
            if not lightmoe_callable:
                lightmoe_reason = "LightMoE GEMM or quantizer callable is missing"
        except Exception as exc:  # noqa: BLE001
            lightmoe_reason = f"LightMoE import failed: {type(exc).__name__}: {exc}"

    scaled = getattr(torch.nn.functional, "scaled_grouped_mm", None)
    torch_discovered = callable(scaled)
    torch_reason = cuda_reason or arch_reason
    if torch_discovered and cuda_ready and blackwell:
        torch_reason = "PyTorch scaled_grouped_mm with NVFP4 block scales"
    elif not torch_discovered:
        torch_reason = "torch.nn.functional.scaled_grouped_mm is not present"

    flashinfer_discovered = _module_exists("flashinfer")
    flashinfer_runnable = False
    flashinfer_reason = "flashinfer is not installed"
    if flashinfer_discovered:
        try:
            flashinfer = importlib.import_module("flashinfer")
            flashinfer_gemm = importlib.import_module("flashinfer.cute_dsl.blockscaled_gemm")
            flashinfer_runnable = (
                cuda_ready
                and sm100
                and callable(getattr(flashinfer, "scaled_fp4_grouped_quantize", None))
                and callable(getattr(flashinfer_gemm, "grouped_gemm_nt_masked", None))
            )
            flashinfer_reason = (
                "FlashInfer CuTe DSL grouped_gemm_nt_masked"
                if flashinfer_runnable
                else "FlashInfer grouped NVFP4 callable is missing or the GPU is not SM100"
            )
        except Exception as exc:  # noqa: BLE001
            flashinfer_reason = f"FlashInfer import failed: {type(exc).__name__}: {exc}"

    te_discovered = False
    te_runnable = False
    te_reason = cuda_reason or arch_reason
    if _module_exists("transformer_engine"):
        try:
            te = importlib.import_module("transformer_engine.pytorch")
            recipe = importlib.import_module("transformer_engine.common.recipe")
            te_discovered = callable(getattr(te, "GroupedLinear", None)) and callable(
                getattr(recipe, "NVFP4BlockScaling", None)
            )
            te_runnable = te_discovered and cuda_ready and blackwell
            if not te_discovered:
                te_reason = "Transformer Engine GroupedLinear or NVFP4BlockScaling is missing"
            elif te_runnable:
                te_reason = "Transformer Engine dynamic NVFP4 GroupedLinear"
        except Exception as exc:  # noqa: BLE001
            te_reason = f"Transformer Engine import failed: {type(exc).__name__}: {exc}"
    elif not te_reason:
        te_reason = "transformer_engine is not installed"

    cutlass_discovered = False
    cutlass_reason = "CUTLASS Python package is not installed"
    if _module_exists("cutlass"):
        try:
            cutlass = importlib.import_module("cutlass")
            candidates = (
                getattr(cutlass, "grouped_scaled_mm", None),
                getattr(getattr(cutlass, "op", None), "GroupedScaledGemm", None),
            )
            cutlass_discovered = any(callable(candidate) for candidate in candidates)
            cutlass_reason = (
                "public grouped NVFP4 callable found, but this repository has no stable "
                "versioned invocation adapter"
                if cutlass_discovered
                else "no public grouped NVFP4 callable was found"
            )
        except Exception as exc:  # noqa: BLE001
            cutlass_reason = f"CUTLASS import failed: {type(exc).__name__}: {exc}"

    cublas_discovered = False
    cublas_reason = "no Python cuBLASLt grouped NVFP4 callable was found"
    for module_name in ("cuda.bindings.cublas", "nvidia.cublas"):
        if not _module_exists(module_name):
            continue
        try:
            cublas = importlib.import_module(module_name)
            names = ("cublasLtMatmulGroup", "grouped_nvfp4_gemm", "nvfp4_grouped_gemm")
            if any(callable(getattr(cublas, name, None)) for name in names):
                cublas_discovered = True
                cublas_reason = (
                    "grouped callable found, but no typed NVFP4 Python invocation adapter is "
                    "available in this repository"
                )
                break
        except Exception as exc:  # noqa: BLE001
            cublas_reason = f"{module_name} import failed: {type(exc).__name__}: {exc}"
            continue

    return {
        "lightmoe": BackendStatus(
            "lightmoe",
            lightmoe_callable,
            lightmoe_callable and cuda_ready and sm100,
            MODES,
            DIRECTIONS,
            lightmoe_reason or "LightMoE grouped NVFP4 kernels",
        ),
        "flashinfer_cutedsl": BackendStatus(
            "flashinfer_cutedsl",
            flashinfer_discovered,
            flashinfer_runnable,
            ("prepacked",),
            ("fwd",),
            flashinfer_reason,
        ),
        "torch_scaled_grouped_mm": BackendStatus(
            "torch_scaled_grouped_mm",
            torch_discovered,
            torch_discovered and cuda_ready and sm100 and flashinfer_runnable,
            ("prepacked",),
            ("fwd",),
            torch_reason,
        ),
        "transformer_engine_nvfp4": BackendStatus(
            "transformer_engine_nvfp4",
            te_discovered,
            te_runnable,
            ("dynamic",),
            DIRECTIONS,
            te_reason,
        ),
        "cutlass": BackendStatus(
            "cutlass",
            cutlass_discovered,
            False,
            ("prepacked",),
            ("fwd",),
            cutlass_reason,
        ),
        "cublaslt": BackendStatus(
            "cublaslt",
            cublas_discovered,
            False,
            ("prepacked",),
            ("fwd",),
            cublas_reason,
        ),
    }


def _measure_cuda(fn: Callable[[], object], warmup: int, iterations: int) -> dict[str, float | int]:
    import torch

    fn()  # Compile and allocate before warmup.
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iterations)
    ]
    wall_start = time.perf_counter()
    for start, end in events:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1e3
    samples = [start.elapsed_time(end) for start, end in events]
    return _summarize_cuda_samples(samples, wall_ms, sum(samples))


def _summarize_cuda_samples(
    samples: list[float],
    wall_ms: float,
    health_gpu_ms: float,
) -> dict[str, float | int | bool]:
    ordered = sorted(samples)

    def percentile(fraction: float) -> float:
        return ordered[round(fraction * (len(ordered) - 1))]

    gpu_ms = sum(samples)
    p25 = percentile(0.25)
    p75 = percentile(0.75)
    return {
        "event_ms_p50": statistics.median(samples),
        "event_ms_p10": percentile(0.10),
        "event_ms_p25": p25,
        "event_ms_p75": p75,
        "event_ms_p90": percentile(0.90),
        "event_ms_p95": percentile(0.95),
        "event_ms_iqr": p75 - p25,
        "event_ms_min": min(samples),
        "wall_ms": wall_ms,
        "summed_cuda_event_ms": gpu_ms,
        "host_wall_to_cuda_event_ratio": wall_ms / health_gpu_ms,
        "health_valid": wall_ms / health_gpu_ms <= 1.5,
        "iterations": len(samples),
    }


def _measure_cuda_interleaved(
    functions: dict[str, Callable[[], object]],
    warmup: int,
    iterations: int,
    stabilize_ms: float,
) -> dict[str, dict[str, float | int | bool]]:
    import torch

    names = tuple(functions)

    def order(step: int) -> tuple[str, ...]:
        shuffled = list(names)
        random.Random(20260812 + step).shuffle(shuffled)
        return tuple(shuffled)

    for fn in functions.values():
        fn()
    for step in range(warmup):
        for name in order(step):
            functions[name]()
    torch.cuda.synchronize()

    if stabilize_ms:
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        elapsed = 0.0
        step = 0
        while elapsed < stabilize_ms:
            for _ in range(16):
                for name in order(step):
                    functions[name]()
                step += 1
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end)

    events = {
        name: [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(iterations)
        ]
        for name in names
    }
    wall_start = time.perf_counter()
    for step in range(iterations):
        for name in order(step):
            begin, end = events[name][step]
            begin.record()
            functions[name]()
            end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1e3
    samples = {
        name: [begin.elapsed_time(end) for begin, end in pairs] for name, pairs in events.items()
    }
    total_gpu_ms = sum(sum(values) for values in samples.values())
    return {
        name: _summarize_cuda_samples(
            values,
            wall_ms * sum(values) / total_gpu_ms,
            total_gpu_ms * sum(values) / total_gpu_ms,
        )
        for name, values in samples.items()
    }


def _lightmoe_case(
    case: GemmCase,
    mode: str,
    direction: str,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    import torch

    from lightmoe._quantization import _DEN, TensorScale, quantize_expert_stack
    from lightmoe.kernels.grouped.runtime import grouped_nvfp4_gemm
    from lightmoe.kernels.grouped.wgrad import GroupedWgrad
    from lightmoe.kernels.quantize import (
        nvfp4_quantize_colwise,
        nvfp4_quantize_rowwise,
    )

    if direction == "wgrad":
        torch.manual_seed(20260811)
        counts_cpu = routing_counts(case.token_expert_assignments, case.local_experts, case.routing)
        counts = torch.tensor(counts_cpu, dtype=torch.int32, device="cuda")
        cu = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device="cuda"),
                counts.cumsum(0, dtype=torch.int32),
            )
        )
        off_pad = (((counts + 127) // 128) * 128).cumsum(0).to(torch.int32)
        mp_total = int(off_pad[-1].item())
        dy = torch.randn(case.token_expert_assignments, case.n, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(case.token_expert_assignments, case.k, dtype=torch.bfloat16, device="cuda")
        sy, sx = TensorScale(), TensorScale()
        sy.update(dy)
        sx.update(x)
        qy = torch.empty(case.n, mp_total // 2, dtype=torch.uint8, device="cuda")
        qx = torch.empty(case.k, mp_total // 2, dtype=torch.uint8, device="cuda")
        sfy = torch.empty(case.n * mp_total // 16, dtype=torch.float8_e4m3fn, device="cuda")
        sfx = torch.empty(case.k * mp_total // 16, dtype=torch.float8_e4m3fn, device="cuda")
        nvfp4_quantize_colwise(dy, cu, sy.pair, qy, sfy, padded_offsets=off_pad)
        nvfp4_quantize_colwise(x, cu, sx.pair, qx, sfx, padded_offsets=off_pad)
        out = torch.empty(
            case.local_experts,
            case.n,
            case.k,
            dtype=torch.bfloat16,
            device="cuda",
        )
        alpha = torch.empty(case.local_experts, dtype=torch.float32, device="cuda")
        alpha.copy_((sy.pts * sx.pts).expand(case.local_experts))
        one = torch.ones_like(alpha)
        gemm = GroupedWgrad(case.n, case.k, case.local_experts)

        def prepacked():
            gemm(qy, qx, sfy, sfx, off_pad, out, alpha, one)
            return out

        def dynamic():
            nvfp4_quantize_colwise(dy, cu, sy.pair, qy, sfy, padded_offsets=off_pad)
            nvfp4_quantize_colwise(x, cu, sx.pair, qx, sfx, padded_offsets=off_pad)
            gemm(qy, qx, sfy, sfx, off_pad, out, alpha, one)
            return out

        timing = _measure_cuda(prepacked if mode == "prepacked" else dynamic, warmup, iterations)
        return {
            "status": "ok",
            "timing": timing,
            "finite": bool(torch.isfinite(out).all()),
            "actual_shape": {"e": case.local_experts, "m": case.n, "n": case.k},
        }

    if direction == "fwd":
        gemm_n, gemm_k = case.n, case.k
    else:
        gemm_n, gemm_k = case.k, case.n
    if gemm_k % 64 or gemm_n % 128:
        return {"status": "skipped", "reason": "LightMoE N/K alignment is not satisfied"}

    torch.manual_seed(20260811)
    counts_cpu = routing_counts(case.token_expert_assignments, case.local_experts, case.routing)
    counts = torch.tensor(counts_cpu, dtype=torch.int32, device="cuda")
    cu = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            counts.cumsum(0, dtype=torch.int32),
        )
    )
    a = torch.randn(case.token_expert_assignments, gemm_k, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(
        case.local_experts,
        gemm_n,
        gemm_k,
        dtype=torch.bfloat16,
        device="cuda",
    )
    pts_a = (a.abs().amax().float() / _DEN).clamp_min(1e-30).reshape(1)
    pts_b = (b.abs().amax().float() / _DEN).clamp_min(1e-30).reshape(1)
    pair_a = torch.cat((pts_a, pts_a.reciprocal()))
    qa_u8 = torch.empty(
        case.token_expert_assignments, gemm_k // 2, dtype=torch.uint8, device="cuda"
    )
    sf_rows = -(-case.token_expert_assignments // 128) + case.local_experts
    sfa = torch.zeros(
        1,
        sf_rows,
        gemm_k // 64,
        32,
        4,
        4,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    nvfp4_quantize_rowwise(a, cu, pair_a, qa_u8, sfa)
    qb, sfb = quantize_expert_stack([b[expert] for expert in range(case.local_experts)], pts_b)
    qa = qa_u8.view(torch.float4_e2m1fn_x2)
    out = torch.empty(case.token_expert_assignments, gemm_n, dtype=torch.bfloat16, device="cuda")
    alpha = (pts_a * pts_b).reshape(1)
    tile_n = 256 if gemm_n % 256 == 0 else 128
    gemm = grouped_nvfp4_gemm(
        case.local_experts,
        gemm_n,
        gemm_k,
        128,
        tile_n,
        output_dtype=torch.bfloat16,
    )

    def prepacked():
        gemm(qa, qb, out, cu, sfa, sfb, alpha)
        return out

    def dynamic():
        nvfp4_quantize_rowwise(a, cu, pair_a, qa_u8, sfa)
        gemm(qa, qb, out, cu, sfa, sfb, alpha)
        return out

    timing = _measure_cuda(prepacked if mode == "prepacked" else dynamic, warmup, iterations)
    return {
        "status": "ok",
        "timing": timing,
        "finite": bool(torch.isfinite(out).all()),
        "actual_shape": {
            "m": case.token_expert_assignments,
            "n": gemm_n,
            "k": gemm_k,
        },
    }


@dataclass
class _GroupedArm:
    call: Callable[[], object]
    result: dict[str, object]


def _grouped_accuracy(output, a_groups, weights) -> dict[str, object]:
    import torch

    actual_rows = []
    reference_rows = []
    offset = 0
    for expert, a in enumerate(a_groups):
        rows = min(2, a.shape[0])
        if rows:
            actual_rows.append(output[offset : offset + rows].float())
            reference_rows.append(a[:rows].float() @ weights[expert].float().T)
        offset += a.shape[0]
    actual = torch.cat(actual_rows)
    reference = torch.cat(reference_rows)
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), reference.flatten(), dim=0)
    rmse = (actual - reference).square().mean().sqrt()
    return {
        "finite": bool(torch.isfinite(output).all()),
        "sample_cosine": float(cosine),
        "sample_rmse": float(rmse),
        "accuracy_sample_rows": sum(rows.shape[0] for rows in actual_rows),
        "accuracy_reference": "FP32 GEMM from the same BF16 source operands",
    }


def _prepare_grouped_frontier(
    case: GemmCase,
    selected: tuple[str, ...],
    args: argparse.Namespace,
) -> tuple[dict[str, _GroupedArm], dict[str, dict[str, str]]]:
    import torch

    from lightmoe._quantization import _DEN, quantize_expert_stack
    from lightmoe.kernels.grouped.runtime import grouped_nvfp4_gemm
    from lightmoe.kernels.quantize import nvfp4_quantize_rowwise

    torch.manual_seed(20260812)
    counts_cpu = routing_counts(case.token_expert_assignments, case.local_experts, case.routing)
    counts = torch.tensor(counts_cpu, dtype=torch.int32, device="cuda")
    cu = torch.cat((counts.new_zeros(1), counts.cumsum(0, dtype=torch.int32)))
    scale = case.k**-0.5
    a_groups = [
        (torch.randn(rows, case.k, dtype=torch.bfloat16, device="cuda") * scale).contiguous()
        for rows in counts_cpu
    ]
    a = torch.cat(a_groups)
    weights = (
        torch.randn(
            case.local_experts,
            case.n,
            case.k,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * scale
    ).contiguous()
    pts_a = (a.abs().amax().float() / _DEN).clamp_min(1e-30).reshape(1)
    pts_b = (weights.abs().amax().float() / _DEN).clamp_min(1e-30).reshape(1)
    arms: dict[str, _GroupedArm] = {}
    failures: dict[str, dict[str, str]] = {}

    if "lightmoe" in selected:
        qa_u8 = torch.empty(
            case.token_expert_assignments, case.k // 2, dtype=torch.uint8, device="cuda"
        )
        sf_rows = -(-case.token_expert_assignments // 128) + case.local_experts
        sfa = torch.zeros(
            1,
            sf_rows,
            case.k // 64,
            32,
            4,
            4,
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        nvfp4_quantize_rowwise(
            a,
            cu,
            torch.cat((pts_a, pts_a.reciprocal())),
            qa_u8,
            sfa,
        )
        qb, sfb = quantize_expert_stack(
            [weights[expert] for expert in range(case.local_experts)],
            pts_b,
        )
        qa = qa_u8.view(torch.float4_e2m1fn_x2)
        out = torch.empty(
            case.token_expert_assignments, case.n, dtype=torch.bfloat16, device="cuda"
        )
        tile_ms = (args.tile_m,) if args.tile_m else (128, 256)
        tile_ns = (args.tile_n,) if args.tile_n else ((128, 256) if case.n % 256 == 0 else (128,))
        runtimes = {
            (tile_m, tile_n): grouped_nvfp4_gemm(
                case.local_experts,
                case.n,
                case.k,
                tile_m,
                tile_n,
            )
            for tile_m in tile_ms
            for tile_n in tile_ns
        }

        def lightmoe_call(runtime):
            runtime(qa, qb, out, cu, sfa, sfb, pts_a * pts_b)
            return out

        candidate_timings = _measure_cuda_interleaved(
            {
                f"{tile_m}x{tile_n}": (lambda runtime=runtime: lightmoe_call(runtime))
                for (tile_m, tile_n), runtime in runtimes.items()
            },
            1,
            10,
            min(args.stabilize_ms, 50.0),
        )
        tile_m, tile_n = min(
            runtimes,
            key=lambda value: float(candidate_timings[f"{value[0]}x{value[1]}"]["event_ms_p50"]),
        )
        runtime = runtimes[(tile_m, tile_n)]
        tile_flops = _tile_rounded_flops(
            counts_cpu,
            case.n,
            case.k,
            tile_m,
            tile_n,
        )
        arms["lightmoe"] = _GroupedArm(
            lambda: lightmoe_call(runtime),
            {
                "status": "ok",
                "config": {
                    "tile_m": tile_m,
                    "tile_n": tile_n,
                    "output_contract": "preallocated",
                },
                "work": {
                    "logical_flops": case.flops,
                    "tile_rounded_flops": tile_flops,
                    "tile_padding_overhead_pct": 100.0 * (tile_flops / case.flops - 1.0),
                },
                **_grouped_accuracy(lightmoe_call(runtime), a_groups, weights),
            },
        )

    needs_flashinfer = bool({"flashinfer_cutedsl", "torch_scaled_grouped_mm"} & set(selected))
    if needs_flashinfer:
        try:
            from flashinfer import scaled_fp4_grouped_quantize
            from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked

            max_m = max(counts_cpu)
            a_padded = torch.zeros(
                case.local_experts,
                max_m,
                case.k,
                dtype=torch.bfloat16,
                device="cuda",
            )
            for expert, expert_a in enumerate(a_groups):
                a_padded[expert, : expert_a.shape[0]].copy_(expert_a)
            global_a = pts_a.reciprocal().expand(case.local_experts).contiguous()
            global_b = pts_b.reciprocal().expand(case.local_experts).contiguous()
            qa_fi, sfa_fi = scaled_fp4_grouped_quantize(a_padded, counts, global_a)
            full_n = torch.full_like(counts, case.n)
            qb_fi, sfb_fi = scaled_fp4_grouped_quantize(weights, full_n, global_b)
            alpha = (pts_a * pts_b).expand(case.local_experts).contiguous()
            out_fi = torch.empty(
                case.local_experts,
                max_m,
                case.n,
                dtype=torch.bfloat16,
                device="cuda",
            ).permute(1, 2, 0)

            def flashinfer_call():
                grouped_gemm_nt_masked(
                    (qa_fi, sfa_fi),
                    (qb_fi, sfb_fi),
                    out_fi,
                    counts,
                    ab_dtype="float4_e2m1fn",
                    sf_dtype="float8_e4m3fn",
                    c_dtype="bfloat16",
                    sf_vec_size=16,
                    alpha=alpha,
                    alpha_dtype="float32",
                )
                return out_fi

            if "flashinfer_cutedsl" in selected:
                flashinfer_call()
                fi_concat = torch.cat(
                    [out_fi[:rows, :, expert] for expert, rows in enumerate(counts_cpu)]
                )
                arms["flashinfer_cutedsl"] = _GroupedArm(
                    flashinfer_call,
                    {
                        "status": "ok",
                        "config": {
                            "mma_tiler_mn": [128, 128],
                            "output_contract": "preallocated",
                        },
                        **_grouped_accuracy(fi_concat, a_groups, weights),
                    },
                )

            if "torch_scaled_grouped_mm" in selected:
                if any(rows % 128 for rows in counts_cpu):
                    failures["torch_scaled_grouped_mm"] = {
                        "status": "skipped",
                        "reason": "PyTorch NVFP4 grouped GEMM requires 128-row aligned groups",
                    }
                else:
                    from torch.nn.functional import ScalingType, SwizzleType, scaled_grouped_mm

                    qa_physical = qa_fi.permute(2, 0, 1)
                    qa_torch = torch.cat(
                        [qa_physical[expert, :rows] for expert, rows in enumerate(counts_cpu)]
                    ).view(torch.float4_e2m1fn_x2)
                    qb_torch = qb_fi.permute(2, 0, 1).view(torch.float4_e2m1fn_x2)
                    sfa_torch = sfa_fi.permute(5, 2, 4, 0, 1, 3).reshape(-1, case.k // 16)
                    sfb_torch = sfb_fi.permute(5, 2, 4, 0, 1, 3).flatten(1)

                    def torch_call():
                        return scaled_grouped_mm(
                            qa_torch,
                            qb_torch.transpose(-2, -1),
                            scale_a=[sfa_torch, pts_a.expand(case.local_experts)],
                            scale_recipe_a=[
                                ScalingType.BlockWise1x16,
                                ScalingType.TensorWise,
                            ],
                            scale_b=[sfb_torch, pts_b.expand(case.local_experts)],
                            scale_recipe_b=[
                                ScalingType.BlockWise1x16,
                                ScalingType.TensorWise,
                            ],
                            swizzle_a=SwizzleType.SWIZZLE_32_4_4,
                            swizzle_b=SwizzleType.SWIZZLE_32_4_4,
                            offs=cu[1:],
                            output_dtype=torch.bfloat16,
                        )

                    torch_out = torch_call()
                    arms["torch_scaled_grouped_mm"] = _GroupedArm(
                        torch_call,
                        {
                            "status": "ok",
                            "config": {
                                "input_packing": "flashinfer_nvfp4",
                                "output_contract": "allocating_api",
                            },
                            **_grouped_accuracy(torch_out, a_groups, weights),
                        },
                    )
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
            for backend in ("flashinfer_cutedsl", "torch_scaled_grouped_mm"):
                if backend in selected and backend not in arms and backend not in failures:
                    failures[backend] = {"status": "error", "reason": reason}
    return arms, failures


def _run_grouped_frontier_case(
    case: GemmCase,
    selected: tuple[str, ...],
    statuses: dict[str, BackendStatus],
    args: argparse.Namespace,
    theoretical_peak_tflops: float | None,
) -> bool:
    runnable = tuple(name for name in selected if statuses[name].runnable)
    arms, failures = _prepare_grouped_frontier(case, runnable, args)
    no_backend_errors = all(result.get("status") != "error" for result in failures.values())
    functions = {name: arm.call for name, arm in arms.items()}
    timings = (
        _measure_cuda_interleaved(
            functions,
            args.warmup,
            args.iterations,
            args.stabilize_ms,
        )
        if functions
        else {}
    )
    for backend in selected:
        base = {
            "event": "result",
            "backend": backend,
            "mode": "prepacked",
            "direction": "fwd",
            "case": _case_dict(case),
        }
        if not statuses[backend].runnable:
            result = {"status": "skipped", "reason": statuses[backend].reason}
        elif backend in failures:
            result = failures[backend]
        elif backend not in arms:
            result = {"status": "skipped", "reason": "frontier adapter is unavailable"}
        else:
            result = arms[backend].result
            timing = timings[backend]
            _annotate_throughput(
                timing,
                case.flops,
                theoretical_peak_tflops,
                gemm_only=True,
            )
            result["timing"] = timing
        print(json.dumps({**base, **result}), flush=True)

    if "lightmoe" in arms:
        repeat = _measure_cuda_interleaved(
            functions,
            args.warmup,
            args.iterations,
            args.stabilize_ms,
        )["lightmoe"]
        initial = timings["lightmoe"]
        _annotate_throughput(
            repeat,
            case.flops,
            theoretical_peak_tflops,
            gemm_only=True,
        )
        initial_to_repeat_median_deviation = (
            float(repeat["event_ms_p50"]) / float(initial["event_ms_p50"]) - 1.0
        )
        print(
            json.dumps(
                {
                    "event": "repeat",
                    "backend": "lightmoe",
                    "mode": "prepacked",
                    "direction": "fwd",
                    "case": _case_dict(case),
                    "initial_to_repeat_median_deviation": (initial_to_repeat_median_deviation),
                    "repeat_deviation_valid": abs(initial_to_repeat_median_deviation) <= 0.05,
                    "status": "ok",
                    "timing": repeat,
                    "config": arms["lightmoe"].result["config"],
                }
            ),
            flush=True,
        )
        return (
            no_backend_errors
            and bool(repeat["health_valid"])
            and abs(initial_to_repeat_median_deviation) <= 0.05
            and all(bool(timing["health_valid"]) for timing in timings.values())
        )
    return no_backend_errors and all(bool(timing["health_valid"]) for timing in timings.values())


def _te_case(
    case: GemmCase,
    mode: str,
    direction: str,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    import torch
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import NVFP4BlockScaling

    if mode != "dynamic":
        return {
            "status": "skipped",
            "reason": "Transformer Engine GroupedLinear owns dynamic quantization",
        }
    if direction != "fwd":
        return {
            "status": "skipped",
            "reason": (
                "Transformer Engine exposes input and weight gradients through one autograd "
                "backward; use moe.py with --pass fwd_bwd"
            ),
        }
    torch.manual_seed(20260811)
    counts = routing_counts(case.token_expert_assignments, case.local_experts, case.routing)
    alignment = 64
    padded_counts = tuple(((count + alignment - 1) // alignment) * alignment for count in counts)
    padded_rows = sum(padded_counts)
    x = torch.randn(padded_rows, case.k, dtype=torch.bfloat16, device="cuda")
    group_width = 64
    group_sizes = [
        min(group_width, case.local_experts - start)
        for start in range(0, case.local_experts, group_width)
    ]
    layers = [
        te.GroupedLinear(
            size,
            case.k,
            case.n,
            bias=False,
            params_dtype=torch.bfloat16,
        ).cuda()
        for size in group_sizes
    ]
    recipe = NVFP4BlockScaling()
    first_microbatch = [True]
    accepts_first_microbatch = [True] * len(layers)

    def autocast():
        current = getattr(te, "autocast", None)
        if callable(current):
            return current(enabled=True, recipe=recipe)
        return te.fp8_autocast(enabled=True, fp8_recipe=recipe)

    def call():
        with autocast():
            outputs = []
            row_start = 0
            for group, layer in enumerate(layers):
                expert_start = group * group_width
                local_counts = list(padded_counts[expert_start : expert_start + group_width])
                row_end = row_start + sum(local_counts)
                if accepts_first_microbatch[group]:
                    try:
                        output = layer(
                            x[row_start:row_end],
                            local_counts,
                            is_first_microbatch=first_microbatch[0],
                        )
                    except TypeError:
                        accepts_first_microbatch[group] = False
                        output = layer(x[row_start:row_end], local_counts)
                else:
                    output = layer(x[row_start:row_end], local_counts)
                outputs.append(output)
                row_start = row_end
            first_microbatch[0] = False
            return torch.cat(outputs, dim=0)

    timing = _measure_cuda(call, warmup, iterations)
    timing["padded_logical_flops"] = 2 * padded_rows * case.n * case.k
    return {
        "status": "ok",
        "timing": timing,
        "actual_token_expert_assignments": padded_rows,
        "row_alignment": alignment,
    }


RUNNERS = {"lightmoe": _lightmoe_case, "transformer_engine_nvfp4": _te_case}


def _dense_cases(args: argparse.Namespace) -> list[DenseCase]:
    models = parse_models(args.models)
    rows = parse_ints(args.tokens, args.suite)
    projections = parse_names(args.projections, ("gate_up", "down"), "projection")
    return [
        DenseCase(model, projection, m, *MODEL_SHAPES[model].gemm_shape(projection))
        for model in models
        for m in rows
        for projection in projections
    ]


def _dense_inputs(case: DenseCase):
    import torch

    torch.manual_seed(20260811)
    scale = case.k**-0.5
    a = (torch.randn(case.m, case.k, dtype=torch.bfloat16, device="cuda") * scale).contiguous()
    b = (torch.randn(case.n, case.k, dtype=torch.bfloat16, device="cuda") * scale).contiguous()
    one = torch.ones(1, dtype=torch.float32, device="cuda")
    qa, sfa, _ = quantize(a, one)
    qb, sfb, _ = quantize(b, one)
    return a, b, qa, qb, sfa, sfb, one


def _dense_accuracy(output, a, b) -> dict[str, object]:
    import torch

    sample_rows = min(16, a.shape[0])
    reference = a[:sample_rows].float() @ b.float().T
    actual = output[:sample_rows].float()
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), reference.flatten(), dim=0)
    rmse = (actual - reference).square().mean().sqrt()
    return {
        "finite": bool(torch.isfinite(output).all()),
        "sample_cosine": float(cosine),
        "sample_rmse": float(rmse),
        "accuracy_sample_rows": sample_rows,
        "accuracy_reference": "FP32 GEMM from the same BF16 source operands",
    }


@dataclass
class _DenseArm:
    call: Callable[[], object]
    result: dict[str, object]
    auxiliary: dict[str, Callable[[], object]]


def _prepare_dense_lightmoe(
    case: DenseCase,
    mode: str,
    args: argparse.Namespace,
    inputs,
) -> _DenseArm:
    import torch

    from lightmoe.kernels.dense.runtime import DenseNvfp4Gemm

    a, b, qa, qb, sfa, sfb, one = inputs

    tile_ms = (args.tile_m,) if args.tile_m else (128, 256)
    tile_ns = (args.tile_n,) if args.tile_n else (64, 128, 192, 256)
    trial_out = torch.empty(case.m, case.n, dtype=torch.bfloat16, device="cuda")
    runtimes = {
        (tm, tn): DenseNvfp4Gemm(case.n, case.k, tm, tn) for tm in tile_ms for tn in tile_ns
    }
    candidates = {
        f"{tm}x{tn}": (lambda runtime=runtime: runtime(qa, qb, trial_out, sfa, sfb, one))
        for (tm, tn), runtime in runtimes.items()
    }
    candidate_timings = _measure_cuda_interleaved(
        candidates,
        1,
        20,
        min(args.stabilize_ms, 100.0),
    )
    trials = [
        (float(candidate_timings[f"{tm}x{tn}"]["event_ms_p50"]), tm, tn) for tm, tn in runtimes
    ]
    _, tile_m, tile_n = min(trials)
    runtime = runtimes[(tile_m, tile_n)]
    out = torch.empty(case.m, case.n, dtype=torch.bfloat16, device="cuda")

    def prepacked():
        runtime(qa, qb, out, sfa, sfb, one)
        return out

    def dynamic():
        current_qa, current_sfa, _ = quantize(a, one)
        runtime(current_qa, qb, out, current_sfa, sfb, one)
        return out

    call = prepacked if mode == "prepacked" else dynamic
    result: dict[str, object] = {
        "status": "ok",
        "config": {
            "tile_m": tile_m,
            "tile_n": tile_n,
            "output_contract": "preallocated",
        },
        "work": {
            "logical_flops": case.flops,
            "tile_rounded_flops": _tile_rounded_flops((case.m,), case.n, case.k, tile_m, tile_n),
        },
        "autotune": [
            {"event_ms_p50": trial, "tile_m": tm, "tile_n": tn} for trial, tm, tn in trials
        ],
        **_dense_accuracy(call(), a, b),
    }
    return _DenseArm(call, result, {})


def _prepare_dense_grouped(
    case: DenseCase,
    mode: str,
    args: argparse.Namespace,
    inputs,
) -> _DenseArm:
    import torch

    from lightmoe.kernels.grouped.runtime import GroupedNvfp4Gemm

    a, b, qa, qb, sfa, sfb, one = inputs
    out = torch.empty(case.m, case.n, dtype=torch.bfloat16, device="cuda")
    cu = torch.tensor([0, case.m], dtype=torch.int32, device="cuda")
    grouped_sfa = torch.zeros(
        (1, sfa.shape[0] + 1, *sfa.shape[1:]),
        dtype=sfa.dtype,
        device=sfa.device,
    )
    grouped_sfa[0, : sfa.shape[0]].copy_(sfa)
    tile_ms = (args.tile_m,) if args.tile_m else (128, 256)
    tile_ns = (args.tile_n,) if args.tile_n else ((128, 256) if case.n % 256 == 0 else (128,))
    runtimes = {
        (tm, tn): GroupedNvfp4Gemm(1, case.n, case.k, tm, tn) for tm in tile_ms for tn in tile_ns
    }

    def call(runtime, current_qa=qa, current_sfa=grouped_sfa):
        runtime(
            current_qa,
            qb.unsqueeze(0),
            out,
            cu,
            current_sfa,
            sfb.unsqueeze(0),
            one,
        )
        return out

    trials = []
    candidates = {
        f"{tm}x{tn}": (lambda runtime=runtime: call(runtime))
        for (tm, tn), runtime in runtimes.items()
    }
    candidate_timings = _measure_cuda_interleaved(
        candidates,
        1,
        20,
        min(args.stabilize_ms, 100.0),
    )
    for tm, tn in runtimes:
        trials.append((float(candidate_timings[f"{tm}x{tn}"]["event_ms_p50"]), tm, tn))
    _, tile_m, tile_n = min(trials)
    runtime = runtimes[(tile_m, tile_n)]

    def dynamic():
        current_qa, current_sfa, _ = quantize(a, one)
        grouped_sfa[0, : current_sfa.shape[0]].copy_(current_sfa)
        return call(runtime, current_qa, grouped_sfa)

    benchmark_call = (lambda: call(runtime)) if mode == "prepacked" else dynamic
    result = {
        "status": "ok",
        "config": {
            "tile_m": tile_m,
            "tile_n": tile_n,
            "output_contract": "preallocated",
        },
        "work": {
            "logical_flops": case.flops,
            "tile_rounded_flops": _tile_rounded_flops((case.m,), case.n, case.k, tile_m, tile_n),
        },
        **_dense_accuracy(benchmark_call(), a, b),
    }
    return _DenseArm(benchmark_call, result, {})


def _prepare_dense_cublas(case: DenseCase, mode: str, inputs) -> _DenseArm | None:
    if mode != "prepacked":
        return None
    import torch
    from torch.nn.functional import ScalingType, SwizzleType, scaled_mm

    a, b, qa, qb, sfa, sfb, _ = inputs
    scale_a = sfa.reshape(-1)
    scale_b = sfb.reshape(-1)

    def call():
        return scaled_mm(
            qa,
            qb.T,
            scale_a,
            ScalingType.BlockWise1x16,
            scale_b,
            ScalingType.BlockWise1x16,
            swizzle_a=SwizzleType.SWIZZLE_32_4_4,
            swizzle_b=SwizzleType.SWIZZLE_32_4_4,
            output_dtype=torch.bfloat16,
        )

    result = {
        "status": "ok",
        "config": {"output_contract": "allocating_api"},
        **_dense_accuracy(call(), a, b),
    }
    return _DenseArm(call, result, {})


def _prepare_dense_arm(
    backend: str,
    case: DenseCase,
    mode: str,
    args: argparse.Namespace,
    inputs,
) -> _DenseArm | None:
    if backend == "lightmoe":
        return _prepare_dense_lightmoe(case, mode, args, inputs)
    if backend == "lightmoe_grouped":
        return _prepare_dense_grouped(case, mode, args, inputs)
    if backend == "cublaslt":
        return _prepare_dense_cublas(case, mode, inputs)
    raise ValueError(f"unknown dense backend: {backend}")


def _dense_main(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.warmup < 0 or args.iterations <= 0 or args.max_cases < 0 or args.stabilize_ms < 0:
        parser.error("warmup/max-cases/stabilize-ms must be non-negative and iterations positive")
    try:
        cases = _dense_cases(args)
        selected = parse_names(
            args.backends or "lightmoe,cublaslt",
            DENSE_BACKEND_NAMES,
            "backend",
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_cases:
        cases = cases[: args.max_cases]
    runtime = None if args.list else _runtime_metadata(args.theoretical_peak_tflops)
    theoretical_peak_tflops = None if runtime is None else runtime["theoretical_peak_tflops"]
    available = {
        "lightmoe": _module_exists("lightmoe"),
        "lightmoe_grouped": _module_exists("lightmoe"),
        "cublaslt": _dense_cublas_available(),
    }
    payload = {
        "benchmark": "standalone_nvfp4_dense_gemm",
        "suite": args.suite,
        "definitions": {
            "operation": "C = A @ B.T",
            "prepacked": "NVFP4 operands and scale factors are resident",
            "dynamic": "BF16 A quantization plus GEMM; B stays resident",
            "logical_flops": "2*M*N*K; one multiply-add is two FLOPs",
            "throughput": "logical FLOPs divided by CUDA-event median latency",
            "peak": "single-GPU dense FP4 specification; structured sparsity is excluded",
            "timing": "backend calls are deterministically shuffled per iteration after stabilization",
        },
        "backends": {name: {"available": available[name]} for name in selected},
        "case_count": len(cases),
        "cases": [
            asdict(case) | {"label": case.label, "logical_flops": case.flops} for case in cases
        ],
    }
    if args.list:
        print(json.dumps(payload, indent=2))
        return 0
    modes = MODES if args.mode == "both" else (args.mode,)
    print(
        json.dumps({"event": "start", **payload, "cases": None, "runtime": runtime}),
        flush=True,
    )
    started = time.time()
    run_valid = True
    for case in cases:
        for mode in modes:
            inputs = _dense_inputs(case)
            prepared = {}
            failures = {}
            for backend in selected:
                if not available[backend]:
                    failures[backend] = {
                        "status": "skipped",
                        "reason": f"{backend} is not installed",
                    }
                    continue
                try:
                    arm = _prepare_dense_arm(backend, case, mode, args, inputs)
                    if arm is None:
                        failures[backend] = {
                            "status": "skipped",
                            "reason": "the cuBLASLt row excludes quantization",
                        }
                    else:
                        prepared[backend] = arm
                except Exception as exc:  # noqa: BLE001
                    failures[backend] = {
                        "status": "error",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }

            functions = {backend: arm.call for backend, arm in prepared.items()}
            lightmoe = prepared.get("lightmoe")
            if lightmoe is not None:
                functions.update(
                    {f"lightmoe_{name}": call for name, call in lightmoe.auxiliary.items()}
                )
            timings = (
                _measure_cuda_interleaved(
                    functions,
                    args.warmup,
                    args.iterations,
                    args.stabilize_ms,
                )
                if functions
                else {}
            )
            run_valid &= all(bool(timing["health_valid"]) for timing in timings.values())

            for backend in selected:
                base = {
                    "event": "result",
                    "backend": backend,
                    "mode": mode,
                    "case": asdict(case) | {"label": case.label, "logical_flops": case.flops},
                }
                if backend in failures:
                    result = failures[backend]
                else:
                    result = prepared[backend].result
                    timing = timings[backend]
                    _annotate_throughput(
                        timing,
                        case.flops,
                        theoretical_peak_tflops,
                        gemm_only=mode == "prepacked",
                    )
                    result["timing"] = timing
                    if backend == "lightmoe":
                        for name in lightmoe.auxiliary:
                            auxiliary_timing = timings[f"lightmoe_{name}"]
                            _annotate_throughput(
                                auxiliary_timing,
                                case.flops,
                                theoretical_peak_tflops,
                                gemm_only=True,
                            )
                            result[f"{name}_timing"] = auxiliary_timing
                run_valid &= result.get("status") != "error"
                print(json.dumps({**base, **result}), flush=True)

            if lightmoe is not None:
                repeat_timings = _measure_cuda_interleaved(
                    functions,
                    args.warmup,
                    args.iterations,
                    0.0,
                )
                repeat_timing = repeat_timings["lightmoe"]
                _annotate_throughput(
                    repeat_timing,
                    case.flops,
                    theoretical_peak_tflops,
                    gemm_only=mode == "prepacked",
                )
                initial_timing = timings["lightmoe"]
                initial_to_repeat_median_deviation = (
                    float(repeat_timing["event_ms_p50"]) / float(initial_timing["event_ms_p50"])
                    - 1.0
                )
                run_valid &= bool(repeat_timing["health_valid"]) and (
                    abs(initial_to_repeat_median_deviation) <= 0.05
                )
                print(
                    json.dumps(
                        {
                            "event": "repeat",
                            "backend": "lightmoe",
                            "mode": mode,
                            "case": asdict(case)
                            | {"label": case.label, "logical_flops": case.flops},
                            "initial_to_repeat_median_deviation": (
                                initial_to_repeat_median_deviation
                            ),
                            "repeat_deviation_valid": (
                                abs(initial_to_repeat_median_deviation) <= 0.05
                            ),
                            "status": "ok",
                            "timing": repeat_timing,
                            "config": lightmoe.result["config"],
                        }
                    ),
                    flush=True,
                )
    print(
        json.dumps({"event": "done", "wall_seconds": time.time() - started, "valid": run_valid}),
        flush=True,
    )
    return 0 if run_valid else 2


def _case_dict(case: GemmCase) -> dict[str, object]:
    row = asdict(case)
    row["label"] = case.label
    row["logical_flops"] = case.flops
    counts = routing_counts(case.token_expert_assignments, case.local_experts, case.routing)
    mean = statistics.mean(counts)
    row["expert_assignment_counts"] = counts
    row["routing_statistics"] = {
        "min_assignments": min(counts),
        "max_assignments": max(counts),
        "mean_assignments": mean,
        "coefficient_of_variation": statistics.pstdev(counts) / mean if mean else 0.0,
        "zero_assignment_experts": sum(count == 0 for count in counts),
        "experts_aligned_to_128_assignments": sum(count % 128 == 0 for count in counts),
    }
    return row


def listing_payload(args: argparse.Namespace) -> dict[str, object]:
    models = parse_models(args.models)
    tokens = parse_ints(args.tokens, args.suite)
    routing_selection = (
        (QUICK_ROUTINGS if args.suite == "quick" else FULL_ROUTINGS)
        if args.routing == "all"
        else args.routing
    )
    routings = parse_names(
        routing_selection,
        FULL_ROUTINGS,
        "routing",
    )
    projections = parse_names(args.projections, ("gate_up", "down"), "projection")
    cases = generate_gemm_cases(models, tokens, args.suite, routings, projections)
    statuses = detect_backends()
    return {
        "benchmark": "standalone_nvfp4_grouped_gemm",
        "suite": args.suite,
        "definitions": {
            "prepacked": "resident NVFP4 operands and scales; grouped GEMM only",
            "dynamic": "BF16 activation quantization plus GEMM; weights stay resident/prepacked",
            "token_expert_assignments": "input tokens * top-k before capacity padding",
            "logical_flops": "2*token_expert_assignments*N*K; one multiply-add is two FLOPs",
            "tile_rounded_flops": "MMA work after per-expert M and N/K tile rounding",
            "throughput": "logical FLOPs divided by CUDA-event median latency",
            "peak": "single-GPU dense FP4 specification; structured sparsity is excluded",
            "timing": "runnable prepacked forward backends are shuffled per iteration in one session",
            "routing": {
                "balanced": "all expert weights are 1",
                "imbalanced": "expert e has weight 1 + ((17*e + 11) mod 31)",
                "single_expert_skew": (
                    "expert 0 weight is max(1, num_experts/2); every other weight is 1"
                ),
                "alignment_stress": (
                    "weights repeat 1,15,16,127,128,129,255,256,257; last is zero"
                ),
                "allocation": (
                    "floor(assignments*weight/sum(weights)), then largest remainders first"
                ),
            },
        },
        "models": {key: asdict(MODEL_SHAPES[key]) for key in models},
        "backends": {name: asdict(status) for name, status in statuses.items()},
        "case_count": len(cases),
        "cases": [_case_dict(case) for case in cases],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("grouped", "dense"), default="grouped")
    parser.add_argument("--list", action="store_true", help="print matrix and availability only")
    parser.add_argument("--suite", choices=("quick", "full"), default="quick")
    parser.add_argument("--models", default="all", help="comma-separated registry keys")
    parser.add_argument("--tokens", default=None, help="comma-separated token counts")
    parser.add_argument("--backends", default=None)
    parser.add_argument("--mode", choices=(*MODES, "both"), default="prepacked")
    parser.add_argument("--direction", choices=DIRECTIONS, default="fwd")
    parser.add_argument("--projections", default="gate_up,down")
    parser.add_argument("--routing", default="all")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--stabilize-ms", type=float, default=200.0)
    parser.add_argument(
        "--theoretical-peak-tflops",
        type=float,
        default=None,
        help="dense FP4 specification peak; B200 defaults to 9000 TFLOP/s",
    )
    parser.add_argument("--tile-m", type=int, choices=(0, 128, 256), default=0)
    parser.add_argument("--tile-n", type=int, choices=(0, 64, 128, 192, 256), default=0)
    parser.add_argument(
        "--max-cases", type=int, default=0, help="0 runs the complete selected matrix"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.theoretical_peak_tflops is not None and args.theoretical_peak_tflops <= 0:
        parser.error("theoretical-peak-tflops must be positive")
    if args.workload == "dense":
        return _dense_main(args, parser)
    try:
        payload = listing_payload(args)
        selected = parse_names(
            args.backends
            or "lightmoe,torch_scaled_grouped_mm,transformer_engine_nvfp4,cutlass,cublaslt",
            BACKEND_NAMES,
            "backend",
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.list:
        print(json.dumps(payload, indent=2))
        return 0
    if args.warmup < 0 or args.iterations <= 0 or args.max_cases < 0 or args.stabilize_ms < 0:
        parser.error("warmup/max-cases must be non-negative and iterations must be positive")

    cases = [
        GemmCase(
            **{
                key: value
                for key, value in row.items()
                if key
                not in {
                    "label",
                    "logical_flops",
                    "expert_assignment_counts",
                    "routing_statistics",
                }
            }
        )
        for row in payload["cases"]
    ]
    if args.max_cases:
        cases = cases[: args.max_cases]
    modes = MODES if args.mode == "both" else (args.mode,)
    statuses = detect_backends()
    header = {
        key: value for key, value in payload.items() if key not in {"models", "cases", "backends"}
    }
    runtime = _runtime_metadata(args.theoretical_peak_tflops)
    theoretical_peak_tflops = runtime["theoretical_peak_tflops"]
    print(
        json.dumps(
            {
                "event": "start",
                **header,
                "selected_backends": selected,
                "runtime": runtime,
            }
        ),
        flush=True,
    )
    started = time.time()
    run_valid = True
    for case in cases:
        for mode in modes:
            if mode == "prepacked" and args.direction == "fwd":
                run_valid &= _run_grouped_frontier_case(
                    case,
                    selected,
                    statuses,
                    args,
                    theoretical_peak_tflops,
                )
                continue
            for backend in selected:
                status = statuses[backend]
                base = {
                    "event": "result",
                    "backend": backend,
                    "mode": mode,
                    "direction": args.direction,
                    "case": _case_dict(case),
                }
                if not status.runnable or backend not in RUNNERS:
                    result = {"status": "skipped", "reason": status.reason}
                elif mode not in status.modes or args.direction not in status.capabilities:
                    result = {"status": "skipped", "reason": "mode or direction is unsupported"}
                else:
                    try:
                        result = RUNNERS[backend](
                            case,
                            mode,
                            args.direction,
                            args.warmup,
                            args.iterations,
                        )
                    except Exception as exc:  # noqa: BLE001
                        result = {
                            "status": "error",
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                if result.get("status") == "ok" and "timing" in result:
                    _annotate_throughput(
                        result["timing"],
                        case.flops,
                        theoretical_peak_tflops,
                        gemm_only=mode == "prepacked",
                    )
                    run_valid &= bool(result["timing"]["health_valid"])
                run_valid &= result.get("status") != "error"
                print(json.dumps({**base, **result}), flush=True)
    print(
        json.dumps({"event": "done", "wall_seconds": time.time() - started, "valid": run_valid}),
        flush=True,
    )
    return 0 if run_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
