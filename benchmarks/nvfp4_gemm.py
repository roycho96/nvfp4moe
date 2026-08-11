"""Standalone grouped NVFP4 GEMM benchmark.

The prepacked mode measures only GEMM launch and execution. Dynamic mode adds
activation quantization while keeping resident expert weights prepacked. Every
row is either produced by the named backend or reported as skipped; this script
never substitutes another implementation for a missing backend.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

try:
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
except ImportError:
    from model_shapes import (  # type: ignore[no-redef]
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


BACKEND_NAMES = ("native", "torch_scaled_grouped_mm", "te_nvfp4", "cutlass", "cublaslt")
DIRECTIONS = ("fwd", "dgrad", "wgrad")
MODES = ("prepacked", "dynamic")


@dataclass(frozen=True)
class BackendStatus:
    name: str
    discovered: bool
    runnable: bool
    modes: tuple[str, ...]
    capabilities: tuple[str, ...]
    reason: str


def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


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

    native_callable = False
    native_reason = cuda_reason or ("" if sm100 else "native kernel requires SM100")
    if cuda_ready and sm100:
        try:
            module = importlib.import_module("nvfp4moe.kernels.gemm")
            quant = importlib.import_module("nvfp4moe.kernels.quantize")
            native_callable = callable(getattr(module, "grouped_nvfp4_gemm", None)) and callable(
                getattr(quant, "nvfp4_quantize_rowwise", None)
            )
            if not native_callable:
                native_reason = "native GEMM or quantizer callable is missing"
        except Exception as exc:  # noqa: BLE001
            native_reason = f"native import failed: {type(exc).__name__}: {exc}"

    scaled = getattr(torch.nn.functional, "scaled_grouped_mm", None)
    torch_discovered = callable(scaled)
    torch_reason = cuda_reason or arch_reason
    if torch_discovered and cuda_ready and blackwell:
        torch_reason = (
            "callable found, but no stable public NVFP4 packing adapter was detected; "
            "use the installed PyTorch benchmark for its exact scaling recipe"
        )
    elif not torch_discovered:
        torch_reason = "torch.nn.functional.scaled_grouped_mm is not present"

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
                te_reason = "TE GroupedLinear or NVFP4BlockScaling is missing"
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
        "native": BackendStatus(
            "native",
            native_callable,
            native_callable and cuda_ready and sm100,
            MODES,
            DIRECTIONS,
            native_reason or "native grouped NVFP4 kernels",
        ),
        "torch_scaled_grouped_mm": BackendStatus(
            "torch_scaled_grouped_mm",
            torch_discovered,
            False,
            ("prepacked",),
            ("fwd",),
            torch_reason,
        ),
        "te_nvfp4": BackendStatus(
            "te_nvfp4",
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
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    ordered = sorted(samples)
    return {
        "event_ms_p50": statistics.median(samples),
        "event_ms_p95": ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))],
        "event_ms_min": min(samples),
        "iterations": iterations,
    }


def _native_case(
    case: GemmCase,
    mode: str,
    direction: str,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    import torch

    from nvfp4moe.kernels.gemm import grouped_nvfp4_gemm
    from nvfp4moe.kernels.quantize import (
        nvfp4_quantize_colwise,
        nvfp4_quantize_rowwise,
    )
    from nvfp4moe.kernels.wgrad import GroupedWgrad
    from nvfp4moe.layer import _quant_expert_stack
    from nvfp4moe.recipe import _DEN, TensorScale

    if direction == "wgrad":
        torch.manual_seed(20260811)
        counts_cpu = routing_counts(case.routed_rows, case.local_experts, case.routing)
        counts = torch.tensor(counts_cpu, dtype=torch.int32, device="cuda")
        cu = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device="cuda"),
                counts.cumsum(0, dtype=torch.int32),
            )
        )
        off_pad = (((counts + 127) // 128) * 128).cumsum(0).to(torch.int32)
        mp_total = int(off_pad[-1].item())
        dy = torch.randn(case.routed_rows, case.n, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(case.routed_rows, case.k, dtype=torch.bfloat16, device="cuda")
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
        p50 = float(timing["event_ms_p50"])
        timing["effective_tflops"] = case.flops / (p50 * 1e9)
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
        return {"status": "skipped", "reason": "native N/K alignment is not satisfied"}

    torch.manual_seed(20260811)
    counts_cpu = routing_counts(case.routed_rows, case.local_experts, case.routing)
    counts = torch.tensor(counts_cpu, dtype=torch.int32, device="cuda")
    cu = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            counts.cumsum(0, dtype=torch.int32),
        )
    )
    a = torch.randn(case.routed_rows, gemm_k, dtype=torch.bfloat16, device="cuda")
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
    qa_u8 = torch.empty(case.routed_rows, gemm_k // 2, dtype=torch.uint8, device="cuda")
    sf_rows = -(-case.routed_rows // 128) + case.local_experts
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
    qb, sfb = _quant_expert_stack([b[expert] for expert in range(case.local_experts)], pts_b)
    qa = qa_u8.view(torch.float4_e2m1fn_x2)
    out = torch.empty(case.routed_rows, gemm_n, dtype=torch.bfloat16, device="cuda")
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
    p50 = float(timing["event_ms_p50"])
    timing["effective_tflops"] = 2 * case.routed_rows * gemm_n * gemm_k / (p50 * 1e9)
    return {
        "status": "ok",
        "timing": timing,
        "finite": bool(torch.isfinite(out).all()),
        "actual_shape": {"m": case.routed_rows, "n": gemm_n, "k": gemm_k},
    }


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
        return {"status": "skipped", "reason": "TE GroupedLinear owns dynamic quantization"}
    if direction != "fwd":
        return {
            "status": "skipped",
            "reason": "TE exposes dgrad and wgrad as one autograd backward; use nvfp4_moe.py fwd_bwd",
        }
    torch.manual_seed(20260811)
    counts = routing_counts(case.routed_rows, case.local_experts, case.routing)
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
    p50 = float(timing["event_ms_p50"])
    timing["effective_tflops"] = case.flops / (p50 * 1e9)
    timing["executed_tflops"] = 2 * padded_rows * case.n * case.k / (p50 * 1e9)
    return {
        "status": "ok",
        "timing": timing,
        "actual_routed_rows": padded_rows,
        "row_alignment": alignment,
    }


RUNNERS = {"native": _native_case, "te_nvfp4": _te_case}


def _case_dict(case: GemmCase) -> dict[str, object]:
    row = asdict(case)
    row["label"] = case.label
    row["flops"] = case.flops
    return row


def listing_payload(args: argparse.Namespace) -> dict[str, object]:
    models = parse_models(args.models)
    tokens = parse_ints(args.tokens, args.suite)
    routings = parse_names(
        args.routing,
        QUICK_ROUTINGS if args.suite == "quick" else FULL_ROUTINGS,
        "routing",
    )
    projections = parse_names(args.projections, ("fc1", "fc2"), "projection")
    cases = generate_gemm_cases(models, tokens, args.suite, routings, projections)
    statuses = detect_backends()
    return {
        "benchmark": "standalone_nvfp4_grouped_gemm",
        "suite": args.suite,
        "definitions": {
            "prepacked": "resident NVFP4 operands and scales; grouped GEMM only",
            "dynamic": "BF16 activation quantization plus GEMM; weights stay resident/prepacked",
            "routed_rows": "tokens * top-k before any capacity padding",
        },
        "models": {key: asdict(MODEL_SHAPES[key]) for key in models},
        "backends": {name: asdict(status) for name, status in statuses.items()},
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
        "--backends", default="native,torch_scaled_grouped_mm,te_nvfp4,cutlass,cublaslt"
    )
    parser.add_argument("--mode", choices=(*MODES, "both"), default="prepacked")
    parser.add_argument("--direction", choices=DIRECTIONS, default="fwd")
    parser.add_argument("--projections", default="fc1,fc2")
    parser.add_argument("--routing", default="all")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
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
        print(json.dumps(payload, indent=2))
        return 0
    if args.warmup < 0 or args.iterations <= 0 or args.max_cases < 0:
        parser.error("warmup/max-cases must be non-negative and iterations must be positive")

    cases = [
        GemmCase(**{key: value for key, value in row.items() if key not in {"label", "flops"}})
        for row in payload["cases"]
    ]
    if args.max_cases:
        cases = cases[: args.max_cases]
    modes = MODES if args.mode == "both" else (args.mode,)
    statuses = detect_backends()
    header = {
        key: value for key, value in payload.items() if key not in {"models", "cases", "backends"}
    }
    print(json.dumps({"event": "start", **header, "selected_backends": selected}), flush=True)
    started = time.time()
    for case in cases:
        for mode in modes:
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
                print(json.dumps({**base, **result}), flush=True)
    print(json.dumps({"event": "done", "wall_seconds": time.time() - started}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
