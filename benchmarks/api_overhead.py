"""Measure the public GEMM API against the direct runtime on one B200."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from lightmoe.gemm import DenseGemm, GroupedGemm, quantize, quantize_grouped
from lightmoe.kernels.dense.runtime import DenseNvfp4Gemm
from lightmoe.kernels.grouped.runtime import GroupedNvfp4Gemm

try:
    from .model_shapes import routing_counts
except ImportError:
    from model_shapes import routing_counts  # type: ignore[no-redef]


@dataclass(frozen=True)
class Case:
    label: str
    kind: str
    m: int
    n: int
    k: int
    tile_m: int
    tile_n: int
    experts: int = 1
    routing: str = "balanced"


CASES = (
    Case("qwen3-30b-down-dense", "dense", 8192, 2048, 768, 256, 256),
    Case("deepseek-v3-gate-up-dense", "dense", 8192, 4096, 7168, 256, 256),
    Case(
        "qwen3-30b-down-grouped-imbalanced",
        "grouped",
        65536,
        2048,
        768,
        256,
        256,
        8,
        "imbalanced",
    ),
    Case(
        "deepseek-v3-down-grouped-imbalanced",
        "grouped",
        65536,
        7168,
        2048,
        256,
        256,
        8,
        "imbalanced",
    ),
    Case(
        "deepseek-v3-down-grouped-alignment-stress",
        "grouped",
        65536,
        7168,
        2048,
        256,
        256,
        8,
        "alignment_stress",
    ),
)


def _summary(samples: list[float]) -> dict[str, float]:
    quartiles = statistics.quantiles(samples, n=4, method="inclusive")
    ordered = sorted(samples)
    return {
        "p50": statistics.median(samples),
        "iqr": quartiles[2] - quartiles[0],
        "p10": ordered[max(0, int(0.1 * (len(ordered) - 1)))],
        "p90": ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))],
    }


def _measure(
    functions: dict[str, Callable[[], object]],
    warmup: int,
    iterations: int,
    stabilize_ms: float,
    repeats: int,
) -> tuple[dict[str, dict[str, float]], float]:
    names = tuple(functions)
    for function in functions.values():
        function()
    torch.cuda.synchronize()

    for step in range(warmup):
        order = names if step % 2 else tuple(reversed(names))
        for name in order:
            functions[name]()
    torch.cuda.synchronize()

    deadline = time.perf_counter() + stabilize_ms / 1000
    step = 0
    while time.perf_counter() < deadline:
        order = names if step % 2 else tuple(reversed(names))
        for name in order:
            functions[name]()
        step += 1
    torch.cuda.synchronize()

    events = {
        name: [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(iterations)
        ]
        for name in names
    }
    cpu_us = {name: [] for name in names}
    wall_start = time.perf_counter()
    for step in range(iterations):
        order = names if step % 2 else tuple(reversed(names))
        for name in order:
            begin, end = events[name][step]
            begin.record()
            cpu_start = time.perf_counter_ns()
            for _ in range(repeats):
                functions[name]()
            cpu_us[name].append((time.perf_counter_ns() - cpu_start) / (1000 * repeats))
            end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000

    gpu_ms_raw = {
        name: [begin.elapsed_time(end) for begin, end in pairs] for name, pairs in events.items()
    }
    total_gpu_ms = sum(sum(samples) for samples in gpu_ms_raw.values())
    results = {}
    for name in names:
        gpu_ms = [sample / repeats for sample in gpu_ms_raw[name]]
        results[name] = {
            **{f"event_ms_{key}": value for key, value in _summary(gpu_ms).items()},
            **{f"submit_us_{key}": value for key, value in _summary(cpu_us[name]).items()},
        }
    return results, wall_ms / total_gpu_ms


def _repeat_measurement(
    function: Callable[[], object],
    iterations: int,
    stabilize_ms: float,
    repeats: int,
) -> float:
    deadline = time.perf_counter() + stabilize_ms / 1000
    while time.perf_counter() < deadline:
        function()
    torch.cuda.synchronize()
    events = []
    for _ in range(iterations):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(repeats):
            function()
        end.record()
        events.append((begin, end))
    torch.cuda.synchronize()
    samples = [begin.elapsed_time(end) / repeats for begin, end in events]
    return statistics.median(samples)


def _finish(
    case: Case,
    functions: dict[str, Callable[[], object]],
    output: torch.Tensor,
    warmup: int,
    iterations: int,
    stabilize_ms: float,
    repeats: int,
) -> dict[str, object]:
    timing, health_ratio = _measure(functions, warmup, iterations, stabilize_ms, repeats)
    functions["direct"]()
    torch.cuda.synchronize()
    direct_output = output.clone()
    functions["public"]()
    torch.cuda.synchronize()
    bitwise_equal = bool(torch.equal(direct_output, output))
    direct_ms = timing["direct"]["event_ms_p50"]
    public_ms = timing["public"]["event_ms_p50"]
    repeat_ms = _repeat_measurement(
        functions["direct"],
        max(10, iterations // 2),
        stabilize_ms,
        repeats,
    )
    initial_to_repeat_median_deviation = repeat_ms / direct_ms - 1.0
    event_regression = public_ms / direct_ms - 1.0
    valid = (
        health_ratio <= 1.5
        and abs(initial_to_repeat_median_deviation) <= 0.05
        and event_regression <= 0.02
    )
    return {
        "event": "api_overhead",
        "case": case.__dict__,
        "direct": timing["direct"],
        "public": timing["public"],
        "event_regression": event_regression,
        "submit_regression": (
            timing["public"]["submit_us_p50"] / timing["direct"]["submit_us_p50"] - 1.0
        ),
        "repeat_ms": repeat_ms,
        "initial_to_repeat_median_deviation": initial_to_repeat_median_deviation,
        "host_wall_to_cuda_event_ratio": health_ratio,
        "calls_per_sample": repeats,
        "bitwise_equal": bitwise_equal,
        "valid": valid and bitwise_equal,
    }


def _cosine(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            actual.float().flatten(),
            reference.float().flatten(),
            dim=0,
        )
    )


def _dense(
    case: Case,
    warmup: int,
    iterations: int,
    stabilize_ms: float,
    repeats: int,
) -> dict[str, object]:
    torch.manual_seed(20260817)
    a = torch.randn(case.m, case.k, dtype=torch.bfloat16, device="cuda") * case.k**-0.5
    b = torch.randn(case.n, case.k, dtype=torch.bfloat16, device="cuda") * case.k**-0.5
    one = torch.ones(1, dtype=torch.float32, device="cuda")
    qa, sfa, _ = quantize(a.contiguous(), one)
    qb, sfb, _ = quantize(b.contiguous(), one)
    out = torch.empty(case.m, case.n, dtype=torch.bfloat16, device="cuda")
    plan = DenseGemm(case.n, case.k, case.tile_m, case.tile_n)

    def direct_call():
        plan(qa, qb, out, sfa, sfb, one)
        return out

    def public_call():
        plan.run(qa, qb, out, sfa, sfb, one)
        return out

    result = _finish(
        case,
        {"direct": direct_call, "public": public_call},
        out,
        warmup,
        iterations,
        stabilize_ms,
        repeats,
    )
    result["sample_cosine"] = _cosine(out[:2], a[:2].float() @ b.float().T)
    result["valid"] = result["valid"] and result["sample_cosine"] >= 0.98
    return result


def _grouped(
    case: Case,
    warmup: int,
    iterations: int,
    stabilize_ms: float,
    repeats: int,
) -> dict[str, object]:
    torch.manual_seed(20260817)
    counts_cpu = routing_counts(case.m, case.experts, case.routing)
    counts = torch.tensor(counts_cpu, dtype=torch.int32, device="cuda")
    m_indptr = torch.cat((counts.new_zeros(1), counts.cumsum(0, dtype=torch.int32)))
    scale = case.k**-0.5
    a = (torch.randn(case.m, case.k, dtype=torch.bfloat16, device="cuda") * scale).contiguous()
    b = (
        torch.randn(case.experts, case.n, case.k, dtype=torch.bfloat16, device="cuda") * scale
    ).contiguous()
    qa, qb, sfa, sfb, alpha = quantize_grouped(a, b, m_indptr)
    out = torch.empty(case.m, case.n, dtype=torch.bfloat16, device="cuda")
    plan = GroupedGemm(
        case.experts,
        case.n,
        case.k,
        case.tile_m,
        case.tile_n,
    )

    def direct_call():
        plan(qa, qb, out, m_indptr, sfa, sfb, alpha)
        return out

    def public_call():
        plan.run(qa, qb, out, m_indptr, sfa, sfb, alpha)
        return out

    result = _finish(
        case,
        {"direct": direct_call, "public": public_call},
        out,
        warmup,
        iterations,
        stabilize_ms,
        repeats,
    )
    result["expert_rows"] = counts_cpu
    actual_rows = []
    reference_rows = []
    offset = 0
    for expert, rows in enumerate(counts_cpu):
        sample_rows = min(2, rows)
        if sample_rows:
            actual_rows.append(out[offset : offset + sample_rows])
            reference_rows.append(a[offset : offset + sample_rows].float() @ b[expert].float().T)
        offset += rows
    result["sample_cosine"] = _cosine(
        torch.cat(actual_rows),
        torch.cat(reference_rows),
    )
    result["valid"] = result["valid"] and result["sample_cosine"] >= 0.99
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--stabilize-ms", type=float, default=1000.0)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.iterations < 10 or args.stabilize_ms < 0 or args.repeats <= 0:
        parser.error("invalid warmup, iterations, stabilization, or repeat count")

    identity = {
        "dense_class": DenseGemm is DenseNvfp4Gemm,
        "grouped_class": GroupedGemm is GroupedNvfp4Gemm,
        "dense_run": DenseGemm.run is DenseGemm.__call__,
        "grouped_run": GroupedGemm.run is GroupedGemm.__call__,
    }
    print(json.dumps({"event": "api_identity", **identity}), flush=True)
    results = []
    for case in CASES:
        result = (
            _dense(case, args.warmup, args.iterations, args.stabilize_ms, args.repeats)
            if case.kind == "dense"
            else _grouped(case, args.warmup, args.iterations, args.stabilize_ms, args.repeats)
        )
        results.append(result)
        print(json.dumps(result), flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    valid = all(identity.values()) and all(result["valid"] for result in results)
    print(
        json.dumps(
            {
                "event": "api_overhead_summary",
                "valid": valid,
                "max_event_regression": max(result["event_regression"] for result in results),
                "max_abs_initial_to_repeat_median_deviation": max(
                    abs(result["initial_to_repeat_median_deviation"]) for result in results
                ),
                "max_host_wall_to_cuda_event_ratio": max(
                    result["host_wall_to_cuda_event_ratio"] for result in results
                ),
            }
        ),
        flush=True,
    )
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
