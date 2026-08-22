"""Compare the inference-only MoE plan with FlashInfer on one B200."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass

import torch

try:
    from .model_shapes import MODEL_SHAPES
except ImportError:
    from model_shapes import MODEL_SHAPES  # type: ignore[no-redef]


@dataclass(frozen=True)
class Case:
    model: str
    tokens: int
    local_experts: int
    routing: str


def _routing(tokens: int, experts: int, topk: int, kind: str):
    generator = torch.Generator(device="cpu").manual_seed(20260819)
    positions = torch.arange(tokens * topk, dtype=torch.int64).view(tokens, topk)
    if kind == "balanced":
        ids = positions.remainder(experts)
    elif kind == "single_expert_skew":
        ids = torch.empty(tokens, topk, dtype=torch.int64)
        ids[:, 0] = 0
        ids[:, 1:] = 1 + torch.arange(tokens * (topk - 1)).view(tokens, topk - 1).remainder(
            experts - 1
        )
    elif kind == "empty":
        if topk >= experts:
            raise ValueError("empty routing requires top-k smaller than the expert count")
        ids = positions.remainder(experts - 1)
    else:
        raise ValueError("routing must be balanced, single_expert_skew, or empty")
    ids = ids.to(device="cuda", dtype=torch.int32)
    weights = torch.rand(tokens, topk, generator=generator, dtype=torch.float32)
    weights.div_(weights.sum(dim=1, keepdim=True))
    return ids, weights.cuda()


def _quantize_flashinfer(x, global_scale):
    from flashinfer.fp4_quantization import fp4_quantize

    q, sf = fp4_quantize(
        x,
        global_scale=global_scale,
        sf_vec_size=16,
        is_sf_swizzled_layout=False,
    )
    if q.dtype != torch.uint8:
        q = q.view(torch.uint8)
    return q, sf.unsqueeze(-1)


def _prepare_flashinfer_weights(w1, w2, experts, hidden, intermediate):
    from flashinfer.fused_moe.prepare import prepare_cute_dsl_nvfp4_weights

    return prepare_cute_dsl_nvfp4_weights(
        w1,
        w2,
        num_local_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
    )


def _gpu_telemetry():
    fields = ("sm_clock_mhz", "max_sm_clock_mhz", "pstate", "power_w", "power_limit_w")
    command = (
        "nvidia-smi",
        "--query-gpu=clocks.current.sm,clocks.max.sm,pstate,power.draw,power.limit",
        "--format=csv,noheader,nounits",
        "--id=0",
    )
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
        values = tuple(value.strip() for value in result.stdout.splitlines()[0].split(","))
        return dict(zip(fields, values, strict=True))
    except (FileNotFoundError, IndexError, subprocess.SubprocessError, ValueError):
        return {"status": "unavailable"}


def _prepare(arms, warmup: int, stabilize_ms: int):
    for call in arms.values():
        for _ in range(warmup):
            call()
    torch.cuda.synchronize()
    if stabilize_ms:
        stop = time.perf_counter() + stabilize_ms / 1000
        calls = tuple(arms.values())
        index = 0
        while time.perf_counter() < stop:
            calls[index % len(calls)]()
            index += 1
        telemetry = _gpu_telemetry()
        torch.cuda.synchronize()
        return telemetry
    return _gpu_telemetry()


def _samples(arms, iterations: int):
    values = {name: [] for name in arms}
    walls = {name: [] for name in arms}
    order = list(arms)
    rng = random.Random(20260819)
    for _ in range(iterations):
        rng.shuffle(order)
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start.record()
            arms[name]()
            end.record()
            end.synchronize()
            values[name].append(start.elapsed_time(end))
            walls[name].append(1000 * (time.perf_counter() - wall_start))
    return values, walls


def _summary(values, walls):
    records = {}
    for name, samples in values.items():
        ordered = sorted(samples)
        q25 = ordered[int(0.25 * (len(ordered) - 1))]
        q75 = ordered[int(0.75 * (len(ordered) - 1))]
        records[name] = {
            "p50_ms": statistics.median(samples),
            "iqr_ms": q75 - q25,
            "p10_ms": ordered[int(0.10 * (len(ordered) - 1))],
            "p90_ms": ordered[int(0.90 * (len(ordered) - 1))],
            "host_wall_to_cuda_event_ratio": sum(walls[name]) / sum(samples),
        }
    return records


@torch.no_grad()
def run_case(case: Case, warmup: int, iterations: int, stabilize_ms: int):
    from flashinfer import cute_dsl_fused_moe_nvfp4

    from lightmoe import InferenceMoE

    spec = MODEL_SHAPES[case.model]
    if spec.activation != "swiglu":
        result = {
            "event": "inference_moe",
            "case": asdict(case),
            "status": "skipped",
            "reason": "FlashInfer 0.6.17 CuTe DSL fused MoE does not support GeGLU",
        }
        print(json.dumps(result, sort_keys=True), flush=True)
        return result
    hidden = spec.hidden
    intermediate = spec.padded_intermediate
    experts = case.local_experts
    topk = spec.topk
    if topk > experts:
        raise ValueError("local experts must be at least top-k for this single-GPU comparison")

    torch.manual_seed(20260819)
    x = (torch.randn(case.tokens, hidden, device="cuda") / 10).to(torch.bfloat16)
    gate = (torch.randn(experts, intermediate, hidden, device="cuda") / 10).to(torch.bfloat16)
    up = (torch.randn_like(gate) / 10).to(torch.bfloat16)
    down = (torch.randn(experts, hidden, intermediate, device="cuda") / 10).to(torch.bfloat16)
    topk_ids, topk_weights = _routing(case.tokens, experts, topk, case.routing)

    lightmoe_plan = InferenceMoE(
        hidden,
        intermediate,
        experts,
        topk,
        case.tokens,
        activation=spec.activation,
    )
    lightmoe_plan.load_weights(gate, up, down)
    lightmoe_plan.calibrate(x, topk_ids, topk_weights)
    lightmoe_plan.warmup(x, topk_ids, topk_weights)

    # FlashInfer stores the linear branch before the gate branch.
    flashinfer_w1 = torch.cat((up, gate), dim=1)
    flashinfer_weights = _prepare_flashinfer_weights(
        flashinfer_w1, down, experts, hidden, intermediate
    )
    flashinfer_input_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    flashinfer_x, flashinfer_x_sf = _quantize_flashinfer(x, flashinfer_input_scale)
    flashinfer_out = torch.empty_like(x)

    def flashinfer_prepacked():
        return cute_dsl_fused_moe_nvfp4(
            x=flashinfer_x,
            x_sf=flashinfer_x_sf,
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights,
            num_experts=experts,
            top_k=topk,
            num_local_experts=experts,
            moe_output=flashinfer_out,
            use_fused_finalize=True,
            enable_pdl=True,
            **flashinfer_weights,
        )

    def flashinfer_dynamic():
        q, sf = _quantize_flashinfer(x, flashinfer_input_scale)
        return cute_dsl_fused_moe_nvfp4(
            x=q,
            x_sf=sf,
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights,
            num_experts=experts,
            top_k=topk,
            num_local_experts=experts,
            moe_output=flashinfer_out,
            use_fused_finalize=True,
            enable_pdl=True,
            **flashinfer_weights,
        )

    flashinfer_prepacked()
    flashinfer_dynamic()
    torch.cuda.synchronize()
    if (
        not torch.isfinite(lightmoe_plan.output[: case.tokens]).all()
        or not torch.isfinite(flashinfer_out).all()
    ):
        raise RuntimeError("non-finite output")

    arms = {
        "lightmoe_bf16_input": lambda: lightmoe_plan(x, topk_ids, topk_weights),
        "flashinfer_bf16_input": flashinfer_dynamic,
        "flashinfer_prequantized_input": flashinfer_prepacked,
    }
    telemetry = _prepare(arms, warmup, stabilize_ms)
    repeat_iterations = max(5, iterations // 2)
    initial_values, initial_walls = _samples(
        {"lightmoe_bf16_input": arms["lightmoe_bf16_input"]}, repeat_iterations
    )
    initial = _summary(initial_values, initial_walls)["lightmoe_bf16_input"]
    values, walls = _samples(arms, iterations)
    timing = _summary(values, walls)

    repeat_values, repeat_walls = _samples(
        {"lightmoe_bf16_input": arms["lightmoe_bf16_input"]}, repeat_iterations
    )
    repeat = _summary(repeat_values, repeat_walls)["lightmoe_bf16_input"]
    initial_to_repeat_median_deviation = abs(repeat["p50_ms"] / initial["p50_ms"] - 1)
    valid = initial_to_repeat_median_deviation <= 0.05 and all(
        item["host_wall_to_cuda_event_ratio"] <= 1.5 for item in timing.values()
    )
    result = {
        "event": "inference_moe",
        "case": asdict(case),
        "shape": {
            "hidden": hidden,
            "intermediate": intermediate,
            "top_k": topk,
            "token_expert_assignments": case.tokens * topk,
        },
        "timing": timing,
        "lightmoe_initial_p50_ms": initial["p50_ms"],
        "lightmoe_repeat_p50_ms": repeat["p50_ms"],
        "initial_to_repeat_median_deviation": initial_to_repeat_median_deviation,
        "valid": valid,
        "gpu": torch.cuda.get_device_name(),
        "gpu_telemetry": telemetry,
        "pytorch": torch.__version__,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--stabilize-ms", type=int, default=1000)
    return parser.parse_args()


def main():
    args = _parse_args()
    raw_cases = args.case or ["qwen3_30b_a3b:128:16:balanced"]
    for raw in raw_cases:
        model, tokens, experts, routing = raw.split(":")
        run_case(
            Case(model, int(tokens), int(experts), routing),
            args.warmup,
            args.iterations,
            args.stabilize_ms,
        )


if __name__ == "__main__":
    main()
