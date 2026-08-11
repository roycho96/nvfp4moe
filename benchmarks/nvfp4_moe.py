"""NVFP4 MoE layer benchmark with synthetic routing or captured trace replay.

Expert-core starts with expert-major rows and ends before probability weighting
or combine. Full-layer includes dispatch, expert compute, probability weighting,
and combine. Router logits and top-k selection are outside both scopes so every
backend receives the same assignments.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

try:
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
except ImportError:
    from model_shapes import (  # type: ignore[no-redef]
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


BACKEND_NAMES = (
    "native",
    "te_nvfp4",
    "deepgemm_bf16",
    "deepgemm_fp8_fp4",
    "torch_bf16",
    "torchao_mxfp8",
)
SCOPES = ("expert-core", "full-layer")
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

    native_discovered = False
    native_reason = base_reason or ("" if sm100 else "native fused kernels require SM100")
    if cuda_ready and sm100:
        try:
            package = importlib.import_module("nvfp4moe")
            native_discovered = callable(getattr(package, "MoEDispatch", None)) and callable(
                getattr(package, "MoEExpertLayer", None)
            )
            native_reason = (
                "native fused NVFP4 expert layer"
                if native_discovered
                else "MoEDispatch or MoEExpertLayer is missing"
            )
        except Exception as exc:  # noqa: BLE001
            native_reason = f"native import failed: {type(exc).__name__}: {exc}"

    grouped_mm = getattr(torch, "_grouped_mm", None)
    torch_discovered = callable(grouped_mm)
    torch_reason = base_reason or (
        "torch._grouped_mm BF16 reference"
        if torch_discovered
        else "torch._grouped_mm is not present"
    )

    te_discovered = False
    te_reason = base_reason
    if _module_exists("transformer_engine"):
        try:
            te = importlib.import_module("transformer_engine.pytorch")
            recipe = importlib.import_module("transformer_engine.common.recipe")
            te_discovered = callable(getattr(te, "GroupedLinear", None)) and callable(
                getattr(recipe, "NVFP4BlockScaling", None)
            )
            te_reason = (
                "TE NVFP4 GroupedLinear with TE permutation/combine"
                if te_discovered
                else "TE GroupedLinear or NVFP4BlockScaling is missing"
            )
        except Exception as exc:  # noqa: BLE001
            te_reason = f"Transformer Engine import failed: {type(exc).__name__}: {exc}"
    elif not te_reason:
        te_reason = "transformer_engine is not installed"

    torchao_discovered = False
    torchao_reason = "TorchAO MXFP8 grouped-expert converter is not installed"
    if _module_exists("torchao"):
        candidates = (
            "torchao.prototype.mx_formats.mx_tensor",
            "torchao.float8",
        )
        torchao_discovered = any(_module_exists(name) for name in candidates)
        if torchao_discovered:
            torchao_reason = (
                "TorchAO is present, but MXFP8 expert integration belongs to TorchTitan and has "
                "no standalone stable constructor here"
            )

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
        "native": BackendStatus(
            "native",
            native_discovered,
            native_discovered and cuda_ready and sm100,
            SCOPES,
            PASSES,
            "NVFP4 x NVFP4",
            native_reason,
        ),
        "te_nvfp4": BackendStatus(
            "te_nvfp4",
            te_discovered,
            te_discovered and cuda_ready and blackwell,
            SCOPES,
            PASSES,
            "TE NVFP4",
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
        "torch_bf16": BackendStatus(
            "torch_bf16",
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
            False,
            ("full-layer",),
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
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


def _synthetic_inputs(case: MoeCase) -> dict[str, object]:
    import torch

    torch.manual_seed(20260811)
    counts = routing_counts(case.routed_rows, case.local_experts, case.routing)
    assignments = torch.repeat_interleave(
        torch.arange(case.local_experts, dtype=torch.int64),
        torch.tensor(counts, dtype=torch.int64),
    )
    if assignments.numel() != case.routed_rows:
        raise RuntimeError("routing generator did not preserve routed rows")
    if assignments.numel() > 1:
        assignments = assignments[torch.randperm(assignments.numel())]
    topi = assignments.reshape(case.tokens, case.topk).to(torch.int32).cuda().contiguous()
    topv = torch.rand(case.tokens, case.topk, dtype=torch.float32, device="cuda")
    topv /= topv.sum(-1, keepdim=True)
    x = torch.randn(case.tokens, case.hidden, dtype=torch.bfloat16, device="cuda")
    gate_up = (
        torch.randn(
            case.local_experts,
            2 * case.intermediate,
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
            "requested_counts": list(counts),
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
        gate, up = h.chunk(2, dim=-1)
        hh = torch.nn.functional.silu(gate) * up
        return torch._grouped_mm(hh, self.w2, offs=offsets)

    def full_layer(self, x, topi, topv):
        token, offsets, probs, _ = self.prepare(topi, topv)
        yp = self.expert_core(x[token], offsets).float() * probs[:, None]
        y = self.torch.zeros_like(x, dtype=self.torch.float32)
        y.index_add_(0, token, yp)
        return y.to(x.dtype)


def _expert_layout(topi, experts):
    import torch

    flat = topi.reshape(-1).long()
    order = torch.argsort(flat, stable=True)
    token = order // topi.shape[1]
    counts = torch.bincount(flat[order], minlength=experts)
    cu = torch.cat((counts.new_zeros(1), counts.cumsum(0))).to(torch.int32)
    off_pad = (((counts + 127) // 128) * 128).cumsum(0).to(torch.int32)
    return token, counts, cu, off_pad


def _pad_expert_rows(x, counts, alignment):
    import torch

    pieces = []
    start = 0
    padded_counts = []
    for count_tensor in counts:
        count = int(count_tensor)
        padded = ((count + alignment - 1) // alignment) * alignment
        pieces.append(x.new_zeros((padded, x.shape[1])))
        if count:
            pieces[-1][:count].copy_(x[start : start + count])
        start += count
        padded_counts.append(padded)
    return torch.cat(pieces), torch.tensor(padded_counts, dtype=torch.int32, device=x.device)


def _native_backend(spec: ModelShape, inputs: dict[str, object]):
    try:
        from .moe_backends import Nvfp4MoeExpert
    except ImportError:
        from moe_backends import Nvfp4MoeExpert

    backend = Nvfp4MoeExpert(spec, inputs)
    return backend


def _te_backend(spec: ModelShape, inputs: dict[str, object]):
    try:
        from .moe_backends import TEExpert
    except ImportError:
        from moe_backends import TEExpert

    return TEExpert(spec, inputs, nvfp4=True)


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

    global_spec = MODEL_SHAPES[case.model]
    spec = replace(global_spec, experts=case.local_experts, ep_sizes=(1,), quick_ep=1)
    if scope == "expert-core":
        token, counts, cu, off_pad = _expert_layout(inputs["topk_index"], case.local_experts)
        routed = inputs["expert_input"][token].contiguous()
        gate, up = inputs["gate_up_weight"].chunk(2, dim=1)
        w1 = gate.detach().clone().requires_grad_(True)
        w3 = up.detach().clone().requires_grad_(True)
        w2 = inputs["down_weight"].detach().clone().requires_grad_(True)

        if backend_name == "native":
            from nvfp4moe import NVFP4ExpertCore

            backend = NVFP4ExpertCore(
                case.hidden,
                case.intermediate,
                case.local_experts,
                activation=case.activation,
            ).cuda()
            backend.refresh_weights(w1, w3, w2)
            backend.calibrate(routed, cu, off_pad=off_pad)

            def forward(x, _tv, step):
                backend.runtime.sr_seed = 123_000 + step
                return backend(x, w1, w3, w2, cu, off_pad=off_pad)

            def zero_grad():
                for weight in (w1, w3, w2):
                    weight.grad = None

        elif backend_name == "te_nvfp4":
            try:
                from .moe_backends import TEExpert, _activation, _te_autocast
            except ImportError:
                from moe_backends import TEExpert, _activation, _te_autocast

            backend = TEExpert(spec, inputs, nvfp4=True)
            routed, te_counts = _pad_expert_rows(routed, counts, backend.align)
            splits = tuple(int(value) for value in te_counts.cpu())

            def forward(x, _tv, step):
                del step
                with _te_autocast(backend.te, backend.recipe):
                    h = backend._grouped(backend.gl1, x, splits)
                    hh = _activation(h[:, 0::2], h[:, 1::2], backend.activation)
                    y = backend._grouped(backend.gl2, hh, splits)
                backend.first = False
                return y

            zero_grad = lambda: backend.zero_grad(set_to_none=True)

        elif backend_name == "torch_bf16":
            backend = _TorchGroupedExperts(spec, inputs)
            offsets = cu[1:]

            def forward(x, _tv, step):
                del step
                return backend.expert_core(x, offsets)

            zero_grad = backend.zero_grad
        else:
            raise ValueError(backend_name)

        benchmark_input = routed
    elif backend_name == "native":
        backend = _native_backend(spec, inputs)

        def forward(x, tv, step):
            return backend(x, inputs["topk_index"], tv, step)

        zero_grad = lambda: backend.zero_grad(set_to_none=True)
    elif backend_name == "te_nvfp4":
        backend = _te_backend(spec, inputs)

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
    elif backend_name == "torch_bf16":
        backend = _TorchGroupedExperts(spec, inputs)

        def forward(x, tv, step):
            del step
            return backend.full_layer(x, inputs["topk_index"], tv)

        zero_grad = backend.zero_grad
    else:
        raise ValueError(backend_name)

    if scope != "expert-core":
        benchmark_input = inputs["expert_input"]
    needs_grad = benchmark_pass == "fwd_bwd"
    x = benchmark_input.detach().clone().requires_grad_(needs_grad)
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
    timing["routed_tokens_per_second"] = case.routed_rows * 1000 / p50
    result = {"status": "ok", "timing": timing}
    if scope == "expert-core":
        result["actual_routed_rows"] = x.shape[0]
    if needs_grad:
        result["gradient_status"] = {
            "input": x.grad is not None and bool(torch.isfinite(x.grad).all()),
            "router_weight": (
                None
                if scope == "expert-core"
                else topv.grad is not None and bool(torch.isfinite(topv.grad).all())
            ),
        }
    return result


def _case_dict(case: MoeCase) -> dict[str, object]:
    return {**asdict(case), "label": case.label}


def listing_payload(args: argparse.Namespace) -> dict[str, object]:
    models = parse_models(args.models)
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
        "benchmark": "nvfp4_moe_layer",
        "suite": args.suite,
        "source": {
            "kind": args.source,
            "trace": str(Path(args.trace).expanduser()) if args.trace else None,
        },
        "boundaries": {
            "expert-core": "expert-major rows through FC1, activation, and FC2 only",
            "full-layer": "dispatch, expert compute, probability weighting, and combine",
            "excluded": "router logits and top-k selection",
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
        default=("native,te_nvfp4,deepgemm_bf16,deepgemm_fp8_fp4,torch_bf16,torchao_mxfp8"),
    )
    parser.add_argument("--scope", choices=(*SCOPES, "both"), default="expert-core")
    parser.add_argument("--pass", dest="benchmark_pass", choices=(*PASSES, "both"), default="fwd")
    parser.add_argument("--source", choices=("synthetic", "trace"), default="synthetic")
    parser.add_argument(
        "--trace", help="captured .pt trace; required with --source trace when running"
    )
    parser.add_argument("--routing", default="all")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
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

    cases = [
        MoeCase(**{key: value for key, value in row.items() if key != "label"})
        for row in payload["cases"]
    ]
    if args.max_cases:
        cases = cases[: args.max_cases]
    scopes = SCOPES if args.scope == "both" else (args.scope,)
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
