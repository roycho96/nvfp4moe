"""Measure expert-layer and full-block latency in a fresh B200 container."""

from __future__ import annotations

import gc
import json
import statistics
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from q32_backends import make_backend
from q32_specs import get_spec
from q32_stack import QwenMoEBlock, QwenNonExpert, compile_nonexpert


def _percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * q))]


def _torch_reference(spec, trace, x, topi, topv):
    gate_up = trace["gate_up_weight"]
    down = trace["down_weight"]
    flat = topi.reshape(-1).long()
    order = torch.argsort(flat, stable=True)
    token = order // spec.topk
    probs = topv.reshape(-1)[order]
    counts = torch.bincount(flat, minlength=spec.experts).tolist()
    xg = x[token]
    pieces = []
    start = 0
    for expert, count in enumerate(counts):
        if not count:
            continue
        h = F.linear(xg[start : start + count], gate_up[expert])
        gate, up = h.chunk(2, -1)
        hh = (F.silu(gate) * up if spec.activation == "swiglu"
              else F.gelu(gate, approximate="tanh") * up)
        pieces.append(F.linear(hh, down[expert]))
        start += count
    ym = torch.cat(pieces) * probs[:, None].to(torch.bfloat16)
    y = torch.zeros_like(x)
    y.index_add_(0, token, ym)
    return y


def _accuracy(y, ref):
    yf, rf = y.float().flatten(), ref.float().flatten()
    cosine = F.cosine_similarity(yf, rf, dim=0).item()
    rel_l2 = ((yf - rf).norm() / rf.norm().clamp_min(1e-30)).item()
    max_abs = (yf - rf).abs().max().item()
    return {"cosine": cosine, "rel_l2": rel_l2, "max_abs": max_abs}


def _accuracy_vs_cpu(y, ref_cpu):
    ref = ref_cpu.to(device=y.device, non_blocking=False)
    result = _accuracy(y, ref)
    del ref
    return result


def _crop_weight_grads(spec, grads):
    """Remove alignment-only intermediate channels before checkpoint comparison."""
    gate_up = grads["gate_up"]
    down = grads["down"]
    if spec.padded_intermediate != spec.intermediate:
        gate, up = gate_up.chunk(2, dim=1)
        gate_up = torch.cat(
            (gate[:, : spec.intermediate], up[:, : spec.intermediate]), dim=1
        ).contiguous()
        down = down[:, :, : spec.intermediate].contiguous()
    return {"gate_up": gate_up, "down": down}


def _training_reference(spec, fixed, x_base, topi, topv_base, dout):
    """Build BF16 reference gradients outside the timed region."""
    x = x_base.detach().clone().requires_grad_(True)
    topv = topv_base.detach().clone().requires_grad_(True)
    gate_up = fixed["gate_up_weight"].detach().clone().requires_grad_(True)
    down = fixed["down_weight"].detach().clone().requires_grad_(True)
    ref_inputs = {"gate_up_weight": gate_up, "down_weight": down}
    y = _torch_reference(spec, ref_inputs, x, topi, topv)
    y.backward(dout)
    result = {
        "output": y.detach(),
        "input": x.grad.detach().cpu(),
        "router_weight": topv.grad.detach().cpu(),
        "gate_up_weight": gate_up.grad.detach().cpu(),
        "down_weight": down.grad.detach().cpu(),
    }
    del x, topv, gate_up, down, y
    torch.cuda.empty_cache()
    return result


def _profile_once(fn):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
    torch.cuda.synchronize()
    events = [ev for ev in prof.key_averages() if ev.device_time_total > 0]
    return {
        "gpu_kernel_sum_ms": sum(ev.device_time_total for ev in events) / 1e3,
        "kernel_count": sum(ev.count for ev in events),
        "top_kernels": [
            {"name": ev.key, "ms": ev.device_time_total / 1e3, "count": ev.count}
            for ev in sorted(events, key=lambda item: -item.device_time_total)[:8]
        ],
    }


def _measure(fn, warmup=5, iters=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    profile = _profile_once(fn)
    return {
        "event_ms_p50": statistics.median(samples),
        "event_ms_p10": _percentile(samples, 0.10),
        "event_ms_p90": _percentile(samples, 0.90),
        "event_ms_min": min(samples),
        "samples": len(samples),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        **profile,
    }


def _canary():
    torch.manual_seed(32)
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    out = torch.empty_like(a)
    for _ in range(3):
        torch.mm(a, b, out=out)
    torch.cuda.synchronize()
    values = []
    for _ in range(7):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        torch.mm(a, b, out=out)
        e.record()
        e.synchronize()
        values.append(s.elapsed_time(e))
    result = statistics.median(values)
    del a, b, out
    torch.cuda.empty_cache()
    return result


def _release():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _gradient_status(backend, x, topv):
    parameter_grads = [
        grad
        for parameter in backend.parameters()
        if (grad := parameter.grad) is not None
    ]
    return {
        "x_present": x.grad is not None,
        "x_finite": bool(x.grad is not None and torch.isfinite(x.grad).all()),
        "router_weight_present": topv.grad is not None,
        "router_weight_finite": bool(
            topv.grad is not None and torch.isfinite(topv.grad).all()
        ),
        "parameter_tensors_with_grad": len(parameter_grads),
        "parameter_grads_finite": bool(
            parameter_grads
            and all(torch.isfinite(grad).all() for grad in parameter_grads)
        ),
        "parameter_grads_nonzero": bool(
            parameter_grads
            and any(torch.count_nonzero(grad).item() > 0 for grad in parameter_grads)
        ),
        "x_norm": x.grad.float().norm().item() if x.grad is not None else None,
        "router_weight_norm": (
            topv.grad.float().norm().item() if topv.grad is not None else None
        ),
        "parameter_grad_norm": (
            sum(grad.float().square().sum() for grad in parameter_grads).sqrt().item()
            if parameter_grads
            else None
        ),
    }


def run_expert_suite(spec_key, trace_path, backends, warmup=5, iters=15):
    spec = get_spec(spec_key)
    trace_cpu = torch.load(trace_path, map_location="cpu", weights_only=False)
    expected = trace_cpu["metadata"]["geometry"]
    if expected["hidden"] != spec.hidden or expected["experts"] != spec.experts:
        raise RuntimeError("trace geometry does not match requested spec")

    fixed = {
        key: trace_cpu[key].cuda()
        for key in ("expert_input", "topk_index", "topk_weight",
                    "gate_up_weight", "down_weight")
    }
    x_base = fixed["expert_input"].contiguous()
    topi = fixed["topk_index"].contiguous()
    topv_base = fixed["topk_weight"].contiguous()
    torch.manual_seed(32_032)
    dout_base = torch.randn_like(x_base)
    training_ref = _training_reference(
        spec, fixed, x_base, topi, topv_base, dout_base
    )
    ref = training_ref["output"]
    torch.cuda.synchronize()

    rows = []
    for name in backends:
        print(f"[q32] expert {spec_key} {name}: start", flush=True)
        backend = x = topv = dout = y = fwd = train = None
        row = {"backend": name, "model": spec_key, "status": "error"}
        try:
            backend = make_backend(name, spec, fixed)
            supports_training = backend.info.training
            row["implementation"] = backend.info.name
            row["precision"] = backend.info.precision
            row["training_supported"] = supports_training
            x = x_base.detach().clone().requires_grad_(supports_training)
            topv = topv_base.detach().clone().requires_grad_(supports_training)
            dout = dout_base if supports_training else None
            step = [0]

            def fwd():
                step[0] += 1
                if supports_training:
                    return backend(x, topi, topv, step[0])
                with torch.no_grad():
                    return backend(x, topi, topv, step[0])

            def train():
                backend.zero_grad(set_to_none=True)
                x.grad = None
                topv.grad = None
                y = fwd()
                y.backward(dout)
                return y

            with torch.no_grad():
                y = backend(x.detach(), topi, topv.detach(), 0)
            row["accuracy_vs_bf16_torch"] = _accuracy(y, ref)
            row["forward"] = _measure(fwd, warmup=warmup, iters=iters)
            row["forward"]["tokens_per_second"] = (
                x.shape[0] * 1e3 / row["forward"]["event_ms_p50"]
            )
            row["forward"]["routed_tokens_per_second"] = (
                x.shape[0] * spec.topk * 1e3 / row["forward"]["event_ms_p50"]
            )
            if supports_training:
                row["train_fwd_bwd"] = _measure(
                    train, warmup=max(2, warmup // 2), iters=max(7, iters // 2)
                )
                row["train_fwd_bwd"]["tokens_per_second"] = (
                    x.shape[0] * 1e3 / row["train_fwd_bwd"]["event_ms_p50"]
                )
                row["gradient_status"] = _gradient_status(backend, x, topv)
                weight_grads = _crop_weight_grads(
                    spec, backend.training_gradients()
                )
                row["gradient_accuracy_vs_bf16_torch"] = {
                    "input": _accuracy_vs_cpu(x.grad, training_ref["input"]),
                    "router_weight": _accuracy_vs_cpu(
                        topv.grad, training_ref["router_weight"]
                    ),
                    "gate_up_weight": _accuracy_vs_cpu(
                        weight_grads["gate_up"], training_ref["gate_up_weight"]
                    ),
                    "down_weight": _accuracy_vs_cpu(
                        weight_grads["down"], training_ref["down_weight"]
                    ),
                }
            else:
                row["train_fwd_bwd"] = {
                    "status": "not_supported_by_this_backend_adapter"
                }
                row["gradient_status"] = {"status": "not_applicable"}
                row["gradient_accuracy_vs_bf16_torch"] = {
                    "status": "not_applicable"
                }
            row["status"] = "ok"
        except Exception as exc:  # availability failures are part of the matrix
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
        rows.append(row)
        print(
            f"[q32] expert {spec_key} {name}: {row['status']}"
            + (f" ({row['error']})" if row["status"] != "ok" else ""),
            flush=True,
        )
        backend = x = topv = dout = y = fwd = train = None
        _release()
    ref = training_ref = dout_base = fixed = x_base = topi = topv_base = None
    _release()
    return {"metadata": trace_cpu["metadata"], "rows": rows}


def run_qwen_stack_suite(
    trace_path,
    backends,
    attention_backends,
    compile_modes,
    warmup=3,
    iters=9,
):
    spec = get_spec("qwen3_30b_a3b")
    trace_cpu = torch.load(trace_path, map_location="cpu", weights_only=False)
    fixed = {
        key: trace_cpu[key].cuda()
        for key in ("gate_up_weight", "down_weight", "expert_input")
    }
    # Backend constructors only need these three trace fields.
    fixed["topk_index"] = trace_cpu["topk_index"].cuda()
    fixed["topk_weight"] = trace_cpu["topk_weight"].cuda()
    block_input_base = trace_cpu["block_input"].cuda().contiguous()
    validation_module = QwenNonExpert(spec, trace_cpu["layer0_nonexpert"], "sdpa")
    with torch.inference_mode():
        _, validation_x, validation_topi, validation_topv = validation_module(block_input_base)
    trace_x = fixed["expert_input"]
    trace_topi = fixed["topk_index"]
    trace_topv = fixed["topk_weight"]
    nonexpert_validation = {
        "expert_input": _accuracy(validation_x.reshape_as(trace_x), trace_x),
        "router_topk_index_match": (
            validation_topi.reshape_as(trace_topi) == trace_topi
        ).float().mean().item(),
        "router_topk_weight": _accuracy(
            validation_topv.reshape_as(trace_topv), trace_topv
        ),
    }
    validation_module = validation_x = validation_topi = validation_topv = None
    _release()
    rows = []
    for attention in attention_backends:
        for compile_mode in compile_modes:
            for backend_name in backends:
                print(
                    f"[q32] stack {attention}/{compile_mode}/{backend_name}: start",
                    flush=True,
                )
                block = expert = nonexpert = x = dout = fwd = train = None
                row = {
                    "model": spec.key,
                    "attention": attention,
                    "compile": compile_mode,
                    "backend": backend_name,
                    "status": "error",
                }
                try:
                    expert = make_backend(backend_name, spec, fixed)
                    nonexpert = QwenNonExpert(
                        spec, trace_cpu["layer0_nonexpert"], attention
                    )
                    compile_start = time.perf_counter()
                    nonexpert = compile_nonexpert(nonexpert, compile_mode)
                    block = QwenMoEBlock(nonexpert, expert)
                    x = block_input_base.detach().clone().requires_grad_(True)
                    dout = torch.randn_like(x)
                    counter = [0]

                    def fwd():
                        counter[0] += 1
                        return block(x, counter[0])

                    def train():
                        block.zero_grad(set_to_none=True)
                        x.grad = None
                        y = fwd()
                        y.backward(dout)
                        return y

                    # Keep compilation and first-use JIT out of steady state.
                    fwd()
                    torch.cuda.synchronize()
                    row["compile_and_first_call_s"] = time.perf_counter() - compile_start
                    row["forward"] = _measure(fwd, warmup=warmup, iters=iters)
                    row["train_fwd_bwd"] = _measure(
                        train, warmup=max(1, warmup // 2), iters=max(5, iters // 2)
                    )
                    for phase in ("forward", "train_fwd_bwd"):
                        row[phase]["tokens_per_second"] = (
                            x.numel() / spec.hidden * 1e3 / row[phase]["event_ms_p50"]
                        )
                    row["status"] = "ok"
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    row["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
                rows.append(row)
                print(
                    f"[q32] stack {attention}/{compile_mode}/{backend_name}: "
                    f"{row['status']}"
                    + (f" ({row['error']})" if row["status"] != "ok" else ""),
                    flush=True,
                )
                block = expert = nonexpert = x = dout = fwd = train = None
                _release()
    block_input_base = fixed = None
    _release()
    return {
        "metadata": trace_cpu["metadata"],
        "nonexpert_validation_vs_hf_capture": nonexpert_validation,
        "rows": rows,
    }


def run_all(
    trace_paths,
    exhaustive=False,
    warmup=5,
    iters=15,
    expert_backends=None,
    stack_backends=None,
    attention_backends=None,
):
    versions = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
    }
    for package, import_name in (
        ("transformer_engine", "transformer_engine"),
        ("deep_gemm", "deep_gemm"),
    ):
        try:
            module = __import__(import_name, fromlist=["*"])
            versions[package] = getattr(module, "__version__", "present")
        except Exception as exc:
            versions[package] = f"unavailable: {type(exc).__name__}: {exc}"

    canary_pre = _canary()
    if expert_backends is None:
        expert_backends = (
            "te_bf16", "te_nvfp4", "te_deepgemm_bf16", "nvfp4moe"
        )
    result = {"versions": versions, "expert": {}, "stack": {}}
    for spec_key, path in trace_paths.items():
        if spec_key.endswith("global"):
            continue
        result["expert"][spec_key] = run_expert_suite(
            spec_key, path, expert_backends, warmup=warmup, iters=iters
        )

    compile_modes = (
        ("eager", "default", "reduce-overhead", "max-autotune-no-cudagraphs")
        if exhaustive
        else ("eager", "reduce-overhead")
    )
    if stack_backends is None:
        stack_backends = ("te_bf16", "te_deepgemm_bf16", "nvfp4moe")
    if attention_backends is None:
        attention_backends = ("sdpa",)
    if "qwen3_30b_a3b" in trace_paths:
        result["stack"]["qwen3_30b_a3b"] = run_qwen_stack_suite(
            trace_paths["qwen3_30b_a3b"],
            stack_backends,
            attention_backends,
            compile_modes,
            warmup=max(2, warmup // 2),
            iters=max(7, iters // 2),
        )
    canary_post = _canary()
    drift = abs(canary_post - canary_pre) / canary_pre
    result["health"] = {
        "canary_pre_ms": canary_pre,
        "canary_post_ms": canary_post,
        "drift": drift,
        "healthy": drift <= 0.05,
    }
    return result


def save_result(result, path):
    Path(path).write_text(json.dumps(result, indent=2, ensure_ascii=False))


__all__ = ["run_all", "save_result"]
