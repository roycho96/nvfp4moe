"""Distributed expert-parallel MoE benchmark.

The timed pipeline starts from fixed global expert assignments and includes
token packing, NCCL dispatch, local expert compute, reverse NCCL dispatch, and
weighted combine. Router logits and top-k selection are outside the boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, replace

import torch
import torch.distributed as dist

try:
    from .model_shapes import MODEL_SHAPES, ModelShape
    from .moe_backends import Nvfp4MoeExpert, TEFusedExpert
    from .nvfp4_moe import _TorchGroupedExperts
except ImportError:
    from model_shapes import MODEL_SHAPES, ModelShape
    from moe_backends import Nvfp4MoeExpert, TEFusedExpert
    from nvfp4_moe import _TorchGroupedExperts


BACKENDS = ("native", "te_nvfp4_fused", "torch_bf16")
PASSES = ("fwd", "fwd_bwd")
ROUTINGS = ("balanced", "jagged", "hotspot", "tail")


@dataclass(frozen=True)
class DistributedCase:
    model: str
    batch_per_rank: int
    sequence_length: int
    routing: str
    benchmark_pass: str

    @property
    def tokens_per_rank(self) -> int:
        return self.batch_per_rank * self.sequence_length

    @property
    def label(self) -> str:
        return (
            f"{self.model}:b{self.batch_per_rank}:s{self.sequence_length}:"
            f"{self.routing}:{self.benchmark_pass}"
        )


@dataclass
class RoutePlan:
    send_token: torch.Tensor
    send_prob: torch.Tensor
    send_local_expert: torch.Tensor
    send_counts: list[int]
    recv_counts: list[int]
    recv_local_expert: torch.Tensor
    peer_capacity: int

    @property
    def send_rows(self) -> int:
        return self.send_token.numel()

    @property
    def recv_rows(self) -> int:
        return self.recv_local_expert.numel()


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, output_splits, input_splits, peer_capacity):
        ctx.output_splits = tuple(output_splits)
        ctx.input_splits = tuple(input_splits)
        ctx.peer_capacity = peer_capacity
        return _padded_all_to_all(x, output_splits, input_splits, peer_capacity)

    @staticmethod
    def backward(ctx, grad_out):
        grad_in = _padded_all_to_all(
            grad_out,
            ctx.input_splits,
            ctx.output_splits,
            ctx.peer_capacity,
        )
        return grad_in, None, None, None


def _padded_all_to_all(x, output_splits, input_splits, peer_capacity):
    """Run a fixed-size NCCL all-to-all and remove peer padding."""
    world = dist.get_world_size()
    if len(output_splits) != world or len(input_splits) != world:
        raise ValueError("split arrays must match the process-group size")
    if max((*output_splits, *input_splits), default=0) > peer_capacity:
        raise ValueError("peer capacity is smaller than an all-to-all split")
    if sum(input_splits) != x.shape[0]:
        raise ValueError("input splits do not match the input rows")
    x = x.contiguous()
    padded_shape = (world * peer_capacity, *x.shape[1:])
    send_padded = x.new_zeros(padded_shape)
    input_offset = 0
    for peer, size in enumerate(input_splits):
        send_padded.narrow(0, peer * peer_capacity, size).copy_(x.narrow(0, input_offset, size))
        input_offset += size
    recv_padded = torch.empty_like(send_padded)
    dist.all_to_all_single(recv_padded, send_padded)
    received = [
        recv_padded.narrow(0, peer * peer_capacity, size) for peer, size in enumerate(output_splits)
    ]
    return torch.cat(received, dim=0)


def parse_case(value: str) -> DistributedCase:
    fields = value.split(":")
    if len(fields) != 5:
        raise argparse.ArgumentTypeError("case must be MODEL:BATCH_PER_RANK:SEQUENCE:ROUTING:PASS")
    model, batch, sequence, routing, benchmark_pass = fields
    if model not in MODEL_SHAPES:
        raise argparse.ArgumentTypeError(f"unknown model {model!r}")
    if routing not in ROUTINGS:
        raise argparse.ArgumentTypeError(f"routing must be one of {ROUTINGS}")
    if benchmark_pass not in PASSES:
        raise argparse.ArgumentTypeError(f"pass must be one of {PASSES}")
    try:
        batch_int, sequence_int = int(batch), int(sequence)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch and sequence must be integers") from exc
    if batch_int <= 0 or sequence_int <= 0:
        raise argparse.ArgumentTypeError("batch and sequence must be positive")
    return DistributedCase(model, batch_int, sequence_int, routing, benchmark_pass)


def _routing_weights(experts: int, routing: str) -> torch.Tensor:
    if routing == "balanced":
        values = [1.0] * experts
    elif routing == "jagged":
        values = [float(1 + ((expert * 17 + 11) % 31)) for expert in range(experts)]
    elif routing == "hotspot":
        values = [float(max(1, experts // 2)), *([1.0] * (experts - 1))]
    elif routing == "tail":
        edge = (1, 15, 16, 127, 128, 129, 255, 256, 257)
        values = [float(edge[expert % len(edge)]) for expert in range(experts)]
        if experts > 1:
            values[-1] = 0.0
    else:
        raise ValueError(routing)
    return torch.tensor(values, dtype=torch.float64)


def _global_assignments(case: DistributedCase, spec: ModelShape, rank: int):
    tokens, topk, experts = case.tokens_per_rank, spec.topk, spec.experts
    generator = torch.Generator().manual_seed(20260817 + rank * 104729)
    if case.routing == "balanced":
        token = torch.arange(tokens, dtype=torch.int64)[:, None]
        slot = torch.arange(topk, dtype=torch.int64)[None, :]
        topi = (token * topk + slot + rank * tokens * topk) % experts
    else:
        weights = _routing_weights(experts, case.routing)
        uniforms = torch.rand(tokens, experts, dtype=torch.float64, generator=generator)
        uniforms.clamp_min_(torch.finfo(torch.float64).tiny)
        scores = -uniforms.log() / weights
        topi = scores.topk(topk, dim=1, largest=False).indices
    sorted_topi = topi.sort(1).values
    if topk > 1 and bool((sorted_topi[:, 1:] == sorted_topi[:, :-1]).any()):
        raise RuntimeError("routing generated duplicate expert ids")
    topv = torch.rand(tokens, topk, dtype=torch.float32, generator=generator)
    topv /= topv.sum(1, keepdim=True)
    return topi, topv


def _build_route(case: DistributedCase, spec: ModelShape, rank: int, world: int) -> RoutePlan:
    if spec.experts % world:
        raise ValueError(f"{spec.experts} experts are not divisible by EP size {world}")
    local_experts = spec.experts // world
    topi_cpu, topv_cpu = _global_assignments(case, spec, rank)
    flat_expert = topi_cpu.reshape(-1)
    flat_token = torch.arange(case.tokens_per_rank, dtype=torch.int64).repeat_interleave(spec.topk)
    destination = torch.div(flat_expert, local_experts, rounding_mode="floor")
    order = torch.argsort(destination, stable=True)
    send_counts_t = torch.bincount(destination, minlength=world).to(torch.int64).cuda()
    recv_counts_t = torch.empty_like(send_counts_t)
    dist.all_to_all_single(recv_counts_t, send_counts_t)
    send_counts = [int(value) for value in send_counts_t.cpu().tolist()]
    recv_counts = [int(value) for value in recv_counts_t.cpu().tolist()]
    peer_capacity_t = torch.tensor(max(send_counts), dtype=torch.int64, device="cuda")
    dist.all_reduce(peer_capacity_t, op=dist.ReduceOp.MAX)
    peer_capacity = int(peer_capacity_t)
    send_local_expert = (flat_expert[order] % local_experts).to(torch.int32).cuda()
    recv_local_expert = _padded_all_to_all(
        send_local_expert,
        recv_counts,
        send_counts,
        peer_capacity,
    )
    return RoutePlan(
        send_token=flat_token[order].cuda(),
        send_prob=topv_cpu.reshape(-1)[order].cuda(),
        send_local_expert=send_local_expert,
        send_counts=send_counts,
        recv_counts=recv_counts,
        recv_local_expert=recv_local_expert,
        peer_capacity=peer_capacity,
    )


def _make_trace(spec: ModelShape, rows: int, rank: int):
    torch.manual_seed(20260817 + rank)
    return {
        "expert_input": torch.empty(rows, spec.hidden, dtype=torch.bfloat16, device="cuda"),
        "gate_up_weight": (
            torch.randn(
                spec.experts,
                2 * spec.padded_intermediate,
                spec.hidden,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * spec.hidden**-0.5
        ),
        "down_weight": (
            torch.randn(
                spec.experts,
                spec.hidden,
                spec.padded_intermediate,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * spec.padded_intermediate**-0.5
        ),
    }


class _DistributedArm:
    def __init__(self, name, backend, x, route, benchmark_pass):
        self.name = name
        self.backend = backend
        self.x = x.detach().clone().requires_grad_(benchmark_pass == "fwd_bwd")
        self.prob = route.send_prob.detach().clone().requires_grad_(benchmark_pass == "fwd_bwd")
        self.route = route
        self.benchmark_pass = benchmark_pass
        self.step = 0
        self.output = None

    def zero_grad(self):
        self.x.grad = None
        self.prob.grad = None
        if hasattr(self.backend, "zero_grad"):
            try:
                self.backend.zero_grad(set_to_none=True)
            except TypeError:
                self.backend.zero_grad()

    def _local(self, recv_x, recv_expert):
        self.step += 1
        topi = recv_expert[:, None]
        topv = torch.ones(topi.shape, dtype=torch.float32, device=topi.device)
        if self.name == "torch_bf16":
            return self.backend.full_layer(recv_x, topi, topv)
        return self.backend(recv_x, topi, topv, self.step)

    def forward(self):
        route = self.route
        send_x = self.x.index_select(0, route.send_token)
        recv_x = _AllToAll.apply(
            send_x,
            route.recv_counts,
            route.send_counts,
            route.peer_capacity,
        )
        recv_expert = _padded_all_to_all(
            route.send_local_expert,
            route.recv_counts,
            route.send_counts,
            route.peer_capacity,
        )
        local_y = self._local(recv_x, recv_expert)
        returned = _AllToAll.apply(
            local_y,
            route.send_counts,
            route.recv_counts,
            route.peer_capacity,
        )
        weighted = returned.float() * self.prob[:, None]
        out = torch.zeros(
            self.x.shape,
            dtype=torch.float32,
            device=self.x.device,
        )
        out.index_add_(0, route.send_token, weighted)
        return out.to(torch.bfloat16)

    def call(self, dout):
        self.zero_grad()
        out = self.forward()
        if self.benchmark_pass == "fwd_bwd":
            out.backward(dout)
        self.output = out.detach()
        return out

    def gradients(self):
        if self.benchmark_pass != "fwd_bwd":
            return None
        gradients = {
            "input": self.x.grad,
            "router_prob": self.prob.grad,
        }
        if self.name == "torch_bf16":
            gradients.update(
                {
                    "gate_up_weight": self.backend.w1.grad.transpose(1, 2),
                    "down_weight": self.backend.w2.grad.transpose(1, 2),
                }
            )
        else:
            weight_gradients = self.backend.training_gradients()
            gradients.update(
                {
                    "gate_up_weight": weight_gradients["gate_up"],
                    "down_weight": weight_gradients["down"],
                }
            )
        if any(value is None for value in gradients.values()):
            raise RuntimeError(f"{self.name} did not produce every training gradient")
        return gradients


def _make_backend(name, spec, trace):
    if name == "native":
        return Nvfp4MoeExpert(spec, trace)
    if name == "te_nvfp4_fused":
        return TEFusedExpert(spec, trace)
    if name == "torch_bf16":
        return _TorchGroupedExperts(spec, trace)
    raise ValueError(name)


def _global_max(value: float) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device="cuda")
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor)


def _sample(callable_):
    dist.barrier()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    callable_()
    end.record()
    end.synchronize()
    wall_ms = 1000 * (time.perf_counter() - wall_start)
    event_ms = start.elapsed_time(end)
    return _global_max(event_ms), _global_max(wall_ms)


def _summary(events, walls):
    ordered = sorted(events)
    q25 = ordered[int(0.25 * (len(ordered) - 1))]
    q75 = ordered[int(0.75 * (len(ordered) - 1))]
    ratio = sum(walls) / sum(events)
    return {
        "event_ms_p50": statistics.median(events),
        "event_ms_p10": ordered[int(0.10 * (len(ordered) - 1))],
        "event_ms_p90": ordered[int(0.90 * (len(ordered) - 1))],
        "event_ms_iqr": q75 - q25,
        "wall_ms_p50": statistics.median(walls),
        "wall_to_event_ratio": ratio,
        "health_valid": ratio <= 1.5,
        "iterations": len(events),
    }


def _error(actual, reference):
    a = actual.float().reshape(-1)
    b = reference.float().reshape(-1)
    stride = max(1, (a.numel() + 65535) // 65536)
    a, b = a[::stride], b[::stride]
    diff = a - b
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    cosine = torch.dot(a, b) / denom if float(denom) else a.new_tensor(1.0)
    finite = torch.isfinite(a).all().to(torch.int32)
    relative_l2 = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(b).clamp_min(1e-12)
    max_abs = diff.abs().max()
    sample_count = torch.tensor(a.numel(), dtype=torch.int64, device=a.device)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    dist.all_reduce(cosine, op=dist.ReduceOp.MIN)
    dist.all_reduce(relative_l2, op=dist.ReduceOp.MAX)
    dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)
    dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
    return {
        "sample_count_all_ranks": int(sample_count),
        "finite_all_ranks": bool(finite),
        "minimum_rank_cosine": float(cosine),
        "maximum_rank_relative_l2": float(relative_l2),
        "maximum_absolute_error": float(max_abs),
    }


def _gradient_error(actual, reference):
    if actual.keys() != reference.keys():
        raise ValueError("gradient sets do not match")
    return {name: _error(actual[name], reference[name]) for name in actual}


def _transport(route, x):
    send = x.index_select(0, route.send_token)
    recv = _padded_all_to_all(
        send,
        route.recv_counts,
        route.send_counts,
        route.peer_capacity,
    )
    _padded_all_to_all(
        route.send_local_expert,
        route.recv_counts,
        route.send_counts,
        route.peer_capacity,
    )
    returned = _padded_all_to_all(
        recv,
        route.send_counts,
        route.recv_counts,
        route.peer_capacity,
    )
    return returned


def _run_case(case, backend_names, warmup, iterations, stabilize_ms, rank, world):
    global_spec = MODEL_SHAPES[case.model]
    route = _build_route(case, global_spec, rank, world)
    recv_rows = torch.tensor(route.recv_rows, dtype=torch.int64, device="cuda")
    recv_grid = [torch.empty_like(recv_rows) for _ in range(world)]
    dist.all_gather(recv_grid, recv_rows)
    recv_rows_all = [int(value) for value in recv_grid]
    if rank == 0:
        print(
            json.dumps(
                {
                    "event": "distributed_ep_case_start",
                    "case": case.label,
                    "backends": backend_names,
                    "recv_rows_by_rank": recv_rows_all,
                    "peer_capacity": route.peer_capacity,
                }
            ),
            flush=True,
        )
    local_experts = global_spec.experts // world
    local_spec = replace(global_spec, experts=local_experts, topk=1, ep_sizes=(1,), quick_ep=1)
    trace = _make_trace(local_spec, route.recv_rows, rank)
    torch.manual_seed(20260818 + rank)
    x = torch.randn(
        case.tokens_per_rank,
        global_spec.hidden,
        dtype=torch.bfloat16,
        device="cuda",
    )
    dout = torch.randn_like(x) if case.benchmark_pass == "fwd_bwd" else None
    arms = {}
    errors = {}
    for name in backend_names:
        try:
            backend = _make_backend(name, local_spec, trace)
            arms[name] = _DistributedArm(name, backend, x, route, case.benchmark_pass)
        except Exception as exc:  # noqa: BLE001
            errors[name] = f"{type(exc).__name__}: {exc}"
    del trace
    if not arms:
        raise RuntimeError(f"no runnable backends: {errors}")

    for name, arm in list(arms.items()):
        try:
            arm.call(dout)
            for _ in range(warmup):
                arm.call(dout)
        except Exception as exc:  # noqa: BLE001
            errors[name] = f"warmup failed: {type(exc).__name__}: {exc}"
            del arms[name]
    if not arms:
        raise RuntimeError(f"all backends failed: {errors}")
    dist.barrier()
    torch.cuda.synchronize()
    time.sleep(stabilize_ms / 1000)

    transport_events, transport_walls = [], []
    for _ in range(iterations):
        event, wall = _sample(lambda: _transport(route, x))
        transport_events.append(event)
        transport_walls.append(wall)

    event_samples = {name: [] for name in arms}
    wall_samples = {name: [] for name in arms}
    names = list(arms)
    rng = random.Random(20260817)
    for _ in range(iterations):
        order = names.copy()
        rng.shuffle(order)
        for name in order:
            event, wall = _sample(lambda arm=arms[name]: arm.call(dout))
            event_samples[name].append(event)
            wall_samples[name].append(wall)

    first = names[0]
    canary_events, canary_walls = [], []
    canary_rng = random.Random(20260817)
    for _ in range(iterations):
        order = names.copy()
        canary_rng.shuffle(order)
        for name in order:
            event, wall = _sample(lambda arm=arms[name]: arm.call(dout))
            if name == first:
                canary_events.append(event)
                canary_walls.append(wall)
    canary = _summary(canary_events, canary_walls)
    initial_p50 = statistics.median(event_samples[first])
    canary_drift = abs(canary["event_ms_p50"] / initial_p50 - 1.0)

    validation_outputs = {}
    validation_gradients = {}
    for name, arm in arms.items():
        arm.call(dout)
        validation_outputs[name] = arm.output.clone()
        if case.benchmark_pass == "fwd_bwd":
            validation_gradients[name] = arm.gradients()
    reference = validation_outputs.get("torch_bf16")
    accuracy = {}
    if reference is not None:
        for name in arms:
            if name != "torch_bf16":
                accuracy[name] = _error(validation_outputs[name], reference)
    pairwise_accuracy = None
    if "native" in validation_outputs and "te_nvfp4_fused" in validation_outputs:
        pairwise_accuracy = _error(
            validation_outputs["native"], validation_outputs["te_nvfp4_fused"]
        )
    gradient_accuracy = {}
    gradient_reference = validation_gradients.get("torch_bf16")
    if gradient_reference is not None:
        for name, gradients in validation_gradients.items():
            if name != "torch_bf16":
                gradient_accuracy[name] = _gradient_error(gradients, gradient_reference)

    transport = _summary(transport_events, transport_walls)
    results = {}
    for name in arms:
        timing = _summary(event_samples[name], wall_samples[name])
        timing["tokens_per_second_per_rank"] = case.tokens_per_rank * 1000 / timing["event_ms_p50"]
        timing["global_tokens_per_second"] = (
            case.tokens_per_rank * world * 1000 / timing["event_ms_p50"]
        )
        results[name] = {
            "timing": timing,
            "accuracy_vs_torch_bf16": accuracy.get(name),
            "gradient_accuracy_vs_torch_bf16": gradient_accuracy.get(name),
        }
    valid = (
        canary_drift <= 0.05
        and transport["health_valid"]
        and all(result["timing"]["health_valid"] for result in results.values())
    )
    useful_transport_bytes = [
        (route.send_rows + recv_rows) * 2 * global_spec.hidden + route.send_rows * 4
        for recv_rows in recv_rows_all
    ]
    padded_transport_bytes = route.peer_capacity * world * (4 * global_spec.hidden + 4)
    payload = {
        "event": "distributed_ep_result",
        "case": {**asdict(case), "label": case.label},
        "world_size": world,
        "ep_size": world,
        "local_experts": local_experts,
        "tokens_per_rank": case.tokens_per_rank,
        "global_input_tokens": case.tokens_per_rank * world,
        "routed_rows_per_source_rank": route.send_rows,
        "recv_rows_by_rank": recv_rows_all,
        "forward_transport": {
            "collective": "capacity-padded NCCL all_to_all_single",
            "peer_capacity_rows": route.peer_capacity,
            "useful_bytes_by_rank": useful_transport_bytes,
            "padded_bytes_per_rank": padded_transport_bytes,
            "padding_overhead_vs_global_mean": (
                padded_transport_bytes / statistics.mean(useful_transport_bytes) - 1.0
            ),
            "timing": transport,
        },
        "backends": results,
        "native_vs_te_accuracy": pairwise_accuracy,
        "backend_errors": errors,
        "canary": {
            "backend": first,
            "event_ms_p50": canary["event_ms_p50"],
            "drift": canary_drift,
            "drift_valid": canary_drift <= 0.05,
        },
        "valid": valid,
    }
    if rank == 0:
        print(json.dumps(payload), flush=True)
    return payload


def _environment(rank, world):
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "event": "distributed_ep_environment",
        "rank": rank,
        "world_size": world,
        "hostname": os.uname().nodename,
        "gpu": properties.name,
        "sm_count": properties.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "nccl_net": os.environ.get("NCCL_NET"),
        "nccl_net_plugin": os.environ.get("NCCL_NET_PLUGIN"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--backends", default="native,te_nvfp4_fused,torch_bf16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--stabilize-ms", type=float, default=1000)
    args = parser.parse_args(argv)
    backend_names = tuple(name.strip() for name in args.backends.split(",") if name.strip())
    unknown = sorted(set(backend_names) - set(BACKENDS))
    if unknown:
        parser.error(f"unknown backends {unknown}")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if rank == 0:
        print(json.dumps(_environment(rank, world)), flush=True)
        print(
            json.dumps(
                {
                    "event": "distributed_ep_contract",
                    "timed": (
                        "cached route gather, NCCL dispatch, local expert layer, reverse NCCL "
                        "dispatch, probability weighting, combine"
                    ),
                    "excluded": "router logits, top-k selection, optimizer, weight refresh",
                    "latency": "maximum rank CUDA-event time",
                    "batch": "batch_per_rank * sequence_length; global tokens multiply by EP size",
                    "backends": backend_names,
                }
            ),
            flush=True,
        )
    all_valid = True
    try:
        for case in args.case:
            result = _run_case(
                case,
                backend_names,
                args.warmup,
                args.iterations,
                args.stabilize_ms,
                rank,
                world,
            )
            all_valid &= result["valid"]
    finally:
        dist.destroy_process_group()
    return 0 if all_valid else 2


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
