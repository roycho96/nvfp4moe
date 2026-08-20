"""Inference-only MoE execution checks."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="SM100 required",
)


def _reference(x, topk_ids, topk_weights, gate, up, down):
    result = torch.zeros_like(x, dtype=torch.float32)
    for token in range(x.shape[0]):
        source = x[token].float()
        for route in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, route])
            g = source @ gate[expert].float().T
            u = source @ up[expert].float().T
            hidden = torch.nn.functional.silu(g) * u
            result[token] += topk_weights[token, route] * (hidden @ down[expert].float().T)
    return result


def test_inference_plan_is_repeatable_and_allocation_free():
    from nvfp4moe import InferenceMoE

    torch.manual_seed(20260819)
    tokens, hidden, intermediate, experts, topk = 17, 256, 128, 4, 2
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    gate = (torch.randn(experts, intermediate, hidden, device="cuda") * hidden**-0.5).to(
        torch.bfloat16
    )
    up = (torch.randn_like(gate) * hidden**-0.5).to(torch.bfloat16)
    down = (torch.randn(experts, hidden, intermediate, device="cuda") * intermediate**-0.5).to(
        torch.bfloat16
    )
    topk_ids = torch.tensor([[0, 1], [1, 0]] * 8 + [[0, 1]], dtype=torch.int32, device="cuda")
    topk_weights = torch.rand(tokens, topk, dtype=torch.float32, device="cuda")
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)

    plan = InferenceMoE(hidden, intermediate, experts, topk, tokens)
    assert plan._qx_u8.data_ptr() == plan._routed_out.data_ptr()
    assert plan._qh_u8.data_ptr() == plan._out.data_ptr()
    plan.load_weights(gate, up, down)
    plan.calibrate(x, topk_ids, topk_weights)
    plan.warmup(x, topk_ids, topk_weights)

    first = plan(x, topk_ids, topk_weights).clone()
    prefill = plan.run_prefill(x, topk_ids, topk_weights).clone()
    plan._sfh.fill_(0)
    second = plan(x, topk_ids, topk_weights).clone()
    torch.cuda.synchronize()
    assert torch.equal(first, second)
    assert torch.equal(first, prefill)

    before = torch.cuda.memory_allocated()
    ptr = plan(x, topk_ids, topk_weights).data_ptr()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    assert ptr == plan.output.data_ptr()

    x_original = x.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = plan(x, topk_ids, topk_weights)
    graph.replay()
    graph_first = captured.clone()
    x.mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(graph_first, captured)
    x.copy_(x_original)

    reference = _reference(x, topk_ids, topk_weights, gate, up, down)
    cosine = torch.nn.functional.cosine_similarity(
        first.float().flatten(), reference.flatten(), dim=0
    )
    assert cosine > 0.95


def test_routed_and_full_paths_match():
    from nvfp4moe import InferenceMoE
    from nvfp4moe.kernels.finalize import moe_finalize

    torch.manual_seed(20260820)
    tokens, hidden, intermediate, experts, topk = 8, 256, 128, 4, 2
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(experts, intermediate, hidden, dtype=torch.bfloat16, device="cuda")
    up = torch.randn_like(gate)
    down = torch.randn(experts, hidden, intermediate, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2]] * 2, device="cuda").int()
    topk_weights = torch.full((tokens, topk), 0.5, device="cuda")

    plan = InferenceMoE(hidden, intermediate, experts, topk, tokens)
    plan.load_weights(gate, up, down)
    plan.calibrate(x, topk_ids, topk_weights)
    full = plan(x, topk_ids, topk_weights).clone()

    rows = tokens * topk
    routed_x = x.index_select(0, plan._gather[:rows].long())
    routed_y = plan.run_routed(routed_x, plan._m_indptr, plan._padded_offsets).clone()
    combined = torch.empty_like(full)
    moe_finalize(
        routed_y,
        plan._slots[:rows],
        combined,
        topk,
        tile_t=2,
        n_frag=2,
        weights=plan._probs[:rows],
    )
    torch.cuda.synchronize()
    assert torch.equal(full, combined)


def test_batch_one_decode_matches_prefill():
    from nvfp4moe import InferenceMoE

    torch.manual_seed(20260822)
    hidden, intermediate, experts, topk = 256, 128, 4, 2
    x = torch.randn(1, hidden, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(experts, intermediate, hidden, dtype=torch.bfloat16, device="cuda")
    up = torch.randn_like(gate)
    down = torch.randn(experts, hidden, intermediate, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.tensor([[1, 3]], dtype=torch.int32, device="cuda")
    topk_weights = torch.tensor([[0.625, 0.375]], dtype=torch.float32, device="cuda")

    plan = InferenceMoE(hidden, intermediate, experts, topk, 1)
    plan.load_weights(gate, up, down)
    plan.calibrate(x, topk_ids, topk_weights)
    decode = plan.run_decode(x, topk_ids, topk_weights).clone()
    prefill = plan.run_prefill(x, topk_ids, topk_weights).clone()
    torch.cuda.synchronize()

    assert torch.equal(decode, prefill)
    assert torch.equal(plan._gather[:topk], torch.tensor([0, 0], device="cuda").int())
    assert torch.equal(plan._m_indptr, torch.tensor([0, 0, 1, 1, 2], device="cuda").int())


@pytest.mark.parametrize("routing", ("balanced", "hotspot"))
def test_fast_decode_plan_matches_prefill(routing):
    from nvfp4moe import InferenceMoE

    torch.manual_seed(20260823)
    tokens, hidden, intermediate, experts, topk = 8, 2048, 128, 16, 2
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(experts, intermediate, hidden, dtype=torch.bfloat16, device="cuda")
    up = torch.randn_like(gate)
    down = torch.randn(experts, hidden, intermediate, dtype=torch.bfloat16, device="cuda")
    token = torch.arange(tokens, dtype=torch.int32, device="cuda")[:, None]
    route = torch.arange(topk, dtype=torch.int32, device="cuda")[None, :]
    topk_ids = ((token * topk + route) % experts).contiguous()
    if routing == "hotspot":
        topk_ids = route.expand(tokens, -1).contiguous()
    topk_weights = torch.rand(tokens, topk, dtype=torch.float32, device="cuda")
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)

    plan = InferenceMoE(hidden, intermediate, experts, topk, tokens)
    plan.load_weights(gate, up, down)
    plan.set_activation_scales(1.0, 1.0)
    decode = plan.run_decode(x, topk_ids, topk_weights).clone()
    prefill = plan.run_prefill(x, topk_ids, topk_weights).clone()
    torch.cuda.synchronize()

    assert plan._full_execution(tokens, decode=True).fc1.fast_decode_sched
    assert torch.equal(decode, prefill)


def test_decode_uses_static_scheduler_by_default():
    from nvfp4moe import InferenceMoE

    default = InferenceMoE(4096, 128, 4, 2, 1)
    decode = default._new_gemm("fc1", 2, torch.float4_e2m1fn_x2, decode=True)
    prefill = default._new_gemm("fc1", 2, torch.float4_e2m1fn_x2)
    explicit = InferenceMoE(4096, 128, 4, 2, 1, use_dynamic_sched=True)
    forced = explicit._new_gemm("fc1", 2, torch.float4_e2m1fn_x2, decode=True)

    assert not decode.use_dynamic_sched
    assert prefill.use_dynamic_sched
    assert forced.use_dynamic_sched


def test_swapped_fc2_is_limited_to_short_decode():
    from nvfp4moe.inference import _use_swapped_fc2

    assert _use_swapped_fc2(8)
    assert not _use_swapped_fc2(9)


def test_decode_tile_rows_selects_wide_kimi_shapes():
    from nvfp4moe.inference import _decode_tile_rows

    assert _decode_tile_rows(16, 7168, 2048, 48) == 16
    assert _decode_tile_rows(64, 7168, 2048, 48) == 32
    assert _decode_tile_rows(8, 7168, 2048, 48) == 8
    assert _decode_tile_rows(32, 7168, 2048, 32) == 8
    assert _decode_tile_rows(32, 4096, 1536, 128) == 8


@pytest.mark.parametrize("experts", (48, 128, 256))
def test_wide_dispatch_matches_stable_sort(experts):
    from nvfp4moe import MoEDispatch

    torch.manual_seed(20260821)
    tokens, topk = 65, 8
    topk_ids = torch.randint(experts, (tokens, topk), dtype=torch.int32, device="cuda")
    topk_weights = torch.rand(tokens, topk, dtype=torch.float32, device="cuda")
    dispatch = MoEDispatch(tokens, experts, topk)
    gather, m_indptr, probs, slots = dispatch(topk_ids, topk_weights)

    flat = topk_ids.flatten().long()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=experts)
    reference_indptr = torch.zeros(experts + 1, dtype=torch.int32, device="cuda")
    reference_indptr[1:] = counts.cumsum(0).int()
    reference_slots = torch.empty_like(slots)
    reference_slots[order] = torch.arange(flat.numel(), dtype=torch.int32, device="cuda")

    assert torch.equal(gather, (order // topk).int())
    assert torch.equal(m_indptr, reference_indptr)
    assert torch.equal(probs, topk_weights.flatten()[order])
    assert torch.equal(slots, reference_slots)


def test_sparse_large_dispatch_quantize():
    from nvfp4moe.kernels.dispatch import B_MAX, moe_dispatch
    from nvfp4moe.kernels.quantize import nvfp4_quantize_rowwise

    tokens, hidden, experts, topk = 1024, 2048, 128, 8
    rows = tokens * topk
    active_sets = (
        (40, 45, 65, 73, 98, 99, 100, 114),
        (10, 64, 70, 94, 108, 111, 120, 125),
        (30, 35, 58, 92, 93, 94, 99, 110),
        (4, 14, 16, 53, 82, 103, 112, 124),
        (17, 19, 21, 24, 71, 78, 81, 126),
        (20, 26, 69, 76, 91, 98, 103, 104),
    )
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    topk_weights = torch.full((tokens, topk), 1 / topk, device="cuda")
    sf_rows = -(-rows // 128) + experts
    scale = torch.ones(2, dtype=torch.float32, device="cuda")
    workspaces = []
    for _ in active_sets:
        workspaces.append(
            (
                torch.empty(rows, dtype=torch.int32, device="cuda"),
                torch.empty(experts + 1, dtype=torch.int32, device="cuda"),
                torch.empty(rows, dtype=torch.float32, device="cuda"),
                torch.empty(rows, dtype=torch.int32, device="cuda"),
                torch.empty(B_MAX * experts, dtype=torch.int32, device="cuda"),
                torch.empty(experts, dtype=torch.int32, device="cuda"),
                torch.empty(rows, hidden // 2, dtype=torch.uint8, device="cuda"),
                torch.empty(sf_rows * (hidden // 64) * 512, dtype=torch.uint8, device="cuda"),
            )
        )

    for active, workspace in zip(active_sets, workspaces, strict=True):
        gather, cu, probs, slots, parts, offsets, q, sf = workspace
        topk_ids = torch.tensor(active, dtype=torch.int32, device="cuda").repeat(tokens, 1)
        moe_dispatch(
            topk_ids,
            topk_weights,
            experts,
            gather,
            cu,
            probs,
            slots,
            parts,
            offsets,
        )
        nvfp4_quantize_rowwise(
            x,
            cu,
            scale,
            q,
            sf,
            gather_idx=gather,
            padded_offsets=offsets,
            te_math=True,
        )
        torch.cuda.synchronize()
        assert int(gather.min()) == 0
        assert int(gather.max()) == tokens - 1


def test_large_inference_plans_sparse_routing():
    from nvfp4moe import InferenceMoE
    from nvfp4moe.inference import InferenceWorkspace

    tokens, hidden, intermediate, experts, topk = 1024, 2048, 768, 128, 8
    active_sets = (
        (40, 45, 65, 73, 98, 99, 100, 114),
        (10, 64, 70, 94, 108, 111, 120, 125),
        (30, 35, 58, 92, 93, 94, 99, 110),
        (4, 14, 16, 53, 82, 103, 112, 124),
        (17, 19, 21, 24, 71, 78, 81, 126),
        (20, 26, 69, 76, 91, 98, 103, 104),
    )
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    topk_weights = torch.full((tokens, topk), 1 / topk, device="cuda")
    fc1 = torch.randint(
        256,
        (experts, 2 * intermediate, hidden // 2),
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float4_e2m1fn_x2)
    fc2 = torch.randint(
        256,
        (experts, hidden, intermediate // 2),
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float4_e2m1fn_x2)
    sf1 = torch.ones(
        experts,
        2 * intermediate // 128,
        hidden // 64,
        32,
        4,
        4,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    sf2 = torch.ones(
        experts,
        hidden // 128,
        intermediate // 64,
        32,
        4,
        4,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    weight_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    workspace = InferenceWorkspace.allocate(
        hidden,
        intermediate,
        experts,
        topk,
        tokens,
        torch.device("cuda", torch.cuda.current_device()),
    )
    plans = []
    for _ in active_sets:
        plan = InferenceMoE(
            hidden,
            intermediate,
            experts,
            topk,
            tokens,
            workspace=workspace,
        )
        plan.load_packed_weights(fc1, sf1, weight_scale, fc2, sf2, weight_scale)
        plan.set_activation_scales(1.0, 1.0)
        plans.append(plan)

    assert all(plan.workspace is workspace for plan in plans)

    for plan, active in zip(plans, active_sets, strict=True):
        topk_ids = torch.tensor(active, dtype=torch.int32, device="cuda").repeat(tokens, 1)
        plan(x, topk_ids, topk_weights)
        torch.cuda.synchronize()

    decode_tokens = 64
    token = torch.arange(decode_tokens, dtype=torch.int32, device="cuda")[:, None]
    route = torch.arange(topk - 1, dtype=torch.int32, device="cuda")[None, :]
    other_ids = ((token * (topk - 1) + route) % (experts - 1)) + 1
    decode_ids = torch.cat(
        (torch.zeros(decode_tokens, 1, dtype=torch.int32, device="cuda"), other_ids),
        dim=1,
    ).contiguous()
    decode_weights = torch.full((decode_tokens, topk), 1 / topk, dtype=torch.float32, device="cuda")
    decode = plans[0].run_decode(x[:decode_tokens], decode_ids, decode_weights)
    torch.cuda.synchronize()
    assert torch.isfinite(decode).all()


def test_grouped_gemm_accepts_per_expert_alpha():
    from nvfp4moe.gemm import GroupedGemm, quantize_grouped

    torch.manual_seed(20260822)
    experts, n, k = 4, 128, 256
    counts = torch.tensor((3, 5, 2, 7), dtype=torch.int32, device="cuda")
    m_indptr = torch.zeros(experts + 1, dtype=torch.int32, device="cuda")
    m_indptr[1:] = counts.cumsum(0)
    a = torch.randn(int(counts.sum()), k, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(experts, n, k, dtype=torch.bfloat16, device="cuda")
    qa, qb, sfa, sfb, alpha = quantize_grouped(a, b, m_indptr)
    out = torch.empty(a.shape[0], n, dtype=torch.bfloat16, device="cuda")
    gemm = GroupedGemm(experts, n, k, 128, 128)

    gemm(qa, qb, out, m_indptr, sfa, sfb, alpha)
    scalar = out.clone()
    expert_alpha = alpha * torch.arange(1, experts + 1, dtype=torch.float32, device="cuda")
    gemm(qa, qb, out, m_indptr, sfa, sfb, expert_alpha)
    torch.cuda.synchronize()

    for expert in range(experts):
        begin, end = int(m_indptr[expert]), int(m_indptr[expert + 1])
        torch.testing.assert_close(
            out[begin:end].float(),
            scalar[begin:end].float() * (expert + 1),
            rtol=0.02,
            atol=0.05,
        )
