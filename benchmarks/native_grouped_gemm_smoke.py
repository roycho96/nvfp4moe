"""Correctness and latency smoke tests for the native grouped NVFP4 GEMM."""

import argparse
import statistics

import torch

import nvfp4moe.reference as ref
from nvfp4moe.gemm import quantize
from nvfp4moe.kernels.epilogue import GatedBackwardEpilogue
from nvfp4moe.kernels.gemm import grouped_nvfp4_gemm
from nvfp4moe.kernels.quantize import nvfp4_quantize_rowwise
from nvfp4moe.layer import _quant_expert_stack
from nvfp4moe.recipe import _DEN


def _time(fn, warmup=5, iters=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _dequant_expert_rows(qdata, sf, scale, rows, features):
    dense_sf = ref.unpack_sf_blocked(sf, rows, features // 16)
    return ref.dequantize_nvfp4_lastdim(qdata[:rows], dense_sf, scale)


def _check_gemm_samples(output, qa_u8, qb, sfa, sfb, cu, pts_a, pts_b):
    offsets = ref.varlen_sf_tile_offsets(cu.cpu())
    cu_host = cu.cpu().tolist()
    for expert in range(len(cu_host) - 1):
        lo, hi = cu_host[expert], cu_host[expert + 1]
        take = min(4, hi - lo)
        if take == 0:
            continue
        a_sf = sfa[0, offsets[expert] : offsets[expert] + ref.ceil_div(hi - lo, 128)]
        a = _dequant_expert_rows(qa_u8[lo:hi], a_sf, pts_a, hi - lo, qa_u8.shape[1] * 2)
        b = _dequant_expert_rows(
            qb[expert].view(torch.uint8),
            sfb[expert],
            pts_b,
            qb.shape[1],
            qb.shape[2] * 2,
        )
        expected = a[:take] @ b.T
        torch.testing.assert_close(output[lo : lo + take].float(), expected, rtol=2e-2, atol=2e-2)


def _activation_backward(gate, up, dout, kind):
    if kind == "swiglu":
        sigmoid = torch.sigmoid(gate)
        act = gate * sigmoid
        dact = sigmoid * (1.0 + gate * (1.0 - sigmoid))
    elif kind == "geglu":
        coeff = 0.7978845608028654
        z = coeff * (gate + 0.044715 * gate * gate * gate)
        tanh_z = torch.tanh(z)
        act = 0.5 * gate * (1.0 + tanh_z)
        dact = 0.5 * (1.0 + tanh_z) + 0.5 * gate * (1.0 - tanh_z * tanh_z) * coeff * (
            1.0 + 3.0 * 0.044715 * gate * gate
        )
    else:
        act = torch.relu(gate)
        dact = (gate > 0).float()
    return dout * up * dact, dout * act, act * up


def _check_dgrad_samples(
    dh,
    aux,
    qa_u8,
    qb,
    sfa,
    sfb,
    cu,
    pts_a,
    pts_b,
    preact,
    activation,
):
    offsets = ref.varlen_sf_tile_offsets(cu.cpu())
    cu_host = cu.cpu().tolist()
    for expert in range(len(cu_host) - 1):
        lo, hi = cu_host[expert], cu_host[expert + 1]
        take = min(4, hi - lo)
        if take == 0:
            continue
        a_sf = sfa[0, offsets[expert] : offsets[expert] + ref.ceil_div(hi - lo, 128)]
        a = _dequant_expert_rows(qa_u8[lo:hi], a_sf, pts_a, hi - lo, qa_u8.shape[1] * 2)
        b = _dequant_expert_rows(
            qb[expert].view(torch.uint8),
            sfb[expert],
            pts_b,
            qb.shape[1],
            qb.shape[2] * 2,
        )
        dout = a[:take] @ b.T
        gate = preact[lo : lo + take, 0::2].float()
        up = preact[lo : lo + take, 1::2].float()
        dgate, dup, expected_aux = _activation_backward(gate, up, dout, activation)
        expected_dh = torch.stack((dgate, dup), dim=-1).flatten(1)
        torch.testing.assert_close(dh[lo : lo + take].float(), expected_dh, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(aux[lo : lo + take].float(), expected_aux, rtol=3e-2, atol=3e-2)


def run_case(
    experts,
    n,
    k,
    rows,
    tile_m=128,
    tile_n=128,
    output_dtype=torch.bfloat16,
    profile_only=False,
):
    torch.manual_seed(2026)
    device = "cuda"
    counts = torch.full((experts,), rows // experts, dtype=torch.int32, device=device)
    counts[: rows % experts] += 1
    if experts >= 4:
        counts[-1] = 0
        counts[0] += rows // experts
    cu = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=device),
            counts.cumsum(0).to(torch.int32),
        )
    )
    total = int(cu[-1])

    a = torch.randn(total, k, dtype=torch.bfloat16, device=device) * k**-0.5
    b = torch.randn(experts, n, k, dtype=torch.bfloat16, device=device) * k**-0.5
    pts_a = (a.abs().amax().float() / _DEN).clamp_min(1e-30).reshape(1)
    pts_b = (b.abs().amax().float() / _DEN).clamp_min(1e-30).reshape(1)
    pair_a = torch.cat((pts_a, 1.0 / pts_a))

    qa_u8 = torch.empty(total, k // 2, dtype=torch.uint8, device=device)
    rm = -(-total // 128) + experts
    sfa = torch.zeros(
        1,
        rm,
        k // 64,
        32,
        4,
        4,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    nvfp4_quantize_rowwise(a, cu, pair_a, qa_u8, sfa)
    qb, sfb = _quant_expert_stack([b[e] for e in range(experts)], pts_b)
    qa = qa_u8.view(torch.float4_e2m1fn_x2)
    alpha = pts_a * pts_b

    native_out = torch.empty(total, n, dtype=output_dtype, device=device)
    native = grouped_nvfp4_gemm(
        experts,
        n,
        k,
        tile_m,
        tile_n,
        output_dtype=output_dtype,
    )

    def native_call():
        native(qa, qb, native_out, cu, sfa, sfb, alpha)

    native_call()
    torch.cuda.synchronize()
    _check_gemm_samples(native_out, qa_u8, qb, sfa, sfb, cu, pts_a, pts_b)
    if profile_only:
        native.prepare(qa, qb, native_out, cu, sfa, sfb, alpha)
        torch.cuda.nvtx.range_push("nvfp4moe_profile")
        native.launch()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        return None
    native_ms = _time(native_call)
    prepare_ms = _time(lambda: native.prepare(qa, qb, native_out, cu, sfa, sfb, alpha))
    native.prepare(qa, qb, native_out, cu, sfa, sfb, alpha)
    kernel_ms = _time(native.launch)
    result = {
        "experts": experts,
        "m": total,
        "n": n,
        "k": k,
        "tile": (tile_m, tile_n),
        "output_dtype": str(output_dtype),
        "stages": native.stages,
        "native_ms": native_ms,
        "prepare_ms": prepare_ms,
        "kernel_ms": kernel_ms,
    }
    print(result, flush=True)
    return result


def run_dense_profile(n, k, rows, tile_m, tile_n):
    from nvfp4moe.kernels.dense_gemm import DenseNvfp4Gemm

    torch.manual_seed(20260812)
    a = torch.randn(rows, k, dtype=torch.bfloat16, device="cuda") * k**-0.5
    b = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * k**-0.5
    one = torch.ones(1, dtype=torch.float32, device="cuda")
    qa, sfa, _ = quantize(a, one)
    qb, sfb, _ = quantize(b, one)
    out = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
    gemm = DenseNvfp4Gemm(n, k, tile_m, tile_n)
    gemm(qa, qb, out, sfa, sfb, one)
    torch.cuda.synchronize()
    gemm.prepare(qa, qb, out, sfa, sfb, one)
    torch.cuda.nvtx.range_push("nvfp4moe_profile")
    gemm.launch()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


def run_dgrad2_case(
    experts=128,
    n=768,
    k=2048,
    rows=65_536,
    activation="swiglu",
    tile_m=256,
    tile_n=128,
    profile_only=False,
):
    torch.manual_seed(2027)
    device = "cuda"
    counts = torch.full((experts,), rows // experts, dtype=torch.int32, device=device)
    counts[: rows % experts] += 1
    counts[-1] = 0
    counts[0] += rows // experts
    cu = torch.cat(
        (torch.zeros(1, dtype=torch.int32, device=device), counts.cumsum(0).to(torch.int32))
    )
    total = int(cu[-1])

    dout = torch.randn(total, k, dtype=torch.bfloat16, device=device) * k**-0.5
    weight = torch.randn(experts, n, k, dtype=torch.bfloat16, device=device) * k**-0.5
    preact = torch.randn(total, 2 * n, dtype=torch.bfloat16, device=device)
    pts_a = (dout.abs().amax().float() / _DEN).clamp_min(1e-30).reshape(1)
    pts_b = (weight.abs().amax().float() / _DEN).clamp_min(1e-30).reshape(1)
    pair_a = torch.cat((pts_a, 1.0 / pts_a))
    qa_u8 = torch.empty(total, k // 2, dtype=torch.uint8, device=device)
    rm = -(-total // 128) + experts
    sfa = torch.zeros(1, rm, k // 64, 32, 4, 4, dtype=torch.float8_e4m3fn, device=device)
    nvfp4_quantize_rowwise(dout, cu, pair_a, qa_u8, sfa)
    qb, sfb = _quant_expert_stack([weight[e] for e in range(experts)], pts_b)
    qa = qa_u8.view(torch.float4_e2m1fn_x2)
    alpha = pts_a * pts_b

    native_dh = torch.empty(total, 2 * n, dtype=torch.bfloat16, device=device)
    native_aux = torch.empty(total, n, dtype=torch.bfloat16, device=device)
    native = grouped_nvfp4_gemm(
        experts,
        n,
        k,
        tile_m,
        tile_n,
        output_dtype=torch.int32,
        epilogue=GatedBackwardEpilogue(activation),
    )

    def native_call():
        native(
            qa,
            qb,
            native_dh.view(torch.int32),
            cu,
            sfa,
            sfb,
            alpha,
            preact=preact,
            aux=native_aux,
        )

    def native_prepare():
        native.prepare(
            qa,
            qb,
            native_dh.view(torch.int32),
            cu,
            sfa,
            sfb,
            alpha,
            preact=preact,
            aux=native_aux,
        )

    native_call()
    torch.cuda.synchronize()
    _check_dgrad_samples(
        native_dh,
        native_aux,
        qa_u8,
        qb,
        sfa,
        sfb,
        cu,
        pts_a,
        pts_b,
        preact,
        activation,
    )
    first_dh = native_dh.clone()
    first_aux = native_aux.clone()
    native_call()
    torch.cuda.synchronize()
    if not torch.equal(native_dh, first_dh) or not torch.equal(native_aux, first_aux):
        raise AssertionError("native dgrad2 is not deterministic")
    if profile_only:
        native_prepare()
        torch.cuda.nvtx.range_push("nvfp4moe_profile")
        native.launch()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        return None
    native_ms = _time(native_call)
    prepare_ms = _time(native_prepare)
    native_prepare()
    kernel_ms = _time(native.launch)
    result = {
        "operation": "dgrad2",
        "activation": activation,
        "experts": experts,
        "m": total,
        "n": n,
        "k": k,
        "tile": (tile_m, tile_n),
        "stages": native.stages,
        "native_ms": native_ms,
        "prepare_ms": prepare_ms,
        "kernel_ms": kernel_ms,
    }
    print(result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--scheduler-edge", action="store_true")
    parser.add_argument("--dgrad-matrix", action="store_true")
    parser.add_argument("--frontier-matrix", action="store_true")
    parser.add_argument("--training-tile-matrix", action="store_true")
    parser.add_argument(
        "--profile-case",
        choices=(
            "dgrad2-qwen",
            "dgrad2-kimi",
            "dgrad2-deepseek",
            "dgrad2-deepseek-local",
            "fc1-deepseek-local",
            "fc2-deepseek-local",
            "dgrad1-deepseek-local",
            "dense-qwen-fc2",
            "dense-deepseek-fc1",
            "dense-deepseek-fc2",
        ),
    )
    parser.add_argument("--output-dtype", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    output_dtype = torch.float32 if args.output_dtype == "fp32" else torch.bfloat16

    if args.profile_case == "dgrad2-qwen":
        run_dgrad2_case(profile_only=True)
        return
    if args.profile_case == "dgrad2-kimi":
        run_dgrad2_case(
            experts=128,
            n=2048,
            k=7168,
            rows=21_846,
            tile_m=256,
            tile_n=256,
            profile_only=True,
        )
        return
    if args.profile_case == "dgrad2-deepseek":
        run_dgrad2_case(
            experts=256,
            n=2048,
            k=7168,
            rows=65_536,
            tile_m=256,
            tile_n=256,
            profile_only=True,
        )
        return
    if args.profile_case == "dgrad2-deepseek-local":
        run_dgrad2_case(
            experts=8,
            n=2048,
            k=7168,
            rows=65_536,
            tile_m=256,
            tile_n=256,
            profile_only=True,
        )
        return
    if args.profile_case == "fc1-deepseek-local":
        run_case(8, 4096, 7168, 65_536, 256, 256, profile_only=True)
        return
    if args.profile_case == "fc2-deepseek-local":
        run_case(8, 7168, 2048, 65_536, 256, 256, profile_only=True)
        return
    if args.profile_case == "dgrad1-deepseek-local":
        run_case(8, 7168, 4096, 65_536, 256, 256, profile_only=True)
        return
    if args.profile_case == "dense-qwen-fc2":
        run_dense_profile(2048, 768, 8192, 256, 128)
        return
    if args.profile_case == "dense-deepseek-fc1":
        run_dense_profile(4096, 7168, 8192, 256, 256)
        return
    if args.profile_case == "dense-deepseek-fc2":
        run_dense_profile(7168, 2048, 8192, 256, 256)
        return

    if args.frontier_matrix:
        cases = [
            ("kimi_k2_7_ep3", 128, 2048, 7168, 21_846),
            ("minimax_m2", 256, 1536, 3072, 65_536),
            ("deepseek_v3_2", 256, 2048, 7168, 65_536),
        ]
        results = []
        for model, experts, n, k, rows in cases:
            tiles = ((128, 128), (128, 256), (256, 128), (256, 256))
            for tile_m, tile_n in tiles:
                result = run_dgrad2_case(
                    experts,
                    n,
                    k,
                    rows,
                    "swiglu",
                    tile_m,
                    tile_n,
                )
                result["model"] = model
                results.append(result)
            torch.cuda.empty_cache()
        print(results, flush=True)
        return

    if args.training_tile_matrix:
        results = []
        tiles = ((128, 128), (128, 256), (256, 128), (256, 256))
        for name, n, k in (
            ("fc1", 4096, 7168),
            ("fc2", 7168, 2048),
            ("dgrad1", 7168, 4096),
        ):
            for tile_m, tile_n in tiles:
                result = run_case(8, n, k, 65_536, tile_m, tile_n)
                result["operation"] = name
                results.append(result)
            torch.cuda.empty_cache()
        for tile_m, tile_n in tiles:
            result = run_dgrad2_case(
                8,
                2048,
                7168,
                65_536,
                "swiglu",
                tile_m,
                tile_n,
            )
            result["model"] = "deepseek_v3_2"
            results.append(result)
        print(results, flush=True)
        return

    if args.dgrad_matrix:
        cases = [
            ("qwen3_30b", 128, 768, 2048, 65_536, "swiglu"),
            ("gemma4_26b_padded", 128, 768, 2816, 65_536, "geglu"),
            ("olmoe_7b", 64, 1024, 2048, 65_536, "swiglu"),
            ("reglu_synthetic", 8, 512, 512, 4096, "reglu"),
        ]
        results = []
        for model, experts, n, k, rows, activation in cases:
            tiles = ((128, 128), (128, 256), (256, 128), (256, 256))
            for tile_m, tile_n in tiles:
                result = run_dgrad2_case(
                    experts,
                    n,
                    k,
                    rows,
                    activation,
                    tile_m,
                    tile_n,
                )
                result["model"] = model
                results.append(result)
            torch.cuda.empty_cache()
        print(results, flush=True)
        return

    if args.scheduler_edge:
        results = [
            run_case(1, 128, 256, rows, output_dtype=output_dtype) for rows in (1, 129, 257, 385)
        ]
        results.append(run_case(4, 128, 256, 129, output_dtype=output_dtype))
        print(results, flush=True)
        return

    results = [run_case(8, 512, 512, 2048, output_dtype=output_dtype)]
    if output_dtype == torch.bfloat16:
        results.append(run_dgrad2_case())
    if args.quick:
        print(results, flush=True)
        return
    results.append(
        run_case(
            128,
            2048,
            768,
            8192 * 8,
            tile_n=256,
            output_dtype=output_dtype,
        )
    )
    for tile_m, tile_n in ((128, 128), (128, 256), (256, 128), (256, 256)):
        results.append(
            run_case(
                128,
                2048,
                1536,
                8192 * 8,
                tile_m=tile_m,
                tile_n=tile_n,
                output_dtype=output_dtype,
            )
        )
    print(results, flush=True)


if __name__ == "__main__":
    main()
