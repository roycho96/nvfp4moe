import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from nvfp4moe.ops import grouped_nvfp4_gemm, nvfp4_gemm, nvfp4_gemm_out, nvfp4_quantize


def _fake_inputs(experts=4, rows=17, n=256, k=256):
    mode = FakeTensorMode()
    with mode:
        a = torch.empty(rows, k // 2, device="cuda", dtype=torch.float4_e2m1fn_x2)
        b = torch.empty(experts, n, k // 2, device="cuda", dtype=torch.float4_e2m1fn_x2)
        sfa = torch.empty(1, 5, k // 64, 32, 4, 4, device="cuda", dtype=torch.float8_e4m3fn)
        sfb = torch.empty(
            experts,
            n // 128,
            k // 64,
            32,
            4,
            4,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        cu = torch.empty(experts + 1, device="cuda", dtype=torch.int32)
        alpha = torch.empty(1, device="cuda")
    return mode, (a, b, sfa, sfb, cu, alpha)


@pytest.mark.parametrize("activation, output_n", [(None, 256), ("swiglu", 128)])
def test_grouped_gemm_fake_shape(activation, output_n):
    mode, args = _fake_inputs()
    with mode:
        out = grouped_nvfp4_gemm(*args, activation=activation)
    assert out.shape == (17, output_n)
    assert out.dtype == torch.bfloat16


def test_grouped_gemm_rejects_mismatched_offsets():
    mode, args = _fake_inputs()
    with mode:
        bad_cu = torch.empty(4, device="cuda", dtype=torch.int32)
        with pytest.raises(ValueError, match="one offset per expert"):
            grouped_nvfp4_gemm(*args[:4], bad_cu, args[5])


def test_grouped_gemm_is_registered_as_custom_op():
    assert hasattr(torch.ops.nvfp4moe, "grouped_gemm")


def test_dense_gemm_fake_shape():
    mode = FakeTensorMode()
    with mode:
        a = torch.empty(17, 128, device="cuda", dtype=torch.float4_e2m1fn_x2)
        b = torch.empty(384, 128, device="cuda", dtype=torch.float4_e2m1fn_x2)
        sfa = torch.empty(1, 4, 32, 4, 4, device="cuda", dtype=torch.float8_e4m3fn)
        sfb = torch.empty(3, 4, 32, 4, 4, device="cuda", dtype=torch.float8_e4m3fn)
        alpha = torch.ones(1, device="cuda")
        out = nvfp4_gemm(a, b, sfa, sfb, alpha, tile_n=128)
    assert out.shape == (17, 384)
    assert out.dtype == torch.bfloat16


def test_dense_gemm_is_registered_as_custom_op():
    assert hasattr(torch.ops.nvfp4moe, "gemm")
    assert hasattr(torch.ops.nvfp4moe, "gemm_out")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dense_gemm_matches_single_group_and_compiles():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("native kernel requires SM100")

    torch.manual_seed(7)
    n, k = 256, 256
    b = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    qb, sfb, pts_b = nvfp4_quantize(b)
    compiled = torch.compile(nvfp4_gemm, fullgraph=True, dynamic=True)
    compiled_out = torch.compile(nvfp4_gemm_out, fullgraph=True, dynamic=True)
    for rows in (256, 384):
        a = torch.randn(rows, k, dtype=torch.bfloat16, device="cuda")
        qa, sfa, pts_a = nvfp4_quantize(a)
        alpha = pts_a * pts_b
        dense = nvfp4_gemm(qa, qb, sfa, sfb, alpha)
        out = torch.empty_like(dense)
        assert torch.equal(nvfp4_gemm_out(qa, qb, sfa, sfb, alpha, out), dense)
        compiled_out_buffer = torch.empty_like(dense)
        assert torch.equal(
            compiled_out(qa, qb, sfa, sfb, alpha, compiled_out_buffer),
            dense,
        )
        cu = torch.tensor([0, rows], dtype=torch.int32, device="cuda")
        grouped_sfa = torch.zeros(
            1,
            sfa.shape[0] + 1,
            *sfa.shape[1:],
            dtype=sfa.dtype,
            device=sfa.device,
        )
        grouped_sfa[0, : sfa.shape[0]].copy_(sfa)
        grouped = grouped_nvfp4_gemm(
            qa,
            qb.unsqueeze(0),
            grouped_sfa,
            sfb.unsqueeze(0),
            cu,
            alpha,
        )
        assert torch.equal(dense, grouped)
        assert torch.equal(compiled(qa, qb, sfa, sfb, alpha), dense)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dense_gemm_tile_variants_are_bitwise_equal():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("native kernel requires SM100")

    torch.manual_seed(11)
    rows, n, k = 384, 384, 256
    a = torch.randn(rows, k, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    qa, sfa, pts_a = nvfp4_quantize(a)
    qb, sfb, pts_b = nvfp4_quantize(b)
    alpha = pts_a * pts_b
    expected = nvfp4_gemm(qa, qb, sfa, sfb, alpha, tile_m=128, tile_n=128)

    for tile_m in (128, 256):
        for tile_n in (64, 128, 192, 256):
            actual = nvfp4_gemm(
                qa,
                qb,
                sfa,
                sfb,
                alpha,
                tile_m=tile_m,
                tile_n=tile_n,
            )
            assert torch.equal(actual, expected), (tile_m, tile_n)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_grouped_gemm_torch_compile_dynamic_rows():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("native kernel requires SM100")

    experts, n, k = 2, 256, 256
    qb = torch.randint(
        0,
        256,
        (experts, n, k // 2),
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float4_e2m1fn_x2)
    sfb = torch.ones(
        experts,
        n // 128,
        k // 64,
        32,
        4,
        4,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    alpha = torch.ones(1, dtype=torch.float32, device="cuda")
    compiled = torch.compile(grouped_nvfp4_gemm, fullgraph=True, dynamic=True)

    for rows in (256, 384):
        qa = torch.randint(
            0,
            256,
            (rows, k // 2),
            dtype=torch.uint8,
            device="cuda",
        ).view(torch.float4_e2m1fn_x2)
        sfa = torch.ones(
            1,
            -(-rows // 128) + experts,
            k // 64,
            32,
            4,
            4,
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        cu = torch.tensor([0, rows // 2, rows], dtype=torch.int32, device="cuda")
        eager = grouped_nvfp4_gemm(qa, qb, sfa, sfb, cu, alpha)
        actual = compiled(qa, qb, sfa, sfb, cu, alpha)
        assert torch.equal(actual, eager)
