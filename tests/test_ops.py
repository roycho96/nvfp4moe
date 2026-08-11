import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from nvfp4moe.ops import grouped_nvfp4_gemm


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
