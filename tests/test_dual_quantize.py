"""Row and column outputs from the fused quantizer match separate calls."""

import pytest
import torch


@pytest.mark.parametrize("rounding", ["rn", "sr"])
def test_dual_quantize_matches_separate(rounding):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 10:
        pytest.skip("SM100 is required")

    from nvfp4moe.kernels.quantize import (
        nvfp4_quantize_colwise,
        nvfp4_quantize_row_col,
        nvfp4_quantize_rowwise,
        nvfp4_rht_amax,
    )
    from nvfp4moe.recipe import TensorScale

    torch.manual_seed(20260812)
    counts = torch.tensor([300, 0, 137, 253], dtype=torch.int32, device="cuda")
    cu = torch.cat((counts.new_zeros(1), counts.cumsum(0, dtype=torch.int32)))
    off_pad = (((counts + 127) // 128) * 128).cumsum(0, dtype=torch.int32)
    rows, features = int(counts.sum()), 256
    padded_rows = int(off_pad[-1])
    tiles = -(-rows // 128) + counts.numel()
    x = torch.randn(rows, features, dtype=torch.bfloat16, device="cuda")

    row_scale = TensorScale()
    row_scale.update(x)
    col_scale = TensorScale(te_rht=True)
    partials = torch.zeros(tiles * (features // 128), 2, dtype=torch.float32, device="cuda")
    nvfp4_rht_amax(x, cu, partials, padded_offsets=off_pad)
    col_scale.set_amax(partials.amax(0)[1:2])

    row_q = torch.zeros(rows, features // 2, dtype=torch.uint8, device="cuda")
    dual_row_q = torch.zeros_like(row_q)
    row_sf = torch.zeros(
        1,
        tiles,
        features // 64,
        32,
        4,
        4,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    dual_row_sf = torch.zeros_like(row_sf)
    col_q = torch.zeros(features, padded_rows // 2, dtype=torch.uint8, device="cuda")
    dual_col_q = torch.zeros_like(col_q)
    col_sf = torch.zeros(
        features * padded_rows // 16,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    dual_col_sf = torch.zeros_like(col_sf)
    col_amax = torch.zeros(tiles * (features // 128), dtype=torch.float32, device="cuda")
    dual_col_amax = torch.zeros_like(col_amax)

    nvfp4_quantize_rowwise(
        x,
        cu,
        row_scale.pair,
        row_q,
        row_sf,
        rounding=rounding,
        seed=17,
        padded_offsets=off_pad,
    )
    nvfp4_quantize_colwise(
        x,
        cu,
        col_scale.pair,
        col_q,
        col_sf,
        rounding=rounding,
        seed=29,
        rht=True,
        amax_out=col_amax,
        padded_offsets=off_pad,
    )
    nvfp4_quantize_row_col(
        x,
        cu,
        row_scale.pair,
        col_scale.pair,
        dual_row_q,
        dual_row_sf,
        dual_col_q,
        dual_col_sf,
        rounding=rounding,
        row_seed=17,
        col_seed=29,
        rht=True,
        amax_out=dual_col_amax,
        padded_offsets=off_pad,
    )

    torch.cuda.synchronize()
    assert torch.equal(dual_row_q, row_q)
    assert torch.equal(dual_row_sf.view(torch.uint8), row_sf.view(torch.uint8))
    assert torch.equal(dual_col_q, col_q)
    assert torch.equal(dual_col_sf.view(torch.uint8), col_sf.view(torch.uint8))
    assert torch.equal(dual_col_amax, col_amax)
