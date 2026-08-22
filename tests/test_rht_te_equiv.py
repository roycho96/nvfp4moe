"""Compare RHT NVFP4 quantization with Transformer Engine on B200.

The rowwise and post-RHT columnwise amax values, columnwise scale factors,
and columnwise data must match bitwise.

    PYTHONPATH=. python tests/test_rht_te_equiv.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}", flush=True)
    if not cond:
        FAILURES.append(name)


def main():
    if torch.cuda.get_device_capability(0)[0] != 10:
        print("SKIP: requires SM100")
        return 0

    import transformer_engine as te
    import transformer_engine.pytorch
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    print(f"TE version: {te.__version__}")

    from benchmarks import reference as nref
    from lightmoe._quantization import TensorScale
    from lightmoe.kernels.quantize import (
        nvfp4_quantize_colwise,
        nvfp4_quantize_rowwise,
        nvfp4_rht_amax,
    )

    torch.manual_seed(0)
    M, F = 4096, 2048  # single segment, M % 128 == 0 (no padding ambiguity)
    x = torch.randn(M, F, device="cuda", dtype=torch.bfloat16)

    # ---- TE quantize (rowwise + columnwise, RHT on, post-RHT amax, RN)
    quantizer = NVFP4Quantizer(
        rowwise=True,
        columnwise=True,
        with_rht=True,
        with_post_rht_amax=True,
        with_2d_quantization=False,
        stochastic_rounding=False,
        with_random_sign_mask=True,
    )
    qt = quantizer.quantize(x)
    print("TE tensor attrs:")
    for a in dir(qt):
        if a.startswith("_") and isinstance(getattr(qt, a, None), torch.Tensor):
            t = getattr(qt, a)
            print(f"    {a}: shape {tuple(t.shape)} dtype {t.dtype}")
    check("TE quantizer ran", True)

    # TE 2.17 fields: data (rows, F/2) u8; scale_inv LINEAR (rows, F/16) u8
    te_row = qt._rowwise_data.view(torch.uint8)
    te_col = qt._columnwise_data.view(torch.uint8)
    te_row_sf = qt._rowwise_scale_inv.view(torch.uint8)
    te_col_sf = qt._columnwise_scale_inv.view(torch.uint8)
    te_amax_row = qt._amax_rowwise.float().reshape(())
    te_amax_col = qt._amax_columnwise.float().reshape(())
    print(f"  te amax row {te_amax_row.item():.8e} col {te_amax_col.item():.8e}")

    # Compare the fused amax pre-pass with TE.
    cu = torch.tensor([0, M], dtype=torch.int32, device="cuda")
    n_tiles = -(-M // 128) + 1
    part = torch.zeros(n_tiles * (F // 128), 2, device="cuda")
    nvfp4_rht_amax(x, cu, part)
    red = part.amax(0)
    check(
        "rowwise amax bitwise",
        torch.equal(red[0], te_amax_row),
        f"{red[0].item():.8e} vs {te_amax_row.item():.8e}",
    )
    check(
        "post-RHT col amax bitwise",
        torch.equal(red[1], te_amax_col),
        f"{red[1].item():.8e} vs {te_amax_col.item():.8e}",
    )

    # Quantize the same data with TE-compatible RHT scale math.
    ts = TensorScale(te_rht=True)
    ts.set_amax(red[1:2])
    q = torch.zeros(F, M // 2, dtype=torch.uint8, device="cuda")
    sf = torch.zeros(F * M // 16, dtype=torch.float8_e4m3fn, device="cuda")
    nvfp4_quantize_colwise(x, cu, ts.pair, q, sf, rht=True)

    # TE col SF is LINEAR (F, M/16); unblock ours to the same 2D layout
    ours_sf_2d = nref.unpack_sf_blocked(sf.view(F // 128, M // 64, 32, 4, 4), F, M // 16).view(
        torch.uint8
    )
    if tuple(te_col_sf.shape) == (F, M // 16):
        sf_mism = (ours_sf_2d != te_col_sf).sum().item()
        check("col SF bitwise vs TE", sf_mism == 0, f"mismatch {sf_mism}/{te_col_sf.numel()}")
        if te_col.shape != q.shape:
            te_col = te_col.reshape(q.shape)
        d_mism = (q != te_col).sum().item()
        check("col data bitwise vs TE", d_mism == 0, f"mismatch {d_mism}/{q.numel()}")
    else:
        check("col SF layout recognized", False, f"TE col SF shape {tuple(te_col_sf.shape)}")

    # ---- rowwise (no RHT) comparison, same scale-path caveat
    ts_r = TensorScale()
    ts_r.set_amax(red[0:1])
    qr = torch.zeros(M, F // 2, dtype=torch.uint8, device="cuda")
    sfr = torch.zeros(1, n_tiles, F // 64, 32, 4, 4, dtype=torch.float8_e4m3fn, device="cuda")
    nvfp4_quantize_rowwise(x, cu, ts_r.pair, qr, sfr)
    if te_row.shape != qr.shape:
        te_row = te_row.reshape(qr.shape)
    print(f"  row data bytes ours vs TE: {(qr != te_row).sum().item()}/{qr.numel()}")
    if tuple(te_row_sf.shape) == (M, F // 16):
        ours_row_sf = nref.unpack_sf_blocked(sfr[0, : M // 128], M, F // 16).view(torch.uint8)
        print(f"  row SF bytes ours vs TE: {(ours_row_sf != te_row_sf).sum().item()}/{M * F // 16}")
    else:
        print(f"  [!] TE row SF shape {tuple(te_row_sf.shape)} != ({M}, {F // 16})")

    print(f"\n{'ALL PASS' if not FAILURES else f'{len(FAILURES)} FAILURES: {FAILURES}'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
