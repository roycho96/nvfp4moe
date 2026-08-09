"""Validate kernels/nvfp4_ref.py against quack's own quantizer and SF packer.

The reference is the oracle for every fast implementation, so it has to be
bit-exact with quack -- otherwise a "correct" P1 kernel would still produce
operands quack's GEMM reads differently. Runs on any CUDA device (no sm100
needed); pure torch.

    python tests/test_ref_vs_quack.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "refs" / "quack"))

from kernels import nvfp4_ref as ref  # noqa: E402

try:
    from quack.blockscaled.quantize import (  # noqa: E402
        nvfp4_per_tensor_scale,
        pack_scale_2d_to_blocked_contig,
        to_nvfp4,
        unpack_scale_blocked_to_2d,
    )

    HAVE_QUACK = True
except Exception as e:  # noqa: BLE001
    print(f"[warn] quack import failed ({type(e).__name__}: {e}); quack cross-checks skipped")
    HAVE_QUACK = False

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def test_quantize_matches_quack():
    print("\n[1] quantize_nvfp4_lastdim vs quack.to_nvfp4 (bit-exact)")
    if not HAVE_QUACK:
        print("  SKIP")
        return
    torch.manual_seed(0)
    for shape in [(256, 128), (1024, 2048), (37 * 16, 768)]:
        x = torch.randn(*shape, device=DEV, dtype=torch.bfloat16)
        pts = nvfp4_per_tensor_scale(x.abs().max())

        q_ref, sf_ref = ref.quantize_nvfp4_lastdim(x, pts)
        q_q, sf_q, pts_q = to_nvfp4(x, per_tensor_scale=pts)

        check(
            f"qdata {shape}",
            torch.equal(q_ref, q_q),
            f"mismatched bytes: {(q_ref != q_q).sum().item()}/{q_ref.numel()}",
        )
        check(
            f"sf    {shape}",
            torch.equal(sf_ref.view(torch.uint8), sf_q.view(torch.uint8)),
            f"mismatched: {(sf_ref.view(torch.uint8) != sf_q.view(torch.uint8)).sum().item()}",
        )
        check(f"pts   {shape}", torch.equal(pts.float(), pts_q.float()))


def test_sf_blocked_matches_quack():
    print("\n[2] pack_sf_blocked vs quack.pack_scale_2d_to_blocked_contig")
    if not HAVE_QUACK:
        print("  SKIP")
        return
    torch.manual_seed(1)
    for mn, sf_k in [(128, 4), (256, 128), (300, 47), (1024, 48)]:
        sf = (torch.randn(mn, sf_k, device=DEV) * 3).to(torch.float8_e4m3fn)
        mine = ref.pack_sf_blocked(sf)
        theirs = pack_scale_2d_to_blocked_contig(sf).squeeze(0)
        check(
            f"blocked ({mn},{sf_k}) shape",
            tuple(mine.shape) == tuple(theirs.shape),
            f"{tuple(mine.shape)} vs {tuple(theirs.shape)}",
        )
        if tuple(mine.shape) == tuple(theirs.shape):
            check(
                f"blocked ({mn},{sf_k}) bytes",
                torch.equal(mine.view(torch.uint8), theirs.view(torch.uint8)),
            )
        # and my unpack inverts quack's pack
        back = ref.unpack_sf_blocked(theirs, mn, sf_k)
        check(
            f"unpack  ({mn},{sf_k})",
            torch.equal(back.view(torch.uint8), sf.view(torch.uint8)),
        )
        if HAVE_QUACK:
            q_back = unpack_scale_blocked_to_2d(mine.unsqueeze(0), mn, sf_k).squeeze(0)
            check(
                f"xcheck  ({mn},{sf_k})",
                torch.equal(q_back.view(torch.uint8), sf.view(torch.uint8)),
            )


def test_atom_index_mapping():
    """The mapping P1's store must implement: row m -> atom[m%32][m//32][k%4]."""
    print("\n[3] SF atom index mapping (row m -> [m%32][m//32][k%4])")
    mn, sf_k = 128, 4
    sf2d = torch.arange(mn * sf_k, device=DEV, dtype=torch.float32).remainder(200)
    sf2d = sf2d.reshape(mn, sf_k).to(torch.float8_e4m3fn)
    blocked = ref.pack_sf_blocked(sf2d)
    ok = True
    for m in (0, 1, 31, 32, 33, 95, 127):
        for k in range(4):
            got = blocked.view(torch.uint8)[0, 0, m % 32, m // 32, k].item()
            want = sf2d.view(torch.uint8)[m, k].item()
            if got != want:
                ok = False
                print(f"    m={m} k={k}: got {got}, want {want}")
    check("atom mapping", ok)


def test_roundtrip_accuracy():
    print("\n[4] dequantize(quantize(x)) accuracy")
    torch.manual_seed(2)
    x = torch.randn(2048, 1024, device=DEV, dtype=torch.bfloat16)
    pts = ref.per_tensor_scale_from_amax(x.abs().max())
    q, sf = ref.quantize_nvfp4_lastdim(x, pts)
    xr = ref.dequantize_nvfp4_lastdim(q, sf, pts)
    rel = (xr - x.float()).norm() / x.float().norm()
    # NVFP4 with 1x16 blocks lands around 6-9% relative L2 on gaussian data.
    check("rel L2 < 0.12", rel.item() < 0.12, f"rel={rel.item():.4f}")
    cos = torch.nn.functional.cosine_similarity(xr.flatten(), x.float().flatten(), dim=0)
    check("cosine > 0.99", cos.item() > 0.99, f"cos={cos.item():.5f}")


def test_varlen_offsets():
    print("\n[5] varlen SF tile offsets (quack offset_batch_SFA contract)")
    cu = torch.tensor([0, 100, 300, 300, 812], dtype=torch.int32)
    offs = ref.varlen_sf_tile_offsets(cu)
    # cu[b]//128 + b -> 0//128+0, 100//128+1, 300//128+2, 300//128+3
    check("offsets", offs == [0, 1, 4, 5], f"got {offs}")
    # each expert's tiles must not overlap the next expert's start
    ok = True
    for b in range(len(offs)):
        n = ref.ceil_div(int(cu[b + 1]) - int(cu[b]), 128)
        nxt = offs[b + 1] if b + 1 < len(offs) else ref.varlen_sf_num_tiles(cu)
        if offs[b] + n > nxt:
            ok = False
            print(f"    expert {b}: {offs[b]}+{n} > {nxt}")
    check("no tile overlap between experts", ok)
    check("total tiles", ref.varlen_sf_num_tiles(cu) == 9, f"got {ref.varlen_sf_num_tiles(cu)}")


def test_fused_op_consistency():
    print("\n[6] fused_gather_dual_quantize_ref internal consistency")
    torch.manual_seed(3)
    T, d, E, k = 512, 256, 4, 2
    x = torch.randn(T, d, device=DEV, dtype=torch.bfloat16)
    M = T * k
    gather_idx = torch.randint(0, T, (M,), device=DEV, dtype=torch.int32)
    counts = torch.tensor([M // E] * E, dtype=torch.int32)
    cu = torch.cat([torch.zeros(1, dtype=torch.int32), counts.cumsum(0).to(torch.int32)])
    pts = ref.per_tensor_scale_from_amax(x.abs().max())

    out = ref.fused_gather_dual_quantize_ref(x, gather_idx, cu.to(DEV), pts)
    xg = x[gather_idx.long()]

    # rowwise qdata must equal quantizing the gathered tensor directly
    q_direct, sf_direct = ref.quantize_nvfp4_lastdim(xg, pts)
    check("rowwise qdata == quantize(gather(x))", torch.equal(out["rowwise"]["qdata"], q_direct))

    # each expert's SF block, read back from the padded buffer, must match
    sf_buf = out["rowwise"]["sf"]
    offs = out["rowwise"]["sf_tile_offsets"].tolist()
    cul = cu.tolist()
    ok = True
    for b in range(E):
        lo, hi = cul[b], cul[b + 1]
        n = hi - lo
        blk = sf_buf[offs[b] : offs[b] + ref.ceil_div(n, 128)]
        got = ref.unpack_sf_blocked(blk, n, d // 16)
        if not torch.equal(got.view(torch.uint8), sf_direct[lo:hi].view(torch.uint8)):
            ok = False
            print(f"    expert {b} SF mismatch")
    check("rowwise SF readback per expert", ok)

    # colwise: dequantizing a segment must approximate the gathered segment's transpose
    col = out["colwise"]
    seg_off = col["seg_offsets"].tolist()
    seg_len = col["seg_padded_lens"].tolist()
    b = 1
    lo, hi = cul[b], cul[b + 1]
    s0, sl = seg_off[b], seg_len[b]
    qd = col["qdata"][:, s0 // 2 : (s0 + sl) // 2]
    sfd = col["sf_2d"][:, s0 // 16 : (s0 + sl) // 16]
    deq = ref.dequantize_nvfp4_lastdim(qd, sfd, col["per_tensor_scale"])[:, : hi - lo]
    tgt = xg[lo:hi].t().float()
    cos = torch.nn.functional.cosine_similarity(deq.flatten(), tgt.flatten(), dim=0)
    check("colwise segment cosine > 0.99", cos.item() > 0.99, f"cos={cos.item():.5f}")


def test_traffic_model():
    """Matched L2-reuse assumptions on both sides. PLAN.md's original 2.3-5.2x
    mixed regimes (no-reuse baseline vs reuse fused); the honest band is
    2.3x-3.9x and this test pins it so the claim cannot drift back."""
    print("\n[7] traffic model, matched reuse assumptions")
    t = ref.traffic_bytes(T=32768, d=2048, k=8)
    for r in ("noreuse", "reuse"):
        print(f"    {r:8s} baseline {t[f'baseline_{r}']/1e9:.2f} GB -> fused "
              f"{t[f'fused_{r}']/1e9:.2f} GB  = {t[f'speedup_{r}']:.2f}x")
    check("no-reuse speedup ~2.3x", 2.2 <= t["speedup_noreuse"] <= 2.4,
          f"{t['speedup_noreuse']:.2f}")
    check("full-reuse speedup ~3.9x", 3.8 <= t["speedup_reuse"] <= 4.0,
          f"{t['speedup_reuse']:.2f}")
    check("no regime reaches the retracted 5.2x",
          max(t["speedup_noreuse"], t["speedup_reuse"]) < 4.5)


if __name__ == "__main__":
    print(f"device={DEV}  torch={torch.__version__}  quack={'yes' if HAVE_QUACK else 'no'}")
    test_quantize_matches_quack()
    test_sf_blocked_matches_quack()
    test_atom_index_mapping()
    test_roundtrip_accuracy()
    test_varlen_offsets()
    test_fused_op_consistency()
    test_traffic_model()
    print(f"\n{'ALL PASS' if not FAILURES else f'{len(FAILURES)} FAILURES: {FAILURES}'}")
    sys.exit(1 if FAILURES else 0)
