"""Q2 correctness: gated-postact NVFP4 quant epilogue x varlen_m on sm100.

Oracle: plain GEMM with fp32 D gives the exact accumulator (same mainloop /
k order as the mod kernels); reglu = relu(gate)*up is single-rounded fp32
arithmetic identical to the epilogue's, so ref-quantizing it must match
gated_quant_mod("reglu") bit-exactly (qdata AND SF bytes). Zero-amax blocks:
the epilogue stores SF = 0x00 where the standalone quantizer clamps to
E4M3_EPS (0x01); both dequantize to exactly zero (encoded in the reference).

Requires SM100 + the fork on the path:
    PYTHONPATH=third_party/quack python tests/test_q2_gated_quant_varlen.py
(The Modal harness harness/q2_correctness.py runs the full matrix on B200.)
"""

import itertools
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "quack"))
sys.path.insert(0, str(ROOT))

from kernels import nvfp4_ref as ref  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def run_case(seqlens, d, I, blockscaled=False, gather=None, T=None, label=""):
    from quack.blockscaled.quantize import pack_scale_2d_to_blocked_contig, to_nvfp4
    from quack.epilogue.library import gated_quant_mod
    from quack.gemm import gemm as gemm_plain

    torch.manual_seed(0)
    N = 2 * I
    E = len(seqlens)
    M = int(sum(seqlens))
    cu = torch.tensor([0] + list(itertools.accumulate(seqlens)), dtype=torch.int32).cuda()
    kw = dict(tile_M=128, tile_N=256, cluster_M=1, cluster_N=1, cu_seqlens_m=cu)
    gkw = dict(cu_seqlens_m=cu)
    if not blockscaled:
        A = torch.randn(M, d, device="cuda", dtype=torch.bfloat16) / d**0.25
        B = torch.randn(E, N, d, device="cuda", dtype=torch.bfloat16) / d**0.25
        sfk = {}
    else:
        pts1 = torch.tensor(1.0, device="cuda")
        rows = T if gather else M
        x = (torch.randn(rows, d, device="cuda", dtype=torch.bfloat16) * d**-0.5).contiguous()
        qa_u8, sf_a, _ = to_nvfp4(x, 16, pts1)
        w = torch.randn(E, N, d, device="cuda", dtype=torch.bfloat16) * d**-0.5
        qb_l, sfb_l = [], []
        for e in range(E):
            qb_e, sfb_e, _ = to_nvfp4(w[e].contiguous(), 16, pts1)
            qb_l.append(qb_e)
            sfb_l.append(pack_scale_2d_to_blocked_contig(sfb_e)[0])
        B = torch.stack(qb_l).view(torch.float4_e2m1fn_x2)
        A = qa_u8.view(torch.float4_e2m1fn_x2)
        sfk = dict(SFB=torch.stack(sfb_l), bs_format_a="nvfp4", bs_format_b="nvfp4")
        if gather:
            idx = torch.randint(0, T, (M,), device="cuda", dtype=torch.int32)
            sfk["SFA"] = sf_a.view(torch.uint8).view(torch.int32)
            kw.update(A_idx=idx, use_tma_gather=gather == "tma")
            gkw.update(A_idx=idx, use_tma_gather=gather == "tma")
        else:
            cu_l = cu.tolist()
            offs = ref.varlen_sf_tile_offsets(cu.cpu())
            SFA = torch.zeros(1, ref.varlen_sf_num_tiles(cu.cpu()), d // 64, 32, 4, 4,
                              device="cuda", dtype=torch.float8_e4m3fn)
            for b in range(E):
                lo, hi = cu_l[b], cu_l[b + 1]
                if hi > lo:
                    blk = pack_scale_2d_to_blocked_contig(sf_a[lo:hi])
                    SFA[0, offs[b] : offs[b] + blk.shape[1]] = blk[0]
            sfk["SFA"] = SFA

    D_f32 = torch.empty(M, N, device="cuda", dtype=torch.float32)
    gemm_plain(A, B, D_f32, None, None, tile_M=128, tile_N=256, cluster_M=1,
               cluster_N=1, **gkw, **sfk)
    torch.cuda.synchronize()
    post = torch.relu(D_f32[:, 0::2]) * D_f32[:, 1::2]
    pts = (post.abs().amax().float() / (448.0 * 6.0)).cpu()

    q_out = torch.empty(M, I // 2, device="cuda", dtype=torch.float4_e2m1fn_x2)
    rm_total = ref.varlen_sf_num_tiles(cu.cpu())
    sf_out = torch.zeros(1, rm_total, I // 64, 32, 4, 4, device="cuda",
                         dtype=torch.float8_e4m3fn)
    gated_quant_mod("reglu").gemm(
        A, B, None,
        epi_args=dict(postact=q_out, postact_sf=sf_out,
                      sfd_norm_const=float(1.0 / pts.item())),
        **kw, **sfk,
    )
    torch.cuda.synchronize()

    q_ref, sf_ref = ref.quantize_nvfp4_lastdim(post, pts.cuda())
    zero = post.reshape(M, I // 16, 16).abs().amax(-1) == 0
    sf_ref = torch.where(zero, torch.zeros_like(sf_ref.view(torch.uint8)),
                         sf_ref.view(torch.uint8)).view(torch.float8_e4m3fn)
    check(f"{label} qdata", torch.equal(q_out.view(torch.uint8), q_ref))
    cu_l = cu.tolist()
    offs = ref.varlen_sf_tile_offsets(cu.cpu())
    ok = True
    for b in range(E):
        lo, hi = cu_l[b], cu_l[b + 1]
        if hi == lo:
            continue
        blk = sf_out[0, offs[b] : offs[b] + ref.ceil_div(hi - lo, 128)]
        got = ref.unpack_sf_blocked(blk, hi - lo, I // 16)
        ok = ok and torch.equal(got.view(torch.uint8), sf_ref[lo:hi].view(torch.uint8))
    check(f"{label} SF", ok)


def main():
    if torch.cuda.get_device_capability(0)[0] != 10:
        print("SKIP: requires SM100")
        return 0
    lens = [3000, 1, 128, 4095, 4968, 3192, 0, 1000]
    run_case([100, 200, 1, 723], 256, 512, label="varlen bf16")
    run_case(lens, 2048, 768, blockscaled=True, label="varlen nvfp4-in")
    run_case(lens, 2048, 768, blockscaled=True, gather="tma", T=8192,
             label="fullfuse tma-gather")
    run_case(lens, 2048, 768, blockscaled=True, gather="cpasync", T=8192,
             label="fullfuse cp.async-gather")
    print(f"\n{'ALL PASS' if not FAILURES else f'{len(FAILURES)} FAILURES: {FAILURES}'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
