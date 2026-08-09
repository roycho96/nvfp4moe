"""Q1 correctness: blockscaled x gather_A (linear-SF cp.async loader) on sm100.

Oracle: the same input quantized ONCE at T rows and gather-GEMM'ed must be
bit-exact with pre-permuting (gather of qdata + per-expert blocked SFA pack)
and running the existing varlen blockscaled path — tile-internal reduction
order is identical, so any divergence is a hard failure, not a tolerance.

Requires an SM100 GPU and the fork on the path:
    PYTHONPATH=third_party/quack python tests/test_q1_gather_blockscaled.py
(The Modal harness harness/q1_correctness.py runs this same logic on B200.)
"""

import itertools
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "quack"))

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def build_case(T, d, n, seqlens, seed=0):
    from quack.blockscaled.quantize import pack_scale_2d_to_blocked_contig, to_nvfp4

    torch.manual_seed(seed)
    E = len(seqlens)
    M = int(sum(seqlens))
    cu = torch.tensor([0] + list(itertools.accumulate(seqlens)), dtype=torch.int32)
    x = (torch.randn(T, d, device="cuda", dtype=torch.bfloat16) * (d**-0.5)).contiguous()
    pts = torch.tensor(1.0, device="cuda")
    qa_T_u8, sf_T, _ = to_nvfp4(x, 16, pts)
    qa_T = qa_T_u8.view(torch.float4_e2m1fn_x2)
    sfa_linear_w = sf_T.view(torch.uint8).view(torch.int32)  # (T, d/64)
    gather_idx = torch.randint(0, T, (M,), device="cuda", dtype=torch.int32)

    qa_M = qa_T_u8[gather_idx.long()].contiguous().view(torch.float4_e2m1fn_x2)
    sf_M = sf_T[gather_idx.long()].contiguous()
    rk = d // 64
    cu_l = cu.tolist()
    offs = [cu_l[b] // 128 + b for b in range(E)]
    rm_total = (cu_l[E - 1] // 128 + (E - 1)) + -(-int(seqlens[-1]) // 128)
    SFA_blocked = torch.zeros(
        1, max(rm_total, 1), rk, 32, 4, 4, device="cuda", dtype=torch.float8_e4m3fn
    )
    for b in range(E):
        lo, hi = cu_l[b], cu_l[b + 1]
        if hi > lo:
            blk = pack_scale_2d_to_blocked_contig(sf_M[lo:hi])
            SFA_blocked[0, offs[b] : offs[b] + blk.shape[1]] = blk[0]

    w = torch.randn(E, n, d, device="cuda", dtype=torch.bfloat16) * (d**-0.5)
    qb_list, sfb_list = [], []
    for e in range(E):
        qb_e, sfb_e, _ = to_nvfp4(w[e].contiguous(), 16, pts)
        qb_list.append(qb_e)
        sfb_list.append(pack_scale_2d_to_blocked_contig(sfb_e)[0])
    qb = torch.stack(qb_list).view(torch.float4_e2m1fn_x2)
    SFB = torch.stack(sfb_list)
    return dict(
        qa_T=qa_T, sfa_linear_w=sfa_linear_w, gather_idx=gather_idx,
        qa_M=qa_M, SFA_blocked=SFA_blocked, qb=qb, SFB=SFB, cu=cu.cuda(), M=M, n=n,
    )


def run_pair(case, tile_m, tile_n, cm, cn, label, tma_gather=False):
    from quack.gemm import gemm

    M, n = case["M"], case["n"]
    kw = dict(
        C=None, tile_count_semaphore=None,
        tile_M=tile_m, tile_N=tile_n, cluster_M=cm, cluster_N=cn,
        is_dynamic_persistent=True, cu_seqlens_m=case["cu"],
        SFB=case["SFB"], bs_format_a="nvfp4", bs_format_b="nvfp4",
    )
    out_base = torch.empty(M, n, device="cuda", dtype=torch.bfloat16)
    gemm(case["qa_M"], case["qb"], out_base, SFA=case["SFA_blocked"], **kw)
    out_gather = torch.empty(M, n, device="cuda", dtype=torch.bfloat16)
    gemm(case["qa_T"], case["qb"], out_gather, A_idx=case["gather_idx"],
         SFA=case["sfa_linear_w"], use_tma_gather=tma_gather, **kw)
    torch.cuda.synchronize()
    n_mis = (out_base != out_gather).sum().item()
    check(label, torch.equal(out_base, out_gather),
          f"mismatched {n_mis}/{out_base.numel()}")


def main():
    cc = torch.cuda.get_device_capability(0)
    if cc[0] != 10:
        print(f"SKIP: requires SM100, got {cc}")
        return 0
    # small, uneven experts including an empty one
    case = build_case(T=512, d=256, n=256, seqlens=[100, 200, 1, 723])
    run_pair(case, 128, 128, 1, 1, "small 128x128 cp.async")
    run_pair(case, 128, 128, 1, 1, "small 128x128 tma-gather", tma_gather=True)
    # target shape
    case = build_case(T=8192, d=2048, n=768,
                      seqlens=[3000, 1, 128, 4095, 4968, 3192, 0, 1000])
    run_pair(case, 128, 256, 1, 1, "d2048 128x256 cp.async")
    run_pair(case, 128, 256, 1, 1, "d2048 128x256 tma-gather", tma_gather=True)
    run_pair(case, 128, 256, 2, 1, "d2048 128x256 (2,1) tma-gather", tma_gather=True)
    run_pair(case, 256, 256, 2, 1, "d2048 256x256 2-CTA cp.async")
    print(f"\n{'ALL PASS' if not FAILURES else f'{len(FAILURES)} FAILURES: {FAILURES}'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
