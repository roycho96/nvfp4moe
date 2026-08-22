"""End-to-end MoEExpertLayer correctness and determinism checks for SM100.

PYTHONPATH=. python tests/test_training_gpu.py
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


def route(T, E, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(T, E, device="cuda", generator=g)
    topv, topi = torch.topk(torch.softmax(logits, -1), k, dim=-1)
    topv = topv / topv.sum(-1, keepdim=True)
    flat = topi.reshape(-1)
    order = torch.argsort(flat, stable=True)
    gi = (order // k).to(torch.int32)
    counts = torch.bincount(flat, minlength=E)
    cu = torch.zeros(E + 1, dtype=torch.int32, device="cuda")
    cu[1:] = counts.cumsum(0)
    ps = topv.reshape(-1)[order].float()
    slots = torch.empty(T * k, dtype=torch.int32, device="cuda")
    slots[order] = torch.arange(T * k, dtype=torch.int32, device="cuda")
    return gi, cu, ps, slots


def ref_layer(x, w1, w2, gi, cu, ps, T, E, topk):
    # fp32 master-weight reference (quantization error shows as the residual)
    d = x.shape[1]
    xg = x[gi.long()].float()
    cu_l = cu.tolist()
    y = torch.zeros(T, d, device="cuda")
    for e in range(E):
        lo, hi = cu_l[e], cu_l[e + 1]
        if hi == lo:
            continue
        h = xg[lo:hi] @ w1[e].float().T
        hh = torch.nn.functional.silu(h[:, 0::2]) * h[:, 1::2]
        y.index_add_(0, gi[lo:hi].long(), (hh @ w2[e].float().T) * ps[lo:hi, None])
    return y


def ref_bwd(x, w1, w2, dY, gi, cu, ps, T, E):
    d = x.shape[1]
    I2 = w1.shape[1]
    xg = x[gi.long()].float()
    cu_l = cu.tolist()
    dX = torch.zeros(T, d, device="cuda")
    dW1 = torch.zeros_like(w1, dtype=torch.float32)
    dW2 = torch.zeros_like(w2, dtype=torch.float32)
    for e in range(E):
        lo, hi = cu_l[e], cu_l[e + 1]
        if hi == lo:
            continue
        xe = xg[lo:hi]
        h = xe @ w1[e].float().T
        g, u = h[:, 0::2], h[:, 1::2]
        sg = torch.sigmoid(g)
        hh = g * sg * u
        dy2 = dY[gi[lo:hi].long()].float() * ps[lo:hi, None]
        dhh = dy2 @ w2[e].float()
        du = dhh * g * sg
        dg = dhh * u * sg * (1 + g * (1 - sg))
        dh = torch.stack([dg, du], dim=-1).reshape(hi - lo, I2)
        dX.index_add_(0, gi[lo:hi].long(), dh @ w1[e].float())
        dW2[e] = dy2.T @ hh
        dW1[e] = dh.T @ xe
    return dX, dW1, dW2


def cosrel(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (
        torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
        ((a - b).norm() / b.norm()).item(),
    )


def padded_offsets(cu):
    counts = cu[1:] - cu[:-1]
    return (((counts + 127) // 128) * 128).cumsum(0).to(torch.int32)


def check_grouped_wgrad():
    """Compare grouped wgrad with the per-expert GEMM reference."""
    from lightmoe._quantization import TensorScale
    from lightmoe.kernels.grouped.wgrad import GroupedWgrad
    from lightmoe.kernels.quantize import nvfp4_quantize_colwise

    torch.manual_seed(1)
    E = 8
    lens = [400, 0, 137, 256, 71, 0, 128, 32]  # hot 39% + two empty
    M = sum(lens)
    m, n = 256, 128
    cu = torch.zeros(E + 1, dtype=torch.int32, device="cuda")
    cu[1:] = torch.tensor(lens, device="cuda").cumsum(0)
    off_pad = ((torch.tensor(lens, device="cuda") + 127) // 128 * 128).cumsum(0).to(torch.int32)
    mp_tot = int(off_pad[-1])
    za = torch.randn(M, m, device="cuda", dtype=torch.bfloat16)
    zb = torch.randn(M, n, device="cuda", dtype=torch.bfloat16)
    sa_s, sb_s = TensorScale(), TensorScale()
    sa_s.update(za)
    sb_s.update(zb)
    qa = torch.zeros(m, mp_tot // 2, dtype=torch.uint8, device="cuda")
    qb = torch.zeros(n, mp_tot // 2, dtype=torch.uint8, device="cuda")
    sfa = torch.zeros(m * mp_tot // 16, dtype=torch.float8_e4m3fn, device="cuda")
    sfb = torch.zeros(n * mp_tot // 16, dtype=torch.float8_e4m3fn, device="cuda")
    nvfp4_quantize_colwise(za, cu, sa_s.pair, qa, sfa)
    nvfp4_quantize_colwise(zb, cu, sb_s.pair, qb, sfb)
    qa_fast = torch.zeros_like(qa)
    sfa_fast = torch.zeros_like(sfa)
    nvfp4_quantize_colwise(
        za,
        cu,
        sa_s.pair,
        qa_fast,
        sfa_fast,
        padded_offsets=off_pad,
    )
    check(
        "colwise padded-prefix lookup (bitwise)",
        torch.equal(qa_fast, qa) and torch.equal(sfa_fast, sfa),
    )

    # Columnwise oracle for the concatenated scale-factor layout.
    from benchmarks import reference as nref

    okq = oksf = True
    off_l0 = [0] + off_pad.tolist()
    for e in range(E):
        if lens[e] == 0:
            continue
        mp = -(-lens[e] // 128) * 128
        o = off_l0[e]
        seg = torch.zeros(m, mp, device="cuda")
        seg[:, : lens[e]] = za[cu[e] : cu[e + 1]].t().float()
        q_ref, sf_ref = nref.quantize_nvfp4_lastdim(seg, sa_s.pts)
        okq = okq and torch.equal(qa[:, o // 2 : (o + mp) // 2], q_ref)
        sf_blk = nref.pack_sf_blocked(sf_ref)
        got = sfa[m * o // 16 : m * (o + mp) // 16].view(m // 128, mp // 64, 32, 4, 4)
        oksf = oksf and torch.equal(got.view(torch.uint8), sf_blk.view(torch.uint8))
    check("colwise quantize oracle (concat-SF layout)", okq and oksf, f"q {okq} sf {oksf}")

    # rowwise oracle (varlen SFA contract), same routing
    from lightmoe.kernels.quantize import nvfp4_quantize_rowwise

    rm_tot = nref.varlen_sf_num_tiles(cu.cpu())
    qr = torch.zeros(M, m // 2, dtype=torch.uint8, device="cuda")
    sfr = torch.zeros(1, rm_tot, m // 64, 32, 4, 4, dtype=torch.float8_e4m3fn, device="cuda")
    nvfp4_quantize_rowwise(za, cu, sa_s.pair, qr, sfr)
    qr_fast = torch.zeros_like(qr)
    sfr_fast = torch.zeros_like(sfr)
    nvfp4_quantize_rowwise(
        za,
        cu,
        sa_s.pair,
        qr_fast,
        sfr_fast,
        padded_offsets=off_pad,
    )
    check(
        "rowwise padded-prefix lookup (bitwise)",
        torch.equal(qr_fast, qr) and torch.equal(sfr_fast, sfr),
    )
    q_ref, sf_ref = nref.quantize_nvfp4_lastdim(za.float(), sa_s.pts)
    okq = torch.equal(qr, q_ref)
    oksf = True
    offs_t = nref.varlen_sf_tile_offsets(cu.cpu())
    for e in range(E):
        lo, hi = int(cu[e]), int(cu[e + 1])
        if hi == lo:
            continue
        blk = sfr[0, offs_t[e] : offs_t[e] + nref.ceil_div(hi - lo, 128)]
        got = nref.unpack_sf_blocked(blk, hi - lo, m // 16)
        oksf = oksf and torch.equal(got.view(torch.uint8), sf_ref[lo:hi].view(torch.uint8))
    check("rowwise quantize oracle (varlen SFA)", okq and oksf, f"q {okq} sf {oksf}")

    # Dequantized per-expert reference with concatenated scale-factor slicing.
    ref = torch.zeros(E, m, n, device="cuda", dtype=torch.bfloat16)
    off_l = [0] + off_pad.tolist()
    for e in range(E):
        if lens[e] == 0:
            continue
        mp = -(-lens[e] // 128) * 128
        o = off_l[e]
        sa_e = sfa[m * o // 16 : m * (o + mp) // 16].view(m // 128, mp // 64, 32, 4, 4)
        sb_e = sfb[n * o // 16 : n * (o + mp) // 16].view(n // 128, mp // 64, 32, 4, 4)
        sa_dense = nref.unpack_sf_blocked(sa_e, m, mp // 16)
        sb_dense = nref.unpack_sf_blocked(sb_e, n, mp // 16)
        a_deq = nref.dequantize_nvfp4_lastdim(qa[:, o // 2 : (o + mp) // 2], sa_dense, sa_s.pts)
        b_deq = nref.dequantize_nvfp4_lastdim(qb[:, o // 2 : (o + mp) // 2], sb_dense, sb_s.pts)
        ref[e] = (a_deq @ b_deq.T).to(torch.bfloat16)

    wg = GroupedWgrad(m, n, E)
    out = torch.full((E, m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    gs = torch.empty(E, dtype=torch.float32, device="cuda")
    gs.copy_((sa_s.pts * sb_s.pts).expand(E))
    ones = torch.ones(E, dtype=torch.float32, device="cuda")
    wg(qa, qb, sfa, sfb, off_pad, out, gs, ones)
    c, r = cosrel(out, ref)
    check(
        "grouped wgrad vs dequantized reference",
        c > 0.99999,
        f"cos {c:.7f} rel {r:.2e} bitwise={torch.equal(out, ref)}",
    )
    check(
        "grouped wgrad empty experts zeroed",
        torch.equal(out[1], torch.zeros_like(out[1]))
        and torch.equal(out[5], torch.zeros_like(out[5])),
    )
    out2 = torch.empty_like(out)
    wg(qa, qb, sfa, sfb, off_pad, out2, gs, ones)
    check("grouped wgrad deterministic (bitwise)", torch.equal(out, out2))


def check_stochastic_rounding():
    """Check reproducibility, seed sensitivity, and unbiasedness."""
    from benchmarks import reference as nref
    from lightmoe._quantization import TensorScale
    from lightmoe.kernels.quantize import nvfp4_quantize_rowwise

    torch.manual_seed(2)
    M, F = 8192, 512  # E=1; 4.19M elements per draw
    cu = torch.tensor([0, M], dtype=torch.int32, device="cuda")
    z = torch.randn(M, F, device="cuda", dtype=torch.bfloat16)
    ts = TensorScale()
    ts.update(z)
    q = torch.zeros(M, F // 2, dtype=torch.uint8, device="cuda")
    sf = torch.zeros(1, M // 128 + 1, F // 64, 32, 4, 4, dtype=torch.float8_e4m3fn, device="cuda")

    def run(seed):
        q.zero_()
        sf.zero_()
        nvfp4_quantize_rowwise(z, cu, ts.pair, q, sf, rounding="sr", seed=seed)
        return q.clone(), sf.clone()

    qa_, _ = run(123)
    qb_, _ = run(123)
    qc_, _ = run(7)
    check("SR fixed-seed reproducible (bitwise)", torch.equal(qa_, qb_))
    check("SR seed-sensitive", not torch.equal(qa_, qc_))
    zf = z.float()
    s1, s2, n = 0.0, 0.0, 0
    for seed in (7, 999, 424242):
        qs, sfs = run(seed)
        sf2d = nref.unpack_sf_blocked(sfs[0, : M // 128], M, F // 16)
        dq = nref.dequantize_nvfp4_lastdim(qs, sf2d, ts.pts)
        err = (dq - zf).double()
        s1 += err.sum().item()
        s2 += (err * err).sum().item()
        n += err.numel()
    mean = s1 / n
    sigma3 = 3 * ((s2 / n - mean**2) / n) ** 0.5
    check(
        "SR unbiased (|E[q-x]| <= 3sig, n=12.6M)",
        abs(mean) <= sigma3,
        f"mean {mean:.3e} 3sig {sigma3:.3e}",
    )

    # linear-SF rowwise variant (the fwd x-quantize / gather-GEMM SFA
    # contract: SF row-major (rows, F/16), a row's sf bytes stride-1)
    ql = torch.zeros(M, F // 2, dtype=torch.uint8, device="cuda")
    sfl = torch.zeros(M, F // 16, dtype=torch.float8_e4m3fn, device="cuda")
    nvfp4_quantize_rowwise(z, cu, ts.pair, ql, sfl, sf_layout="linear")
    q_ref, sf_ref = nref.quantize_nvfp4_lastdim(zf, ts.pts)
    check(
        "linear-SF rowwise oracle (q + sf bitwise)",
        torch.equal(ql, q_ref) and torch.equal(sfl.view(torch.uint8), sf_ref.view(torch.uint8)),
    )


def check_rht():
    """Check RHT quantization, gather behavior, and determinism."""
    from benchmarks import reference as nref
    from lightmoe._quantization import TensorScale
    from lightmoe.kernels.quantize import nvfp4_quantize_colwise, nvfp4_rht_amax, rht_matrix

    torch.manual_seed(3)
    E = 4
    lens = [300, 0, 137, 253]  # non-16-multiple tails + an empty expert
    M = sum(lens)
    F = 256
    cu = torch.zeros(E + 1, dtype=torch.int32, device="cuda")
    cu[1:] = torch.tensor(lens, device="cuda").cumsum(0)
    z = torch.randn(M, F, device="cuda", dtype=torch.bfloat16) * 2

    check(
        "rht matrix == reference bits",
        torch.equal(rht_matrix().view(torch.uint16), nref.rht_matrix_ref().view(torch.uint16)),
    )

    def ref_segs(src, gidx=None):
        srcg = src if gidx is None else src[gidx.long()]
        segs, amax = [], torch.zeros((), device="cuda")
        for e in range(E):
            lo, hi = int(cu[e]), int(cu[e + 1])
            if hi == lo:
                segs.append(None)
                continue
            mp = -(-(hi - lo) // 128) * 128
            seg = torch.zeros(F, mp, device="cuda", dtype=torch.bfloat16)
            seg[:, : hi - lo] = srcg[lo:hi].t()
            t = nref.rht_transform_ref(seg.reshape(F, mp))
            segs.append(t)
            amax = torch.maximum(amax, t.float().abs().amax())
        return segs, amax

    # fused amax pre-pass: raw row amax + post-RHT col amax, both bitwise
    n_tiles = -(-M // 128) + E
    part = torch.zeros(n_tiles * (F // 128), 2, device="cuda")
    nvfp4_rht_amax(z, cu, part)
    red = part.amax(0)
    segs, col_ref = ref_segs(z)
    check("rht_amax row bits", torch.equal(red[0], z.float().abs().amax()))
    check("rht_amax col bits (post-RHT)", torch.equal(red[1], col_ref))

    # colwise rht=True oracle: bitwise q + sf vs transform + TE-scale-math ref
    off_pad = ((torch.tensor(lens) + 127) // 128 * 128).cumsum(0)
    mp_tot = int(off_pad[-1])
    ts = TensorScale(te_rht=True)
    ts.set_amax(red[1:2])
    q = torch.zeros(F, mp_tot // 2, dtype=torch.uint8, device="cuda")
    sf = torch.zeros(F * mp_tot // 16, dtype=torch.float8_e4m3fn, device="cuda")
    nvfp4_quantize_colwise(z, cu, ts.pair, q, sf, rht=True)
    off_l = [0] + off_pad.tolist()
    okq = oksf = True
    for e in range(E):
        if lens[e] == 0:
            continue
        mp = -(-lens[e] // 128) * 128
        o = off_l[e]
        q_ref, sf_ref, _ = nref.quantize_nvfp4_lastdim_te(segs[e], red[1])
        okq = okq and torch.equal(q[:, o // 2 : (o + mp) // 2], q_ref)
        got = sf[F * o // 16 : F * (o + mp) // 16].view(F // 128, mp // 64, 32, 4, 4)
        oksf = oksf and torch.equal(
            got.view(torch.uint8), nref.pack_sf_blocked(sf_ref).view(torch.uint8)
        )
    check("colwise rht oracle (q + sf bitwise)", okq and oksf, f"q {okq} sf {oksf}")
    q2 = torch.zeros_like(q)
    sf2 = torch.zeros_like(sf)
    nvfp4_quantize_colwise(z, cu, ts.pair, q2, sf2, rht=True)
    check(
        "colwise rht deterministic (bitwise)",
        torch.equal(q, q2) and torch.equal(sf.view(torch.uint8), sf2.view(torch.uint8)),
    )

    # gather variant (the X wgrad operand path)
    T = 500
    zs = torch.randn(T, F, device="cuda", dtype=torch.bfloat16)
    gi = torch.randint(0, T, (M,), device="cuda", dtype=torch.int32)
    part.zero_()
    nvfp4_rht_amax(zs, cu, part, gather_idx=gi)
    redg = part.amax(0)
    segs_g, col_ref_g = ref_segs(zs, gi)
    check("gather rht_amax col bits", torch.equal(redg[1], col_ref_g))
    tsg = TensorScale(te_rht=True)
    tsg.set_amax(redg[1:2])
    qg = torch.zeros_like(q)
    sfg = torch.zeros_like(sf)
    nvfp4_quantize_colwise(zs, cu, tsg.pair, qg, sfg, gather_idx=gi, rht=True)
    okq = True
    for e in range(E):
        if lens[e] == 0:
            continue
        mp = -(-lens[e] // 128) * 128
        o = off_l[e]
        q_ref, _, _ = nref.quantize_nvfp4_lastdim_te(segs_g[e], redg[1])
        okq = okq and torch.equal(qg[:, o // 2 : (o + mp) // 2], q_ref)
    check("gather colwise rht oracle (q bitwise)", okq)

    # SR + RHT: fixed seed reproducible, seed-sensitive
    qs1 = torch.zeros_like(q)
    ss1 = torch.zeros_like(sf)
    nvfp4_quantize_colwise(z, cu, ts.pair, qs1, ss1, rounding="sr", seed=11, rht=True)
    qs2 = torch.zeros_like(q)
    ss2 = torch.zeros_like(sf)
    nvfp4_quantize_colwise(z, cu, ts.pair, qs2, ss2, rounding="sr", seed=11, rht=True)
    qs3 = torch.zeros_like(q)
    ss3 = torch.zeros_like(sf)
    nvfp4_quantize_colwise(z, cu, ts.pair, qs3, ss3, rounding="sr", seed=12, rht=True)
    check("SR+rht fixed-seed reproducible (bitwise)", torch.equal(qs1, qs2))
    check("SR+rht seed-sensitive", not torch.equal(qs1, qs3))


def check_delayed_amax():
    """Check warmup, scale lag, gradient behavior, and determinism."""
    from lightmoe import MoEExpertLayer

    print("delayed_col_amax:", flush=True)
    torch.manual_seed(11)
    T, d, I, E, k = 2048, 2048, 768, 8, 8
    base = MoEExpertLayer(d, I, E, k).cuda()
    base.refresh_weights()
    x0 = torch.randn(T, d, device="cuda", dtype=torch.bfloat16) * d**-0.5
    dY0 = torch.randn(T, d, device="cuda", dtype=torch.bfloat16)
    gi, cu, ps, slots = route(T, E, k, 3)
    off_pad = padded_offsets(cu)
    base.calibrate(x0, gi, cu, ps)

    def clone_arm(**kw):
        m = MoEExpertLayer(d, I, E, k, **kw).cuda()
        with torch.no_grad():
            m.w1.copy_(base.w1)
            m.w2.copy_(base.w2)
        m.refresh_weights()
        m.calibrate(x0, gi, cu, ps)
        return m

    def steps(m, datas):
        outs = []
        for xs, dys in datas:
            xs = xs.clone().requires_grad_(True)
            m.w1.grad = None
            m.w2.grad = None
            y = m(xs, gi, cu, ps, slots, off_pad=off_pad)
            y.backward(dys)
            outs.append((y.detach().clone(), xs.grad.clone(), m.w1.grad.clone(), m.w2.grad.clone()))
        return outs

    fixed = [(x0, dY0)] * 3
    # varying: scale modulation exercises a real amax lag between steps
    varying = [(x0 * s, dY0 * s) for s in (1.0, 1.3, 0.8)]

    o_v0 = steps(clone_arm(), fixed)
    o_dc = steps(clone_arm(delayed_col_amax=True), fixed)
    for i in range(3):
        check(
            f"delayed fixed-data step{i + 1} bitwise == default",
            all(torch.equal(a, b) for a, b in zip(o_dc[i], o_v0[i])),
        )

    o_v0v = steps(clone_arm(), varying)
    dc_arm = clone_arm(delayed_col_amax=True)
    o_dcv = steps(dc_arm, varying)
    for i in range(3):
        y_ok = torch.equal(o_dcv[i][0], o_v0v[i][0])
        dx_ok = torch.equal(o_dcv[i][1], o_v0v[i][1])
        c1 = cosrel(o_dcv[i][2], o_v0v[i][2])[0]
        c2 = cosrel(o_dcv[i][3], o_v0v[i][3])[0]
        check(
            f"delayed varying step{i + 1}: y+dX bitwise == default, dW close",
            y_ok and dx_ok and c1 > 0.99 and c2 > 0.99,
            f"dW1 cos {c1:.6f} dW2 cos {c2:.6f}",
        )
    # step 1 is warmup (same-step seeding): fully bitwise incl dW
    check(
        "delayed warmup step1 dW bitwise == default",
        torch.equal(o_dcv[0][2], o_v0v[0][2]) and torch.equal(o_dcv[0][3], o_v0v[0][3]),
    )
    # lag must actually engage from step 2 (scales moved 1.0 -> 1.3)
    check(
        "delayed lag engages (step2 dW differs from default)",
        not torch.equal(o_dcv[1][2], o_v0v[1][2]),
    )

    o_dcv2 = steps(clone_arm(delayed_col_amax=True), varying)
    check(
        "delayed determinism (sequence twice, all bitwise)",
        all(torch.equal(a, b) for oo, o2 in zip(o_dcv, o_dcv2) for a, b in zip(oo, o2)),
    )

    # SR grad casts under delayed: fixed seed keeps sequence determinism
    m_sr = clone_arm(delayed_col_amax=True)
    m_sr.sr_seed = 77
    o_sr1 = steps(m_sr, varying)
    m_sr2 = clone_arm(delayed_col_amax=True)
    m_sr2.sr_seed = 77
    o_sr2 = steps(m_sr2, varying)
    check(
        "delayed+SR determinism (fixed seed, sequence bitwise)",
        all(torch.equal(a, b) for oo, o2 in zip(o_sr1, o_sr2) for a, b in zip(oo, o2)),
    )


def check_dispatch():
    """Compare fused dispatch with stable PyTorch routing."""
    from lightmoe import MoEDispatch

    def make_route(T, E, k, dist, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        logits = torch.randn(T, E, device="cuda", generator=g)
        if dist == "hot":
            logits[:, seed % E] += 4.0
        elif dist == "sparse":
            logits[:, : E // 2] -= 6.0
        topv, topi = torch.topk(torch.softmax(logits, -1), k, dim=-1)
        topv = (topv / topv.sum(-1, keepdim=True)).float().contiguous()
        return topi.to(torch.int32).contiguous(), topv

    def ref_dispatch(topi, topv, E, k):
        flat = topi.reshape(-1).long()
        order = torch.argsort(flat, stable=True)
        gi = (order // k).to(torch.int32)
        counts = torch.bincount(flat, minlength=E)
        cu = torch.zeros(E + 1, dtype=torch.int32, device="cuda")
        cu[1:] = counts.cumsum(0).to(torch.int32)
        ps = topv.reshape(-1)[order].float()
        slots = torch.empty(flat.numel(), dtype=torch.int32, device="cuda")
        slots[order] = torch.arange(flat.numel(), dtype=torch.int32, device="cuda")
        return gi, cu, ps, slots

    for T, E, k in ((2048, 8, 8), (1024, 32, 8), (512, 64, 8), (4096, 8, 2), (8192, 32, 8)):
        disp = MoEDispatch(T, E, k)
        ok = det = imm = okop = diffok = True
        for dist in ("uniform", "hot", "sparse"):
            for seed in (0, 1, 2):
                topi, topv = make_route(T, E, k, dist, seed)
                ti0, tv0 = topi.clone(), topv.clone()
                r = ref_dispatch(topi, topv, E, k)
                o1 = [t.clone() for t in disp(topi, topv)]
                op1 = disp.off_pad.clone()
                o2 = disp(topi, topv)
                ok = ok and all(torch.equal(a, b) for a, b in zip(o1, r))
                det = det and all(torch.equal(a, b) for a, b in zip(o1, o2))
                det = det and torch.equal(op1, disp.off_pad)
                imm = imm and torch.equal(topi, ti0) and torch.equal(topv, tv0)
                seg = r[1][1:] - r[1][:-1]
                op_ref = (((seg + 127) // 128) * 128).cumsum(0).to(torch.int32)
                okop = okop and torch.equal(op1, op_ref)
                topv_grad = topv.detach().clone().requires_grad_(True)
                _, _, ps_raw, _ = disp(topi, topv_grad.detach())
                ps_diff = disp.differentiable_probs(topv_grad)
                ps_diff.sum().backward()
                diffok = diffok and torch.equal(ps_diff, ps_raw)
                diffok = diffok and torch.equal(topv_grad.grad, torch.ones_like(topv_grad))
        check(f"dispatch bitwise vs torch chain (T{T} E{E} k{k})", ok)
        check(f"dispatch off_pad bitwise (T{T} E{E} k{k})", okop)
        check(f"dispatch differentiable probs (T{T} E{E} k{k})", diffok)
        check(
            f"dispatch deterministic + router untouched (T{T} E{E} k{k})",
            det and imm,
            f"det {det} imm {imm}",
        )


def main():
    if torch.cuda.get_device_capability(0)[0] != 10:
        print("SKIP: requires SM100")
        return 0
    from lightmoe import MoEExpertLayer
    from lightmoe.kernels.routing.combine import moe_finalize, moe_finalize_bwd

    check_dispatch()
    check_grouped_wgrad()
    check_stochastic_rounding()
    check_rht()
    check_delayed_amax()

    torch.manual_seed(0)
    T, d, I, E, k = 2048, 2048, 768, 8, 8
    layer = MoEExpertLayer(d, I, E, k).cuda()
    layer.refresh_weights()
    x = (torch.randn(T, d, device="cuda", dtype=torch.bfloat16) * d**-0.5).requires_grad_(True)
    gi, cu, ps, slots = route(T, E, k, 0)
    op_ref = padded_offsets(cu)
    layer.calibrate(x.detach(), gi, cu, ps)

    # finalize-bwd kernel oracle (bitwise: same op sequence as torch)
    dYt = torch.randn(T, d, device="cuda", dtype=torch.bfloat16)
    out_k = torch.empty(T * k, d, device="cuda", dtype=torch.bfloat16)
    moe_finalize_bwd(dYt, gi, ps.contiguous(), out_k)
    ref_k = (dYt[gi.long()].float() * ps[:, None]).to(torch.bfloat16)
    check("finalize_bwd bitwise vs torch chain", torch.equal(out_k, ref_k))

    # Weighted combine used by the down-projection path.
    yw_raw = torch.randn(T * k, d, device="cuda", dtype=torch.bfloat16)
    out_w = torch.empty(T, d, device="cuda", dtype=torch.bfloat16)
    moe_finalize(yw_raw, slots, out_w, k, tile_t=4, n_frag=2, weights=ps)
    slots_2d = slots.view(T, k).long()
    ref_w = torch.zeros(T, d, device="cuda")
    for j in range(k):
        sj = slots_2d[:, j]
        ref_w += yw_raw[sj].float() * ps[sj, None]
    out_w2 = torch.empty_like(out_w)
    moe_finalize(yw_raw, slots, out_w2, k, tile_t=4, n_frag=2, weights=ps)
    c, r = cosrel(out_w, ref_w)
    check(
        "weighted finalize closure + determinism",
        c > 0.99999 and r < 0.005 and torch.equal(out_w, out_w2),
        f"cos {c:.7f} rel {r:.2e}",
    )

    # fwd closure + determinism
    y = layer(x, gi, cu, ps, slots, off_pad=op_ref)
    y_ref = ref_layer(x.detach(), layer.w1, layer.w2, gi, cu, ps, T, E, k)
    c, r = cosrel(y, y_ref)
    # Each quantized operand contributes to the expected FP4 error band.
    check("fwd closure vs fp32 master ref", c > 0.95, f"cos {c:.5f} rel {r:.4f} (fp4 band)")
    det = all(torch.equal(y, layer(x, gi, cu, ps, slots, off_pad=op_ref)) for _ in range(5))
    check("fwd deterministic 5x", det)

    # bwd closure + determinism
    dY = torch.randn(T, d, device="cuda", dtype=torch.bfloat16)
    y = layer(x, gi, cu, ps, slots, off_pad=op_ref)
    y.backward(dY)
    gx1, gw1a, gw2a = x.grad.clone(), layer.w1.grad.clone(), layer.w2.grad.clone()
    dXr, dW1r, dW2r = ref_bwd(x.detach(), layer.w1, layer.w2, dY, gi, cu, ps, T, E)
    for n, a, b, cmin in (
        ("dX", gx1, dXr, 0.94),
        ("dW1", gw1a, dW1r, 0.94),
        ("dW2", gw2a, dW2r, 0.94),
    ):
        c, r = cosrel(a, b)
        check(f"bwd {n} closure", c > cmin, f"cos {c:.5f} rel {r:.4f}")
    x.grad = None
    layer.w1.grad = None
    layer.w2.grad = None
    y = layer(x, gi, cu, ps, slots, off_pad=op_ref)
    y.backward(dY)
    check(
        "bwd deterministic (grads bitwise)",
        torch.equal(x.grad, gx1)
        and torch.equal(layer.w1.grad, gw1a)
        and torch.equal(layer.w2.grad, gw2a),
    )

    # SR on grad casts (TE fp4_quant_bwd_grad analogue): closure holds and
    # a fixed seed keeps the library-level bitwise determinism guarantee
    layer.sr_seed = 42
    x.grad = None
    layer.w1.grad = None
    layer.w2.grad = None
    y = layer(x, gi, cu, ps, slots, off_pad=op_ref)
    y.backward(dY)
    g1 = (x.grad.clone(), layer.w1.grad.clone(), layer.w2.grad.clone())
    c, _ = cosrel(g1[0], dXr)
    check("SR bwd dX closure", c > 0.94, f"cos {c:.5f}")
    x.grad = None
    layer.w1.grad = None
    layer.w2.grad = None
    y = layer(x, gi, cu, ps, slots, off_pad=op_ref)
    y.backward(dY)
    check(
        "SR bwd deterministic (fixed seed, grads bitwise)",
        torch.equal(x.grad, g1[0])
        and torch.equal(layer.w1.grad, g1[1])
        and torch.equal(layer.w2.grad, g1[2]),
    )
    layer.sr_seed = None

    # Routed training requires the dispatch-emitted padded wgrad offsets.
    x.grad = None
    layer.w1.grad = None
    layer.w2.grad = None
    y_fg = layer(x, gi, cu, ps, slots, off_pad=op_ref)
    y_fg.backward(dY)
    x.grad = None
    layer.w1.grad = None
    layer.w2.grad = None
    try:
        layer(x, gi, cu, ps, slots, off_pad=None)
        rejected_missing_offsets = False
    except ValueError:
        rejected_missing_offsets = True
    check("missing off_pad is rejected", rejected_missing_offsets)

    # Reuse buffers grown by a large assignment count with a smaller one.
    T_small = 257
    x_small = (
        torch.randn(T_small, d, device="cuda", dtype=torch.bfloat16) * d**-0.5
    ).requires_grad_(True)
    gi_s, cu_s, ps_s, slots_s = route(T_small, E, k, 17)
    off_s = padded_offsets(cu_s)
    dy_s = torch.randn(T_small, d, device="cuda", dtype=torch.bfloat16)
    layer.w1.grad = None
    layer.w2.grad = None
    y_s = layer(x_small, gi_s, cu_s, ps_s, slots_s, off_pad=off_s)
    y_s.backward(dy_s)
    small_grads = (x_small.grad, layer.w1.grad, layer.w2.grad)
    check(
        "buffers support a decreasing assignment count",
        y_s.shape == (T_small, d)
        and torch.isfinite(y_s).all()
        and all(g is not None and torch.isfinite(g).all() for g in small_grads),
    )

    print(f"\n{'ALL PASS' if not FAILURES else f'{len(FAILURES)} FAILURES: {FAILURES}'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
