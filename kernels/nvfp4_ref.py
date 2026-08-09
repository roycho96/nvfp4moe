"""Pure-torch reference for the P1 fused gather + dual-quantize kernel.

This is a slow, layout-exact reference implementation. Every fast
implementation (Triton, then CuTe DSL) is validated bit-exact against it, and
this file is in turn cross-checked against quack's `to_nvfp4` /
`pack_scale_2d_to_blocked_contig` in tests/test_ref_vs_quack.py.

Layout contracts (verified against quack HEAD, 2026-08-07):

  qdata     E2M1, two values per byte, packed along the *contraction* axis.
            Low nibble is the even index. Ragged over experts with no padding
            (quack's varlen manager offsets by cu_seqlens directly).

  SF        E4M3, one per 16 contraction elements, stored in the 128x4 blocked
            atom shared by quack / CUTLASS / cuBLAS:
                (l, rm, rk, 32, 4, 4),  rm = ceil(mn/128), rk = ceil(sf_k/4)
            Within a 128-row tile, logical row m_local lands at
                atom[m_local % 32][m_local // 32][k_local % 4]
            (from quack.blockscaled.quantize.pack_scale_2d_to_blocked_contig:
             view(l,rm,128,rk,4) -> permute -> reshape(l,rm,rk,4,32,4)
             -> transpose(3,4)).

  varlen SF Expert b's region starts at tile index `cu_seqlens[b] // 128 + b`
            in the rm mode -- quack VarlenManager.offset_batch_SFA. Tiles are
            NOT tightly packed; the "+ b" is slack that keeps each expert
            128-aligned so an SF atom never straddles an expert boundary.

  dequant   value = qdata_e2m1 * sf_e4m3 * per_tensor_scale
"""

import torch

F4_E2M1_MAX = 6.0
F8E4M3_MAX = 448.0
E4M3_EPS = 2.0**-9

# E2M1 code -> value, index is the 4-bit code (sign in bit 3).
_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# --------------------------------------------------------------------------
# scalar conversions
# --------------------------------------------------------------------------


def per_tensor_scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    """NVFP4 second-level scale: amax / (448 * 6). Matches quack + TE."""
    return amax.to(torch.float32) / (F8E4M3_MAX * F4_E2M1_MAX)


def _f32_to_e2m1_code(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even into E2M1, returning the 4-bit code as uint8.

    Done by exhaustive comparison against the 8 magnitudes rather than bit
    surgery: clarity is preferred here. Ties go to the even code,
    matching hardware cvt.rn.satfinite.e2m1x2.
    """
    sign = (x < 0) | ((x == 0) & torch.signbit(x))
    mag = x.abs().float().clamp(max=F4_E2M1_MAX)
    levels = _E2M1_VALUES[:8].to(x.device)  # 0, .5, 1, 1.5, 2, 3, 4, 6
    # index of the interval, then pick the nearer endpoint with ties-to-even
    idx = torch.bucketize(mag, levels, right=False).clamp(1, 7)
    lo, hi = levels[idx - 1], levels[idx]
    mid = (lo + hi) / 2
    take_hi = (mag > mid) | ((mag == mid) & (idx % 2 == 0))
    code = torch.where(take_hi, idx, idx - 1).to(torch.uint8)
    code = torch.where(mag <= 0, torch.zeros_like(code), code)
    return code | (sign.to(torch.uint8) << 3)


def _pack_nibbles(code: torch.Tensor) -> torch.Tensor:
    """(..., K) uint8 codes -> (..., K//2) bytes, even index in the low nibble."""
    assert code.shape[-1] % 2 == 0
    even = code[..., 0::2]
    odd = code[..., 1::2]
    return (even | (odd << 4)).contiguous()


def _unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    out = torch.stack([lo, hi], dim=-1)
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


# --------------------------------------------------------------------------
# quantize along the last (contraction) axis
# --------------------------------------------------------------------------


def quantize_nvfp4_lastdim(x: torch.Tensor, per_tensor_scale: torch.Tensor):
    """Quantize a (..., K) tensor along K in groups of 16.

    Returns (qdata (..., K//2) uint8, sf (..., K//16) float8_e4m3fn).
    Mirrors quack.blockscaled.quantize.to_nvfp4 exactly.
    """
    assert x.shape[-1] % 16 == 0, f"K={x.shape[-1]} must be a multiple of 16"
    shape = x.shape
    g = x.float().reshape(*shape[:-1], shape[-1] // 16, 16)

    block_amax = g.abs().amax(dim=-1)
    block_scale = block_amax / F4_E2M1_MAX
    scaled = block_scale / per_tensor_scale.to(torch.float32)
    sf = scaled.clamp(min=E4M3_EPS, max=F8E4M3_MAX).to(torch.float8_e4m3fn)

    recip = (1.0 / per_tensor_scale.to(torch.float32)) / sf.float()
    q = (g * recip.unsqueeze(-1)).clamp(-F4_E2M1_MAX, F4_E2M1_MAX)

    code = _f32_to_e2m1_code(q.reshape(shape))
    return _pack_nibbles(code), sf


_rht_m_ref_cache = {}


def rht_matrix_ref(device="cuda") -> torch.Tensor:
    """(16, 16) bf16 S @ H16 * 0.25 - TE get_rht_matrix
    (with_random_sign_mask=True) reproduced: Sylvester H16, TE's hard-coded
    wgrad sign vector, scale 1/sqrt(16). Entries are +-0.25, exact in bf16."""
    dev = torch.device(device)
    t = _rht_m_ref_cache.get(dev)
    if t is None:
        s = torch.tensor(
            [1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1],
            dtype=torch.float32, device=dev)
        h = torch.ones(1, 1, device=dev)
        while h.shape[0] < 16:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        t = ((s * torch.eye(16, device=dev)) @ (h * 0.25)
             ).to(torch.bfloat16).contiguous()
        _rht_m_ref_cache[dev] = t
    return t


def rht_transform_ref(x: torch.Tensor) -> torch.Tensor:
    """TE randomized Hadamard transform along the LAST axis in 16-groups:
    y = x_g16 @ (S @ H16 * 0.25), bf16 in/out via torch's bf16 matmul (fp32
    accumulate, one k=16 HMMA per output - measured bitwise equal to the
    kernel's mma.sync path over 16.4M elements / 5 distributions on sm_120).
    In the colwise quantizer this axis is the (zero-padded, expert-local)
    token axis."""
    assert x.dtype == torch.bfloat16 and x.shape[-1] % 16 == 0
    shp = x.shape
    return (x.reshape(-1, 16) @ rht_matrix_ref(x.device)).reshape(shp)


def quantize_nvfp4_lastdim_te(x: torch.Tensor, global_amax: torch.Tensor):
    """TE NVFP4 scale-math variant (the rht=True colwise contract).

    Mirrors TE's kernel default path op for op in fp32 (multiply chain, NO
    lower SF clamp - a block whose amax is below ~2^-18 of the global amax
    gets sf == 0 and dequantizes to zero, TE's own behavior):
        ges  = amax > 0 ? min(2688/amax, f32max) : 1.0
        sf   = e4m3(min(vec_max * (ges * fp32(1/6)), 448))
        q    = e2m1_rn(clamp(x * min(1/(sf * (1/ges)), f32max), +-6))
    Returns (qdata (..., K//2) uint8, sf (..., K//16) float8_e4m3fn, gds (1,)
    f32 - the dequant global scale: value = q * sf * gds)."""
    assert x.shape[-1] % 16 == 0
    f32max = torch.finfo(torch.float32).max
    amax = global_amax.to(torch.float32).reshape(1)
    ges = torch.where(amax > 0,
                      torch.clamp(F8E4M3_MAX * F4_E2M1_MAX / amax, max=f32max),
                      torch.ones_like(amax))
    inv6 = torch.tensor(1.0, dtype=torch.float32, device=x.device) \
        / torch.tensor(6.0, dtype=torch.float32, device=x.device)
    mult = ges * inv6
    gds = 1.0 / ges
    shape = x.shape
    g = x.float().reshape(*shape[:-1], shape[-1] // 16, 16)
    vec_max = g.abs().amax(dim=-1)
    sf = (vec_max * mult).clamp(max=F8E4M3_MAX).to(torch.float8_e4m3fn)
    recip = torch.clamp(1.0 / (sf.float() * gds), max=f32max)
    q = (g * recip.unsqueeze(-1)).clamp(-F4_E2M1_MAX, F4_E2M1_MAX)
    code = _f32_to_e2m1_code(q.reshape(shape))
    return _pack_nibbles(code), sf, gds


def dequantize_nvfp4_lastdim(qdata, sf, per_tensor_scale) -> torch.Tensor:
    """Inverse of quantize_nvfp4_lastdim for reference comparisons."""
    code = _unpack_nibbles(qdata).long()
    vals = _E2M1_VALUES.to(qdata.device)[code]
    k = vals.shape[-1]
    s = sf.float().repeat_interleave(16, dim=-1)[..., :k]
    return vals * s * per_tensor_scale.to(torch.float32)


# --------------------------------------------------------------------------
# SF blocked layout
# --------------------------------------------------------------------------


def pack_sf_blocked(sf_2d: torch.Tensor, rm: int | None = None, rk: int | None = None):
    """(mn, sf_k) e4m3 -> (rm, rk, 32, 4, 4) blocked atoms, zero-padded.

    rm/rk may be given larger than the minimum to place this block inside a
    bigger (varlen) buffer.
    """
    mn, sf_k = sf_2d.shape
    rm = rm if rm is not None else ceil_div(mn, 128)
    rk = rk if rk is not None else ceil_div(sf_k, 4)
    u8 = sf_2d.contiguous().view(torch.uint8)
    padded = u8.new_zeros(rm * 128, rk * 4)
    padded[:mn, :sf_k] = u8
    b = padded.view(rm, 128, rk, 4).permute(0, 2, 1, 3)          # (rm, rk, 128, 4)
    b = b.reshape(rm, rk, 4, 32, 4).transpose(2, 3).contiguous()  # (rm, rk, 32, 4, 4)
    return b.view(torch.float8_e4m3fn)


def unpack_sf_blocked(blocked: torch.Tensor, mn: int, sf_k: int) -> torch.Tensor:
    rm, rk = blocked.shape[:2]
    assert tuple(blocked.shape[2:]) == (32, 4, 4)
    u8 = blocked.view(torch.uint8).transpose(2, 3).reshape(rm, rk, 128, 4)
    u8 = u8.permute(0, 2, 1, 3).reshape(rm * 128, rk * 4)
    return u8[:mn, :sf_k].contiguous().view(torch.float8_e4m3fn)


def varlen_sf_tile_offsets(cu_seqlens: torch.Tensor) -> list[int]:
    """quack VarlenManager.offset_batch_SFA: expert b starts at tile
    cu_seqlens[b] // 128 + b."""
    cu = cu_seqlens.tolist()
    return [cu[b] // 128 + b for b in range(len(cu) - 1)]


def varlen_sf_num_tiles(cu_seqlens: torch.Tensor) -> int:
    """Total rm needed to hold every expert at its padded offset."""
    cu = cu_seqlens.tolist()
    e = len(cu) - 1
    last_len = cu[e] - cu[e - 1]
    return (cu[e - 1] // 128 + (e - 1)) + ceil_div(last_len, 128)


# --------------------------------------------------------------------------
# the P1 op, as a reference
# --------------------------------------------------------------------------


def fused_gather_dual_quantize_ref(
    x: torch.Tensor,              # (T, d) bf16 -- unpermuted activations or grads
    gather_idx: torch.Tensor,     # (M,) int32 -- row of x for each permuted slot
    cu_seqlens: torch.Tensor,     # (E+1,) int32 -- expert boundaries in permuted order
    pts_row: torch.Tensor,        # scalar fp32 per-tensor scale, rowwise operand
    pts_col: torch.Tensor | None = None,   # scalar fp32, colwise operand
    want_rowwise: bool = True,
    want_colwise: bool = True,
):
    """Gather x into expert order and emit NVFP4 in both quantization axes.

    Replaces the current Megatron+TE two-kernel chain
        BF16 moe_permute  ->  grouped quantize (+ transpose)
    which materializes a BF16 (M, d) intermediate and quantizes every token k
    times. Here x is read once per output row and nothing high-precision is
    written back.

    rowwise  = quantized along d (hidden). Feeds the fwd/dgrad grouped GEMM as
               the A operand; SF uses the varlen padded tile offsets.
    colwise  = quantized along the token axis, per expert segment. Feeds the
               wgrad grouped GEMM; stored transposed, each expert's token
               extent padded to 128 so an SF atom never straddles an expert.

    The colwise axis is *only* well-defined after the gather: 16 consecutive
    tokens of one expert are not 16 consecutive rows of x. That ordering
    constraint is the reason this op has to fuse rather than compose.
    """
    assert x.dtype == torch.bfloat16 and x.dim() == 2
    T, d = x.shape
    M = gather_idx.numel()
    E = cu_seqlens.numel() - 1
    assert d % 16 == 0
    out = {}

    xg = x[gather_idx.long()]  # (M, d) -- the only high-precision read

    if want_rowwise:
        qdata, sf2d = quantize_nvfp4_lastdim(xg, pts_row)        # (M, d/2), (M, d/16)
        rm_total = varlen_sf_num_tiles(cu_seqlens)
        rk = ceil_div(d // 16, 4)
        sf_buf = torch.zeros(rm_total, rk, 32, 4, 4, dtype=torch.uint8, device=x.device)
        offs = varlen_sf_tile_offsets(cu_seqlens)
        cu = cu_seqlens.tolist()
        for b in range(E):
            lo, hi = cu[b], cu[b + 1]
            if hi == lo:
                continue
            blk = pack_sf_blocked(sf2d[lo:hi], rm=ceil_div(hi - lo, 128), rk=rk)
            sf_buf[offs[b] : offs[b] + blk.shape[0]] = blk.view(torch.uint8)
        out["rowwise"] = {
            "qdata": qdata,                                       # (M, d/2) uint8
            "sf": sf_buf.view(torch.float8_e4m3fn),                # (rm_total, rk, 32,4,4)
            "sf_tile_offsets": torch.tensor(offs, dtype=torch.int32, device=x.device),
            "per_tensor_scale": pts_row,
        }

    if want_colwise:
        pts_col = pts_row if pts_col is None else pts_col
        cu = cu_seqlens.tolist()
        seg_pad = [ceil_div(cu[b + 1] - cu[b], 128) * 128 for b in range(E)]
        m_pad_total = sum(seg_pad)
        q_col = torch.zeros(d, m_pad_total // 2, dtype=torch.uint8, device=x.device)
        sf_col_2d = torch.zeros(d, m_pad_total // 16, dtype=torch.uint8, device=x.device)
        seg_off, cur = [], 0
        for b in range(E):
            lo, hi = cu[b], cu[b + 1]
            seg_off.append(cur)
            n = hi - lo
            if n:
                # (n, d) -> (d, n) then quantize along the token axis; pad the
                # segment to 128 tokens with zeros before quantizing so the
                # group structure matches what the wgrad GEMM will read.
                seg = torch.zeros(seg_pad[b], d, dtype=torch.bfloat16, device=x.device)
                seg[:n] = xg[lo:hi]
                qd, sfd = quantize_nvfp4_lastdim(seg.t().contiguous(), pts_col)
                q_col[:, cur // 2 : cur // 2 + seg_pad[b] // 2] = qd
                sf_col_2d[:, cur // 16 : cur // 16 + seg_pad[b] // 16] = sfd.view(torch.uint8)
            cur += seg_pad[b]
        out["colwise"] = {
            "qdata": q_col,                                        # (d, m_pad/2) uint8
            "sf": pack_sf_blocked(sf_col_2d.view(torch.float8_e4m3fn)),
            "sf_2d": sf_col_2d.view(torch.float8_e4m3fn),
            "seg_offsets": torch.tensor(seg_off, dtype=torch.int32, device=x.device),
            "seg_padded_lens": torch.tensor(seg_pad, dtype=torch.int32, device=x.device),
            "per_tensor_scale": pts_col,
        }

    return out


# --------------------------------------------------------------------------
# traffic model (PLAN.md 3.3) -- used to sanity-check ncu numbers
# --------------------------------------------------------------------------


def traffic_bytes(T: int, d: int, k: int, dual: bool = True):
    """DRAM bytes for the baseline chain vs the fused kernel, per (T,d) tensor.

    The gather read appears in BOTH designs -- the baseline's BF16 permute reads
    x exactly the same way the fused kernel does -- so the L2-reuse assumption
    must be applied to both or the comparison is rigged. Each token is read k
    times; with perfect L2 reuse that costs 2*T*d, with none 2*M*d.

    PLAN.md 3.3 originally quoted 2.3-5.2x, which came from charging the
    baseline the no-reuse read (2*M*d) while crediting the fused kernel with the
    reuse read (2*T*d). Matched assumptions give 2.3x (no reuse) to 3.9x (full
    reuse); the 5.2x figure is not attainable by this kernel and should not be
    quoted.
    """
    M = T * k
    q = 0.5 + 1.0 / 16  # qdata + SF bytes per element
    n_out = 2 if dual else 1
    out = {}
    for regime, read in (("noreuse", 2 * M * d), ("reuse", 2 * T * d)):
        baseline = (
            read                       # BF16 permute: read x
            + 2 * M * d                # BF16 permute: write the permuted copy
            + 2 * M * d                # quantize: read it back (too big for L2)
            + n_out * q * M * d        # quantize: write qdata+SF
        )
        fused = read + n_out * q * M * d
        out[f"baseline_{regime}"] = baseline
        out[f"fused_{regime}"] = fused
        out[f"speedup_{regime}"] = baseline / fused
    return out
