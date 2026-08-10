# Copyright (c) 2026, Sung Hyun Cho.
"""Variable-length rowwise and colwise NVFP4 quantization kernels.

Rowwise output feeds variable-M expert GEMMs. Colwise output uses independently
blocked expert segments for grouped weight gradients. The implementation also
supports deterministic stochastic rounding and Transformer Engine-compatible
columnwise RHT. Per-tensor scales are required to avoid silent underflow.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute.nvgpu.warp as cute_warp
import torch
from cutlass import Float32, Int32, const_expr, cute
from cutlass.cutlass_dsl import T, dsl_user_op
from torch import Tensor

from ._common import (
    _asm,
    _bits_f32,
    _f32_bits,
    cvt_f32x4_e2m1x4_rs,
    fake_tensor,
    jit_cache,
    philox,
    torch2cute_dtype_map,
)

F4_MAX = 6.0
E4M3_MAX = 448.0
E4M3_EPS = 2.0**-9
F_TILE = 256  # Row mode streams directly without an smem transpose.
F_TILE_C = 128  # Keeps the columnwise kernel at four CTAs per SM.
SZ_PAD = 8  # Spreads consecutive ldsm/stsm fragments across all bank quads.
NUM_THREADS = 256


@dsl_user_op
def _div_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    # Preserve IEEE round-to-nearest behavior at E4M3 tie boundaries.
    return Float32(
        _asm(
            T.f32(),
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
        )
    )


@dsl_user_op
def _abs_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    # Clearing the sign bit guarantees +0.0 for all-zero amax reductions.
    return Float32(_asm(T.f32(), "abs.f32 $0, $1;", "=f,f", [Float32(a).ir_value(loc=loc, ip=ip)]))


@dsl_user_op
def _max_u32(a: cutlass.Uint32, b: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Uint32:
    # Unsigned ordering matches finite non-negative IEEE-754 bit patterns.
    return cutlass.Uint32(
        _asm(
            T.i32(),
            "max.u32 $0, $1, $2;",
            "=r,r,r",
            [
                cutlass.Uint32(a).ir_value(loc=loc, ip=ip),
                cutlass.Uint32(b).ir_value(loc=loc, ip=ip),
            ],
        )
    )


@dsl_user_op
def _max_bf16x2(a: cutlass.Uint32, b: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Uint32:
    # Packed bf16 max (one alu op for 2 elements; sm_80+). Comparisons do not
    # round and the winner's bits pass through unchanged, so a bf16-domain
    # abs-max tree produces bitwise the same amax as converting every element
    # to f32 first (bf16 -> f32 is exact and monotone). Inputs here are
    # always abs-masked (& 0x7FFF7FFF), so the -0.0 and NaN corners of the
    # plain (non-.NaN) max variant cannot arise.
    return cutlass.Uint32(
        _asm(
            T.i32(),
            "max.bf16x2 $0, $1, $2;",
            "=r,r,r",
            [
                cutlass.Uint32(a).ir_value(loc=loc, ip=ip),
                cutlass.Uint32(b).ir_value(loc=loc, ip=ip),
            ],
        )
    )


# Promoting BF16 bits to FP32 is an exact 16-bit shift.
_ABS32 = 0x7FFFFFFF
_ABSP = 0x7FFF7FFF
_HI16 = 0xFFFF0000


@dsl_user_op
def _cvt_e4m3_rn(a: Float32, *, loc=None, ip=None) -> cutlass.Uint32:
    # Use the cuBLAS ties-to-even contract and return through a 32-bit value.
    return cutlass.Uint32(
        _asm(
            T.i32(),
            "{\n\t.reg .b16 t;\n\tcvt.rn.satfinite.e4m3x2.f32 t, $1, $2;"
            "\n\tcvt.u32.u16 $0, t;\n\t}",
            "=r,f,f",
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(a).ir_value(loc=loc, ip=ip)],
        )
    )


@dsl_user_op
def _cvt_e2m1_pair_rn(hi: Float32, lo: Float32, *, loc=None, ip=None) -> cutlass.Uint32:
    # Pass even elements as lo so they occupy the low nibble. PTX requires a
    # byte destination for the packed result.
    return cutlass.Uint32(
        _asm(
            T.i32(),
            "{\n\t.reg .b8 t;\n\tcvt.rn.satfinite.e2m1x2.f32 t, $1, $2;\n\tcvt.u32.u8 $0, t;\n\t}",
            "=r,f,f",
            [Float32(hi).ir_value(loc=loc, ip=ip), Float32(lo).ir_value(loc=loc, ip=ip)],
        )
    )


class NVFP4QuantKernel:
    def __init__(
        self,
        in_dtype: type[cutlass.Numeric],
        mode: str,
        gather: bool,
        rounding: str = "rn",
        sf_layout: str = "blocked",
        rht: bool = False,
        emit_amax: bool = False,
        has_tile_offsets: bool = False,
        experts: int = 0,
    ):
        assert mode in ("row", "col", "amax")
        # emit_amax (col only): the quantizer already computes every group's
        # amax over the values it quantizes (post-RHT when rht=True) - emit
        # the CTA max of those as one f32 partial per CTA (same value the
        # mode="amax" pre-pass would produce for this tile, bitwise: max is
        # exact). The delayed-col-amax recipe uses it as NEXT step's pts
        # source, deleting the pre-pass read. No atomics: partials buffer +
        # torch.amax, order-independent.
        assert not emit_amax or mode == "col", "emit_amax is a colwise feature"
        self.emit_amax = emit_amax
        self.has_tile_offsets = has_tile_offsets
        self.experts = experts
        assert not has_tile_offsets or experts > 0
        assert rounding in ("rn", "sr")
        assert sf_layout in ("blocked", "linear")
        assert sf_layout == "blocked" or mode == "row", "linear SF is rowwise"
        # rht: 16-token-group hadamard transform on the quantization axis
        # (TE columnwise convention); "amax" mode always transforms (its whole
        # point is the post-RHT columnwise amax) and also emits the raw row
        # amax from the staged tile.
        assert not rht or mode == "col", "rht applies to the colwise quantizer"
        assert in_dtype == cutlass.BFloat16 or (not rht and mode != "amax"), (
            "RHT is bf16-only (TE constraint)"
        )
        self.in_dtype = in_dtype
        self.mode = mode
        self.rht = rht or mode == "amax"
        self.gather = gather
        # sf_layout="linear" (row only): SF stored (rows, F/16) row-major -
        # the linear-SF gather-GEMM SFA contract (view as int32 (rows, F/64));
        # x is quantized ONCE at T rows before routing, so this is the fwd
        # x-quantize output format. blocked = the NVFP4 atom layout consumed
        # by grouped SM100 GEMMs.
        self.sf_layout = sf_layout
        # rounding="sr": hw cvt.rs.satfinite.e2m1x4 fed by Philox(counter=
        # global OUTPUT position, key=seed) - one philox call per 16-group
        # (4 words x 4 values). Same seed + same routing => bitwise
        # reproducible; SF selection stays RN (TE applies SR to the data
        # cast only). Bit-match vs TE is impossible in principle for SR;
        # the gates are fixed-seed reproducibility + unbiasedness.
        self.rounding = rounding

    @cute.jit
    def __call__(
        self,
        mZ: cute.Tensor,  # (rows, F) in_dtype; gather: rows = T source rows
        mGidx,  # Optional (M,) Int32 source-row indices
        mCu: cute.Tensor,  # (E+1,) Int32
        mTileEnds,  # Optional (E,) Int32 inclusive 128-padded row prefixes
        mQ: cute.Tensor,  # row: (M, F/2) Uint8 | col: (F, mp_tot/2) Uint8
        #                   amax: (num_seg_tiles * F/128, 2) Float32 partials
        mSF,  # Uint8 view of the blocked SF buffer (flat); None in amax mode
        mH,  # Optional (16, 16) BFloat16 rht matrix S@H16*0.25 (rht/amax only)
        mPts,  # (2,) Float32 device-resident [pts, 1/pts]; None in amax mode
        mAmaxOut,  # Optional (>= num_seg_tiles * F/128,) Float32 (emit_amax)
        num_seg_tiles: Int32,  # host upper bound: ceil(M/128) + E
        F: Int32,
        seed: Int32,  # Philox key (rounding="sr" only; ignored for RN)
        stream: cuda.CUstream,
    ):
        f_tile = F_TILE_C if self.mode in ("col", "amax") else F_TILE
        self.kernel(mZ, mGidx, mCu, mTileEnds, mQ, mSF, mH, mPts, mAmaxOut, F, seed).launch(
            grid=[cute.ceil_div(F, f_tile), num_seg_tiles, 1],
            block=[NUM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mZ: cute.Tensor,
        mGidx,
        mCu: cute.Tensor,
        mTileEnds,
        mQ: cute.Tensor,
        mSF,
        mH,
        mPts,
        mAmaxOut,
        F: Int32,
        seed: Int32,
    ):
        fi, p, _ = cute.arch.block_idx()
        pts = Float32(0.0)
        inv_pts = Float32(0.0)
        if const_expr(self.mode != "amax"):
            pts = Float32(mPts[0])
            inv_pts = Float32(mPts[1])
        tidx, _, _ = cute.arch.thread_idx()
        E = self.experts if const_expr(self.has_tile_offsets) else mCu.shape[0] - 1
        # all smem lives at kernel scope: the allocator is a compile-time
        # (Meta) object and cannot be carried through the dynamic e_found
        # branch; emit_amax additionally needs its reduce buffer in EVERY
        # CTA (tiles past the last real segment still store a 0 partial -
        # the caller reduces over the exact grid, no zero-fill kernel)
        smem = cutlass.utils.SmemAllocator()
        cta_amax = Float32(0.0)
        if const_expr(self.emit_amax):
            sRedE = smem.allocate_tensor(Float32, cute.make_layout(NUM_THREADS), byte_alignment=16)
        if const_expr(self.mode in ("col", "amax")):
            # The columnwise path uses a 47 KB staging footprint. Keep spitch
            # consistent in the layout, load destination, and RHT tile.
            spitch = F_TILE_C + SZ_PAD
            sZ = smem.allocate_tensor(
                self.in_dtype,
                cute.make_layout((128, spitch), stride=(spitch, 1)),
                byte_alignment=16,
            )
            # raw-bits view for the integer-pipe quantize loop (same
            # addresses; a 16-bit smem load zero-extends into a 32-bit
            # register for free)
            sZu = cute.make_tensor(
                cute.recast_ptr(sZ.iterator, dtype=cutlass.Uint16),
                sZ.layout,
            )
            if const_expr(self.mode == "col"):
                sQ = smem.allocate_tensor(
                    cutlass.Uint8,
                    cute.make_layout((F_TILE_C, 80), stride=(80, 1)),
                    byte_alignment=16,
                )
            else:
                sRed = smem.allocate_tensor(
                    Float32,
                    cute.make_layout((NUM_THREADS, 2), stride=(2, 1)),
                    byte_alignment=16,
                )
        else:
            sQ = smem.allocate_tensor(
                cutlass.Uint8,
                cute.make_layout((128, 144), stride=(144, 1)),
                byte_alignment=16,
            )
        # scan cu for this segment tile: expert e, local tile lt, and offsets
        acc = Int32(0)
        e_found = Int32(-1)
        lt = Int32(0)
        row0 = Int32(0)
        len_e = Int32(0)
        colpad0 = Int32(0)
        nt_e = Int32(0)
        if const_expr(self.has_tile_offsets):
            lo = Int32(0)
            hi = Int32(E)
            for _ in cutlass.range_constexpr(E.bit_length()):
                mid = (lo + hi) // 2
                if (mid < E) and (p < mTileEnds[mid] // 128):
                    hi = mid
                else:
                    lo = mid + 1
            if lo < E:
                e_found = lo
                prev_tiles = Int32(0)
                if lo > 0:
                    prev_tiles = mTileEnds[lo - 1] // 128
                lt = p - prev_tiles
                row0 = mCu[lo]
                len_e = mCu[lo + 1] - row0
                colpad0 = prev_tiles * 128
                nt_e = mTileEnds[lo] // 128 - prev_tiles
        else:
            for e in cutlass.range(E):
                le = mCu[e + 1] - mCu[e]
                nt = (le + 127) // 128
                if (p >= acc) and (p < acc + nt):
                    e_found = Int32(e)
                    lt = p - acc
                    row0 = mCu[e]
                    len_e = le
                    colpad0 = acc * 128
                    nt_e = nt
                acc += nt
        if e_found >= 0:
            f_tile = F_TILE_C if self.mode in ("col", "amax") else F_TILE
            f0 = fi * f_tile
            valid_rows = cutlass.min(len_e - lt * 128, Int32(128))
            elts = 128 // self.in_dtype.width  # per 16-byte chunk
            atom_ld = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.in_dtype, num_bits_per_copy=128
            )
            lay_v = cute.make_layout(elts)
            zstride = mZ.stride[0]
            # amax accumulators: abs-masked bf16x2 pairs, unpacked to f32 once
            # at the reduce (bitwise equal to an f32 max chain - see the
            # helper docstrings)
            row_pk = cutlass.Uint32(0)
            col_pk = cutlass.Uint32(0)
            if const_expr(self.mode in ("col", "amax")):
                # staged load: 16B chunks, coalesced; OOB rows/cols -> 0
                chunks_per_row = F_TILE_C // elts
                frag = cute.make_rmem_tensor(lay_v, self.in_dtype)
                for i in cutlass.range_constexpr(128 * chunks_per_row // NUM_THREADS):
                    c = tidx + i * NUM_THREADS
                    r = c // chunks_per_row
                    c8 = c % chunks_per_row
                    fcol = f0 + c8 * elts
                    ok = (r < valid_rows) and (fcol < F)
                    if ok:
                        rg = row0 + lt * 128 + r
                        if const_expr(self.gather):
                            rg = mGidx[rg]
                        src = cute.make_tensor(
                            mZ.iterator + cute.assume(rg * zstride + fcol, divby=elts),
                            lay_v,
                        )
                        cute.copy(atom_ld, src, frag)
                    else:
                        frag.fill(0.0)
                    if const_expr(self.mode == "amax"):
                        # raw (pre-RHT) amax of this tile's valid values -
                        # OOB slots are zero-filled so they cannot win the max
                        fu = cute.make_tensor(
                            cute.recast_ptr(frag.iterator, dtype=cutlass.Uint32),
                            cute.make_layout(elts // 2),
                        )
                        for ee in cutlass.range_constexpr(elts // 2):
                            row_pk = _max_bf16x2(row_pk, fu[ee] & cutlass.Uint32(_ABSP))
                    # F_TILE_C prevents mid-row skew; spitch supplies the pad.
                    pcol = c8 * elts
                    dst = cute.make_tensor(
                        sZ.iterator + cute.assume(r * spitch + pcol, divby=elts),
                        lay_v,
                    )
                    cute.copy(atom_ld, frag, dst)
                cute.arch.sync_threads()

            if const_expr(self.rht):
                # Apply the 16-token Hadamard transform in place. Each warp
                # owns a feature range; sync_warp protects A reads from C writes.
                wi = tidx // 32
                lane = tidx % 32
                op = cute_warp.MmaF16BF16Op(cutlass.BFloat16, Float32, (16, 8, 16))
                tiled_mma = cute.make_tiled_mma(op)
                thr_mma = tiled_mma.get_slice(lane)
                atom_cp = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16)
                # Transposed ldmatrix/stmatrix operations preserve the m-major
                # shared-memory layout for each warp-owned 16x16 tile.
                tiled_ld_a = cute.make_tiled_copy_A(
                    cute.make_copy_atom(
                        cute_warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
                        cutlass.BFloat16,
                    ),
                    tiled_mma,
                )
                thr_ld_a = tiled_ld_a.get_slice(lane)
                if const_expr(self.mode == "col"):
                    tiled_st_c = cute.make_tiled_copy_C(
                        cute.make_copy_atom(
                            cute_warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4),
                            cutlass.BFloat16,
                        ),
                        tiled_mma,
                    )
                    thr_st_c = tiled_st_c.get_slice(lane)
                gB = cute.make_tensor(mH.iterator, cute.make_layout((16, 16), stride=(1, 16)))
                tCgB = thr_mma.partition_B(gB)
                tCrB = tiled_mma.make_fragment_B(tCgB.shape)
                cute.copy(atom_cp, tCgB, tCrB)
                for g in cutlass.range_constexpr(8):
                    # divby=8 halfwords = 16B: the ldsm/stsm atoms need the
                    # provable 128-bit base alignment the dynamic wi*16 term
                    # hides (wi*16 halfwords = 32B, always aligned in fact)
                    sA = cute.make_tensor(
                        sZ.iterator + cute.assume(g * 16 * spitch + wi * 16, divby=8),
                        cute.make_layout((16, 16), stride=(1, spitch)),
                    )
                    tCsA = thr_mma.partition_A(sA)
                    tCrA = tiled_mma.make_fragment_A(tCsA.shape)
                    cute.copy(tiled_ld_a, thr_ld_a.partition_S(sA), tiled_ld_a.retile(tCrA))
                    racc = tiled_mma.make_fragment_C(thr_mma.partition_shape_C((16, 16)))
                    racc.fill(0.0)
                    cute.gemm(tiled_mma, racc, tCrA, tCrB, racc)
                    rC = cute.make_fragment_like(racc, cutlass.BFloat16)
                    rC.store(racc.load().to(cutlass.BFloat16))
                    if const_expr(self.mode == "amax"):
                        # post-RHT col amax straight from the bf16-rounded
                        # fragment (max of |bf16| is exact - the reduction
                        # order can never change the result bits)
                        rCu = cute.make_tensor(
                            cute.recast_ptr(rC.iterator, dtype=cutlass.Uint32),
                            cute.make_layout(cute.size(rC.shape) // 2),
                        )
                        for ii in cutlass.range_constexpr(cute.size(rC.shape) // 2):
                            col_pk = _max_bf16x2(col_pk, rCu[ii] & cutlass.Uint32(_ABSP))
                    else:
                        cute.arch.sync_warp()
                        cute.copy(tiled_st_c, tiled_st_c.retile(rC), thr_st_c.partition_D(sA))
                if const_expr(self.mode == "col"):
                    cute.arch.sync_threads()

            atom_q = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.Uint8, num_bits_per_copy=64
            )
            atom_sf = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.Uint8, num_bits_per_copy=32
            )
            lay8 = cute.make_layout(8)
            lay4 = cute.make_layout(4)
            v16 = cute.make_layout(16)
            q32 = cute.make_rmem_tensor(v16, Float32)
            # Hardware conversion provides the cuBLAS ties-to-even contract.
            sf1 = cute.make_rmem_tensor(cute.make_layout(2), cutlass.Float8E4M3FN)
            qbytes = cute.make_rmem_tensor(lay8, cutlass.Uint8)
            sfbytes = cute.make_rmem_tensor(lay8, cutlass.Uint8)

            qrow_stride = mQ.stride[0]
            atom_q16 = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.Uint8, num_bits_per_copy=128
            )
            lay16 = cute.make_layout(16)
            a8u = cute.make_rmem_tensor(lay8, cutlass.Uint32)
            if const_expr(self.mode == "amax"):
                # CTA tree-reduce (row_amax, col_amax) -> one partial
                # pair per CTA; the host reduces partials with torch.amax
                # (max is exact, so this is bitwise order-independent).
                # unpack the bf16x2 accumulators: each abs half's f32 bits
                # are half << 16; non-negative f32 order == u32 order, so
                # one integer max merges the halves exactly
                row_amax = _bits_f32(_max_u32(row_pk << 16, row_pk & cutlass.Uint32(_HI16)))
                col_amax = _bits_f32(_max_u32(col_pk << 16, col_pk & cutlass.Uint32(_HI16)))
                sRed[tidx, 0] = row_amax
                sRed[tidx, 1] = col_amax
                cute.arch.sync_threads()
                for off in (128, 64, 32, 16, 8, 4, 2, 1):
                    if tidx < off:
                        sRed[tidx, 0] = cutlass.max(sRed[tidx, 0], sRed[tidx + off, 0])
                        sRed[tidx, 1] = cutlass.max(sRed[tidx, 1], sRed[tidx + off, 1])
                    cute.arch.sync_threads()
                if tidx == 0:
                    pi = p * (F // F_TILE_C) + fi
                    mQ[pi, 0] = sRed[0, 0]
                    mQ[pi, 1] = sRed[0, 1]
            elif const_expr(self.mode == "col"):
                # thread = (feature col, token half); 4 groups of 16 tokens.
                # One scalar read pass feeds a depth-5 amax tree.
                f = f0 + tidx % 128
                half = tidx // 128
                pc = tidx % 128
                if f < F:
                    for gg in cutlass.range_constexpr(4):
                        g = half * 4 + gg
                        # convert = one shl per element (int pipe); amax =
                        # abs-masked u32 max tree (int pipe) - bitwise an
                        # f32 cvt/neg/max chain
                        for tt in cutlass.range_constexpr(16):
                            q32[tt] = _bits_f32(sZu[g * 16 + tt, pc].to(cutlass.Uint32) << 16)
                        for i2 in cutlass.range_constexpr(8):
                            a8u[i2] = _max_u32(
                                _f32_bits(q32[2 * i2]) & cutlass.Uint32(_ABS32),
                                _f32_bits(q32[2 * i2 + 1]) & cutlass.Uint32(_ABS32),
                            )
                        for i2 in cutlass.range_constexpr(4):
                            a8u[i2] = _max_u32(a8u[i2], a8u[i2 + 4])
                        amax = _bits_f32(
                            _max_u32(_max_u32(a8u[0], a8u[1]), _max_u32(a8u[2], a8u[3]))
                        )
                        if const_expr(self.emit_amax):
                            # every quantized group's amax is already in hand
                            # (post-RHT when rht=True): fold the CTA max for
                            # the delayed-recipe partial (bitwise the value
                            # the mode="amax" pre-pass computes for this tile)
                            cta_amax = cutlass.max(cta_amax, amax)
                        if const_expr(self.rht):
                            # Match TE: sf=min(vec_max*ges/6, 448), without a
                            # lower clamp, and recip=min(1/(sf*gds), f32max).
                            scaled = cutlass.min(_abs_f32(amax) * pts, Float32(E4M3_MAX))
                        else:
                            bs = _div_rn(amax, Float32(F4_MAX))
                            scaled = _div_rn(bs, pts)
                            scaled = cutlass.max(
                                cutlass.min(scaled, Float32(E4M3_MAX)),
                                Float32(E4M3_EPS),
                            )
                        sf1u8 = cute.make_tensor(
                            cute.recast_ptr(sf1.iterator, dtype=cutlass.Uint8),
                            cute.make_layout(1),
                        )
                        sf1u8[0] = cutlass.Uint8(_cvt_e4m3_rn(scaled) & 0xFF)
                        sfbytes[gg] = sf1u8[0]
                        if const_expr(self.rht):
                            recip = cutlass.min(
                                _div_rn(Float32(1.0), sf1[0].to(Float32) * inv_pts),
                                Float32(3.4028234663852886e38),
                            )
                        else:
                            recip = _div_rn(inv_pts, sf1[0].to(Float32))
                        # no +-6 clamp: both e2m1 converters are .satfinite
                        # (|v| > 6 incl inf -> sign|0x7, bitwise the
                        # clamp-then-convert result; NaN cannot arise -
                        # recip is finite by construction in both scale
                        # chains)
                        for tt in cutlass.range_constexpr(16):
                            q32[tt] = q32[tt] * recip
                        if const_expr(self.rounding == "sr"):
                            tg = colpad0 // 16 + lt * 8 + g
                            ctr = (cutlass.Uint64(f) << cutlass.Uint64(32)) | cutlass.Uint64(tg)
                            rw = philox(ctr, cutlass.Uint32(seed))
                            for j4 in cutlass.range_constexpr(4):
                                u16 = cvt_f32x4_e2m1x4_rs(
                                    q32[4 * j4],
                                    q32[4 * j4 + 1],
                                    q32[4 * j4 + 2],
                                    q32[4 * j4 + 3],
                                    rw[j4],
                                )
                                qbytes[2 * j4] = cutlass.Uint8(u16 & 0xFF)
                                qbytes[2 * j4 + 1] = cutlass.Uint8((u16 >> 8) & 0xFF)
                        else:
                            for bb in cutlass.range_constexpr(8):
                                qbytes[bb] = cutlass.Uint8(
                                    _cvt_e2m1_pair_rn(q32[2 * bb + 1], q32[2 * bb]) & 0xFF
                                )
                        dq = cute.make_tensor(
                            sQ.iterator + cute.assume(pc * 80 + g * 8, divby=8),
                            lay8,
                        )
                        cute.copy(atom_q, qbytes, dq)
                    # SF: per-expert-blocked concat (cudnn-FE wgrad contract).
                    # expert base = F*colpad0/16 B; inside, (F/128, 2*nt_e,
                    # 32, 4, 4) atoms; this thread's 4 sf cols = one atom col
                    # (lt*2 + half). F % 128 == 0. (byte offsets stay < 2^31
                    # The supported shapes keep F*mp_tot/16 within Int32.
                    base = (
                        F * (colpad0 // 16)
                        + (f // 128) * (nt_e * 2) * 512
                        + (f % 32) * 16
                        + ((f % 128) // 32) * 4
                    )
                    w4 = cute.make_rmem_tensor(lay4, cutlass.Uint8)
                    for bb in cutlass.range_constexpr(4):
                        w4[bb] = sfbytes[bb]
                    dsf = cute.make_tensor(
                        mSF.iterator + cute.assume(base + (lt * 2 + half) * 512, divby=4),
                        lay4,
                    )
                    cute.copy(atom_sf, w4, dsf)
                cute.arch.sync_threads()
                # cooperative qdata store: 16B chunks, 64B contiguous per row
                w16 = cute.make_rmem_tensor(lay16, cutlass.Uint8)
                for i in cutlass.range_constexpr(F_TILE_C * 4 // NUM_THREADS):
                    c = tidx + i * NUM_THREADS
                    r2 = c // 4
                    c16 = c % 4
                    if f0 + r2 < F:
                        src = cute.make_tensor(
                            sQ.iterator + cute.assume(r2 * 80 + c16 * 16, divby=16),
                            lay16,
                        )
                        cute.copy(atom_q16, src, w16)
                        dq2 = cute.make_tensor(
                            mQ.iterator
                            + cute.assume(
                                (f0 + r2) * qrow_stride + colpad0 // 2 + lt * 64 + c16 * 16,
                                divby=16,
                            ),
                            lay16,
                        )
                        cute.copy(atom_q16, w16, dq2)
            else:
                # Two half-warps read two complete rows, keeping gathered
                # source transactions contiguous.
                warp = tidx // 32
                lane = tidx % 32
                row_half = lane // 16
                g = lane % 16
                frag16 = cute.make_rmem_tensor(cute.make_layout(16), self.in_dtype)
                fragu = cute.make_tensor(
                    cute.recast_ptr(frag16.iterator, dtype=cutlass.Uint32),
                    cute.make_layout(8),
                )
                fcol = Int32(0)
                rg = Int32(0)
                zbase = cutlass.Int64(0)
                rk_tot = Int32(0)
                sftile = Int32(0)
                orow = Int32(0)
                rk_tot = F // 64
                sftile = row0 // 128 + e_found + lt
                for r8 in cutlass.range_constexpr(8):
                    r = warp * 16 + r8 * 2 + row_half
                    if r < valid_rows:
                        fcol = f0 + g * 16
                        if fcol < F:
                            rg = row0 + lt * 128 + r
                            if const_expr(self.gather):
                                rg = mGidx[rg]
                            zbase = rg * zstride
                            for h2 in cutlass.range_constexpr(2):
                                srcv = cute.make_tensor(
                                    mZ.iterator + cute.assume(zbase + fcol + h2 * 8, divby=8),
                                    lay8,
                                )
                                dstv = cute.make_tensor(frag16.iterator + h2 * 8, lay8)
                                cute.copy(atom_ld, srcv, dstv)
                            for i2 in cutlass.range_constexpr(8):
                                w = fragu[i2]
                                q32[2 * i2] = _bits_f32(w << 16)
                                q32[2 * i2 + 1] = _bits_f32(w & cutlass.Uint32(_HI16))
                            t0 = _max_bf16x2(
                                fragu[0] & cutlass.Uint32(_ABSP), fragu[1] & cutlass.Uint32(_ABSP)
                            )
                            t1 = _max_bf16x2(
                                fragu[2] & cutlass.Uint32(_ABSP), fragu[3] & cutlass.Uint32(_ABSP)
                            )
                            t2 = _max_bf16x2(
                                fragu[4] & cutlass.Uint32(_ABSP), fragu[5] & cutlass.Uint32(_ABSP)
                            )
                            t3 = _max_bf16x2(
                                fragu[6] & cutlass.Uint32(_ABSP), fragu[7] & cutlass.Uint32(_ABSP)
                            )
                            pk = _max_bf16x2(_max_bf16x2(t0, t1), _max_bf16x2(t2, t3))
                            amax = _bits_f32(_max_u32(pk << 16, pk & cutlass.Uint32(_HI16)))
                            bs = _div_rn(amax, Float32(F4_MAX))
                            scaled = _div_rn(bs, pts)
                            scaled = cutlass.max(
                                cutlass.min(scaled, Float32(E4M3_MAX)), Float32(E4M3_EPS)
                            )
                            sf1u8 = cute.make_tensor(
                                cute.recast_ptr(sf1.iterator, dtype=cutlass.Uint8),
                                cute.make_layout(1),
                            )
                            sf1u8[0] = cutlass.Uint8(_cvt_e4m3_rn(scaled) & 0xFF)
                            recip = _div_rn(inv_pts, sf1[0].to(Float32))
                            for tt in cutlass.range_constexpr(16):
                                q32[tt] = q32[tt] * recip
                            if const_expr(self.rounding == "sr"):
                                orow = row0 + lt * 128 + r
                                ctr = (cutlass.Uint64(orow) << cutlass.Uint64(32)) | cutlass.Uint64(
                                    fcol // 16
                                )
                                rw = philox(ctr, cutlass.Uint32(seed))
                                for j4 in cutlass.range_constexpr(4):
                                    u16 = cvt_f32x4_e2m1x4_rs(
                                        q32[4 * j4],
                                        q32[4 * j4 + 1],
                                        q32[4 * j4 + 2],
                                        q32[4 * j4 + 3],
                                        rw[j4],
                                    )
                                    qbytes[2 * j4] = cutlass.Uint8(u16 & 0xFF)
                                    qbytes[2 * j4 + 1] = cutlass.Uint8((u16 >> 8) & 0xFF)
                            else:
                                for bb in cutlass.range_constexpr(8):
                                    qbytes[bb] = cutlass.Uint8(
                                        _cvt_e2m1_pair_rn(q32[2 * bb + 1], q32[2 * bb]) & 0xFF
                                    )
                            dq = cute.make_tensor(
                                sQ.iterator + cute.assume(r * 144 + g * 8, divby=8), lay8
                            )
                            cute.copy(atom_q, qbytes, dq)
                            sQ[r, 128 + g] = sf1u8[0]
                cute.arch.sync_threads()
                if const_expr(self.sf_layout == "linear"):
                    r2 = tidx // 2
                    half = tidx % 2
                    if (r2 < valid_rows) and (f0 + half * 128 < F):
                        sf8 = cute.make_rmem_tensor(lay8, cutlass.Uint8)
                        ssf = cute.make_tensor(
                            sQ.iterator + cute.assume(r2 * 144 + 128 + half * 8, divby=8), lay8
                        )
                        cute.copy(atom_q, ssf, sf8)
                        orow = row0 + lt * 128 + r2
                        dsf8 = cute.make_tensor(
                            mSF.iterator
                            + cute.assume(orow * (F // 16) + f0 // 16 + half * 8, divby=8),
                            lay8,
                        )
                        cute.copy(atom_q, sf8, dsf8)
                else:
                    for i in cutlass.range_constexpr(2):
                        c = tidx + i * NUM_THREADS
                        r2 = c // 4
                        atom = c % 4
                        if (r2 < valid_rows) and (f0 + atom * 64 < F):
                            sf4 = cute.make_rmem_tensor(lay4, cutlass.Uint8)
                            ssf = cute.make_tensor(
                                sQ.iterator + cute.assume(r2 * 144 + 128 + atom * 4, divby=4),
                                lay4,
                            )
                            cute.copy(atom_sf, ssf, sf4)
                            base = sftile * rk_tot * 512 + (r2 % 32) * 16 + (r2 // 32) * 4
                            dsf = cute.make_tensor(
                                mSF.iterator + cute.assume(base + (f0 // 64 + atom) * 512, divby=4),
                                lay4,
                            )
                            cute.copy(atom_sf, sf4, dsf)
                # cooperative qdata store: 16B chunks, 128B contiguous per row
                w16 = cute.make_rmem_tensor(lay16, cutlass.Uint8)
                for i in cutlass.range_constexpr(128 * 8 // NUM_THREADS):
                    c = tidx + i * NUM_THREADS
                    r2 = c // 8
                    c16 = c % 8
                    if (r2 < valid_rows) and (f0 + c16 * 32 < F):
                        src = cute.make_tensor(
                            sQ.iterator + cute.assume(r2 * 144 + c16 * 16, divby=16),
                            lay16,
                        )
                        cute.copy(atom_q16, src, w16)
                        dq2 = cute.make_tensor(
                            mQ.iterator
                            + cute.assume(
                                (row0 + lt * 128 + r2) * qrow_stride + f0 // 2 + c16 * 16,
                                divby=16,
                            ),
                            lay16,
                        )
                        cute.copy(atom_q16, w16, dq2)
        if const_expr(self.emit_amax):
            # CTA tree-reduce -> ONE f32 partial per CTA, stored
            # unconditionally (skipped tiles contribute 0, so the caller
            # reduces the exact grid slice without a zero-fill kernel).
            # _abs_f32 pins the zero sign: PTX max.f32 does not define it,
            # and the pre-pass partial must remain bitwise +0.0.
            sRedE[tidx] = cta_amax
            cute.arch.sync_threads()
            for off in (128, 64, 32, 16, 8, 4, 2, 1):
                if tidx < off:
                    sRedE[tidx] = cutlass.max(sRedE[tidx], sRedE[tidx + off])
                cute.arch.sync_threads()
            if tidx == 0:
                mAmaxOut[p * (F // F_TILE_C) + fi] = _abs_f32(sRedE[0])

    @staticmethod
    @jit_cache
    def compile(
        in_dtype,
        mode,
        gather,
        rounding="rn",
        sf_layout="blocked",
        rht=False,
        emit_amax=False,
        has_tile_offsets=False,
        experts=0,
    ):
        rows, F_sym, qm, qn, sfn, m_sym, ep1, am = (cute.sym_int() for _ in range(8))
        mZ = fake_tensor(in_dtype, (rows, F_sym), 8)
        mGidx = fake_tensor(Int32, (m_sym,), 1) if gather else None
        mCu = fake_tensor(Int32, (experts + 1 if has_tile_offsets else ep1,), 1)
        mTileEnds = fake_tensor(Int32, (experts,), 1) if has_tile_offsets else None
        if mode == "amax":
            mQ = fake_tensor(Float32, (qm, 2), 2)  # per-CTA partial pairs
            mSF = None
            mPts = None
        else:
            mQ = fake_tensor(cutlass.Uint8, (qm, qn), 16)  # 16B store chunks
            mSF = fake_tensor(cutlass.Uint8, (sfn,), 8)  # linear: 8B stores
            mPts = fake_tensor(Float32, (2,), 1)
        mAmaxOut = fake_tensor(Float32, (am,), 1) if emit_amax else None
        need_h = rht or mode == "amax"
        mH = fake_tensor(cutlass.BFloat16, (16, 16), 16) if need_h else None
        return cute.compile(
            NVFP4QuantKernel(
                in_dtype,
                mode,
                gather,
                rounding,
                sf_layout,
                rht,
                emit_amax,
                has_tile_offsets,
                experts,
            ),
            mZ,
            mGidx,
            mCu,
            mTileEnds,
            mQ,
            mSF,
            mH,
            mPts,
            mAmaxOut,
            Int32(1),
            Int32(16),
            Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )


# TE get_wgrad_sign_vector (hard-coded random signs; bitmask 0xD7E8)
_RHT_SIGN_VECTOR = (1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1)
_rht_m_cache = {}


def rht_matrix(device="cuda") -> Tensor:
    """(16, 16) bf16 S @ H16 * 0.25 - byte-identical to TE get_rht_matrix
    (with_random_sign_mask=True): Sylvester H16, hard-coded sign vector,
    scale 1/sqrt(16); every entry is +-0.25, exact in bf16."""
    dev = torch.device(device)
    t = _rht_m_cache.get(dev)
    if t is None:
        s = torch.tensor(_RHT_SIGN_VECTOR, dtype=torch.float32, device=dev)
        h = torch.ones(1, 1, device=dev)
        while h.shape[0] < 16:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        t = ((s * torch.eye(16, device=dev)) @ (h * 0.25)).to(torch.bfloat16).contiguous()
        _rht_m_cache[dev] = t
    return t


def nvfp4_rht_amax(
    z: Tensor,
    cu: Tensor,
    partials: Tensor,
    gather_idx: Tensor | None = None,
    padded_offsets: Tensor | None = None,
):
    """TE with_post_rht_amax pre-pass: ONE read of z (rows, F) bf16 emits
    per-CTA partial (raw amax, post-RHT columnwise amax) pairs into partials
    ((>= num_seg_tiles * F/128), 2) f32, which the CALLER must zero-fill
    beforehand and reduce with torch.amax(partials, 0) -> (2,) [row, col].
    max is exact, so the two-level reduction is bitwise deterministic. The
    col amax is the amax of |bf16 RHT transform| over the zero-padded expert
    segments - exactly the values the rht=True colwise quantizer will see."""
    F = z.shape[1]
    assert F % 128 == 0 and z.stride(-1) == 1
    assert z.dtype == torch.bfloat16, "RHT is bf16-only (TE constraint)"
    E = cu.numel() - 1
    _check_padded_offsets(padded_offsets, cu, E)
    M = int(gather_idx.numel()) if gather_idx is not None else z.shape[0]
    n_tiles = -(-M // 128) + E
    assert partials.dtype == torch.float32 and partials.shape[0] >= n_tiles * (F // 128)
    NVFP4QuantKernel.compile(
        torch2cute_dtype_map[z.dtype],
        "amax",
        gather_idx is not None,
        "rn",
        "blocked",
        False,
        False,
        padded_offsets is not None,
        E,
    )(
        z,
        gather_idx,
        cu,
        padded_offsets,
        partials,
        None,
        rht_matrix(z.device),
        None,
        None,
        n_tiles,
        F,
        0,
    )


def _prep(pts) -> Tensor:
    """Return device-resident FP32 ``[pts, 1 / pts]`` without a host sync."""
    if torch.is_tensor(pts):
        if pts.numel() == 2:
            return pts  # prepared [pts, 1/pts] pair (recipe path)
        p32 = pts.detach().to(device="cuda", dtype=torch.float32).reshape(1)
    else:
        assert pts > 0, "per_tensor_scale must be positive (no pts=1 default)"
        p32 = torch.tensor([float(pts)], dtype=torch.float32, device="cuda")
    return torch.cat([p32, 1.0 / p32])


def _check_padded_offsets(padded_offsets: Tensor | None, cu: Tensor, experts: int):
    if padded_offsets is None:
        return
    if (
        padded_offsets.dtype != torch.int32
        or padded_offsets.numel() != experts
        or not padded_offsets.is_cuda
        or not padded_offsets.is_contiguous()
        or padded_offsets.device != cu.device
    ):
        raise ValueError("padded_offsets must be contiguous CUDA int32 with one value per expert")


def nvfp4_quantize_rowwise(
    z: Tensor,
    cu: Tensor,
    pts,
    q_out: Tensor,
    sf_out: Tensor,
    gather_idx: Tensor | None = None,
    rounding: str = "rn",
    seed: int = 0,
    sf_layout: str = "blocked",
    padded_offsets: Tensor | None = None,
):
    """Quantize expert-ordered rows, optionally gathering source rows from ``z``."""
    F = z.shape[1]
    assert F % 32 == 0 and z.stride(-1) == 1  # 16B store chunks
    assert sf_layout == "blocked" or F % 256 == 0  # linear: full-tile 8B rows
    E = cu.numel() - 1
    _check_padded_offsets(padded_offsets, cu, E)
    M = int(gather_idx.numel()) if gather_idx is not None else z.shape[0]
    pts2 = _prep(pts)
    n_tiles = -(-M // 128) + E
    NVFP4QuantKernel.compile(
        torch2cute_dtype_map[z.dtype],
        "row",
        gather_idx is not None,
        rounding,
        sf_layout,
        False,
        False,
        padded_offsets is not None,
        E,
    )(
        z,
        gather_idx,
        cu,
        padded_offsets,
        q_out.view(torch.uint8),
        sf_out.view(torch.uint8).view(-1),
        None,
        pts2,
        None,
        n_tiles,
        F,
        int(seed) & 0x7FFFFFFF,
    )


def nvfp4_quantize_colwise(
    z: Tensor,
    cu: Tensor,
    pts,
    q_out: Tensor,
    sf_out: Tensor,
    gather_idx: Tensor | None = None,
    rounding: str = "rn",
    seed: int = 0,
    rht: bool = False,
    amax_out: Tensor | None = None,
    padded_offsets: Tensor | None = None,
):
    """Colwise (token-axis) quantize for wgrad operands. z (rows, F); with
    gather_idx (M,), rows are read as z[gather_idx[...]] (the X variant - the
    one unavoidable high-precision gather read). q_out (F, mp_tot/2) uint8;
    sf_out e4m3 (flat view): per-expert independently-blocked SF segments
    concatenated - expert base F*off[e]/16 bytes, inside (F/128, K_e/64,
    32, 4, 4) atoms, off = cumsum(ceil(len/128))*128 (= the cudnn-frontend
    grouped-wgrad SF contract; each chunk is also a contiguous blocked SF
    tensor for per-expert GEMMs). pts REQUIRED. rht=True applies the TE
    16-token-group hadamard transform before block scaling (module
    docstring); pts must then come from the POST-RHT amax
    (nvfp4_rht_amax) - TE with_post_rht_amax semantics.

    amax_out ((>= n_tiles * F/128,) f32): emit one partial per CTA - the CTA
    max of the group amaxes this call quantized with (post-RHT when rht=True;
    bitwise the value the nvfp4_rht_amax pre-pass computes on the same
    tensor). Every launched CTA stores (empty tiles store 0), so the caller
    reduces amax_out[:n_tiles * F/128] with torch.amax, NO zero-fill needed.
    This is the delayed-col-amax source: reduce feeds the NEXT step's pts."""
    F = z.shape[1]
    assert F % 128 == 0 and z.stride(-1) == 1
    assert not rht or z.dtype == torch.bfloat16, "RHT is bf16-only"
    E = cu.numel() - 1
    _check_padded_offsets(padded_offsets, cu, E)
    M = int(gather_idx.numel()) if gather_idx is not None else z.shape[0]
    pts2 = _prep(pts)
    n_tiles = -(-M // 128) + E
    if amax_out is not None:
        assert amax_out.dtype == torch.float32 and amax_out.numel() >= n_tiles * (F // 128)
    NVFP4QuantKernel.compile(
        torch2cute_dtype_map[z.dtype],
        "col",
        gather_idx is not None,
        rounding,
        "blocked",
        rht,
        amax_out is not None,
        padded_offsets is not None,
        E,
    )(
        z,
        gather_idx,
        cu,
        padded_offsets,
        q_out.view(torch.uint8),
        sf_out.view(torch.uint8).view(-1),
        rht_matrix(z.device) if rht else None,
        pts2,
        amax_out,
        n_tiles,
        F,
        int(seed) & 0x7FFFFFFF,
    )
