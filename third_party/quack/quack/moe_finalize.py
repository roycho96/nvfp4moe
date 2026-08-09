# Copyright (c) 2026, Sung Hyun Cho.
"""Deterministic MoE finalize: gather-reduce of top-k weighted expert outputs.

    out[t, :] = sum_j yw[slots[t * topk + j], :]        (slots[p] < 0 skipped)

``yw`` is the permuted, ALREADY top-k-weighted expert output (the GEMM2
epilogue multiplies the routing probability and stores bf16), ``slots`` the
inverse of the router's expert-major argsort (`inv[order] = arange`; -1 marks
slots dropped by capacity). Each output token gathers its k contributions in a
fixed j order and accumulates in fp32 registers, storing bf16 once — so the
result is bitwise deterministic run-to-run (no atomics, no cross-CTA order),
unlike scatter-based finalizes (torch index_add / the atomic fused-finalize
epilogues), and the kernel is shape-static under capacity drop (slots keeps
the (T * topk,) extent, dropped entries are -1).

Pure streaming: reads M*d*2 B + writes T*d*2 B; the right ceiling is HBM
bandwidth.
"""

import math
import operator

import torch
from torch import Tensor

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, const_expr

from quack.cache import jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.nvfp4_quant import _ABSP, _HI16, _abs_f32, _max_bf16x2, _max_u32
from quack.reduce import warp_reduce
from quack.rounding import _bits_f32


class MoEFinalizeKernel:
    def __init__(self, dtype: type[cutlass.Numeric], topk: int, tile_t: int = 8,
                 num_threads: int = 256, n_frag: int = 1):
        self.dtype = dtype
        self.topk = topk
        self.tile_t = tile_t
        self.num_threads = num_threads
        self.vec = 128 // dtype.width  # one 128-bit vector per thread
        self.cols_per_cta = num_threads * self.vec
        # n_frag >= 2: software pipeline the k gathers - slot j+1's load is
        # issued (into the other fragment) BEFORE slot j's accumulate, so one
        # extra load is in flight per thread across the branchy j loop. The
        # fp32 accumulation ORDER is unchanged (still ascending j), so the
        # output is bitwise identical for any n_frag.
        assert n_frag in (1, 2)
        self.n_frag = n_frag

    @cute.jit
    def __call__(
        self,
        mYw: cute.Tensor,  # (M, d) dtype, k-major
        mSlots: cute.Tensor,  # (T * topk,) Int32, -1 = dropped slot
        mOut: cute.Tensor,  # (T, d) dtype
        stream: cuda.CUstream,
    ):
        T, d = mOut.shape[0], mOut.shape[1]
        self.kernel(mYw, mSlots, mOut).launch(
            grid=[cute.ceil_div(T, self.tile_t), cute.ceil_div(d, self.cols_per_cta), 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mYw: cute.Tensor, mSlots: cute.Tensor, mOut: cute.Tensor):
        bx, by, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        T, d = mOut.shape[0], mOut.shape[1]
        yw_stride = mYw.stride[0]
        out_stride = mOut.stride[0]
        col0 = by * self.cols_per_cta + tidx * self.vec
        lay_vec = cute.make_layout(self.vec)
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=128
        )
        # d % vec == 0 is asserted host-side, so col0 < d implies a full vector.
        if col0 < d:
            for it in cutlass.range_constexpr(self.tile_t):
                t = bx * self.tile_t + it
                if t < T:
                    acc = cute.make_rmem_tensor(lay_vec, Float32)
                    acc.fill(0.0)
                    # NOTE(measured, do not "fix"): a per-contribution fragment
                    # array (k frags to remove the WAR reuse hazard and let all
                    # k loads fly) was TRIED and is SLOWER - 70.2us vs 55.8us
                    # on B200 (register pressure beats the extra MLP). The old
                    # "70% DRAM = gather page locality" claim is REFUTED: a
                    # perfect-locality slots=arange control runs the same time
                    # at the same DRAM% (Q19 ncu); the wall is in-flight bytes
                    # (1 outstanding 16B load/thread, long_scoreboard ~40).
                    # n_frag == 2 is the depth-2 middle point: slots preloaded,
                    # load j+1 issued before add j (same fp32 add order); with
                    # tile_t=4 (occupancy intact) it wins where deeper unrolls
                    # (tile_t=16, 8 frags) lose to register pressure.
                    if const_expr(self.n_frag == 1):
                        frag = cute.make_rmem_tensor(lay_vec, self.dtype)
                        for j in cutlass.range_constexpr(self.topk):
                            s = mSlots[t * self.topk + j]
                            if s >= 0:  # uniform across CTA: no divergence
                                # row strides are multiples of vec (host
                                # assert): 128-bit alignment is provable
                                src = cute.make_tensor(
                                    mYw.iterator
                                    + cute.assume(s * yw_stride + col0,
                                                  divby=self.vec),
                                    lay_vec,
                                )
                                cute.copy(atom, src, frag)
                                acc.store(acc.load()
                                          + frag.load().to(Float32))
                    else:
                        sl = cute.make_rmem_tensor(
                            cute.make_layout(self.topk), Int32)
                        for j in cutlass.range_constexpr(self.topk):
                            sl[j] = mSlots[t * self.topk + j]
                        frags = [cute.make_rmem_tensor(lay_vec, self.dtype)
                                 for _ in range(self.n_frag)]
                        if sl[0] >= 0:
                            srcd = cute.make_tensor(
                                mYw.iterator
                                + cute.assume(sl[0] * yw_stride + col0,
                                              divby=self.vec),
                                lay_vec,
                            )
                            cute.copy(atom, srcd, frags[0])
                        for j in cutlass.range_constexpr(self.topk):
                            if j + 1 < self.topk:
                                if sl[j + 1] >= 0:
                                    srce = cute.make_tensor(
                                        mYw.iterator
                                        + cute.assume(
                                            sl[j + 1] * yw_stride + col0,
                                            divby=self.vec),
                                        lay_vec,
                                    )
                                    cute.copy(
                                        atom, srce,
                                        frags[(j + 1) % self.n_frag])
                            if sl[j] >= 0:
                                acc.store(
                                    acc.load()
                                    + frags[j % self.n_frag].load()
                                    .to(Float32))
                    out_frag = cute.make_rmem_tensor(lay_vec, self.dtype)
                    out_frag.store(acc.load().to(self.dtype))
                    dst = cute.make_tensor(
                        mOut.iterator + cute.assume(t * out_stride + col0, divby=self.vec),
                        lay_vec,
                    )
                    cute.copy(atom, out_frag, dst)

    @staticmethod
    @jit_cache
    def compile(dtype, topk, tile_t, num_threads, n_frag=1):
        m_sym, t_sym, d_sym, tk_sym = (cute.sym_int() for _ in range(4))
        vec = 128 // dtype.width
        yw = fake_tensor(dtype, (m_sym, d_sym), vec)
        slots = fake_tensor(Int32, (tk_sym,), 1)
        out = fake_tensor(dtype, (t_sym, d_sym), vec)
        return cute.compile(
            MoEFinalizeKernel(dtype, topk, tile_t=tile_t,
                              num_threads=num_threads, n_frag=n_frag),
            yw,
            slots,
            out,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )


def moe_finalize(yw: Tensor, slots: Tensor, out: Tensor, topk: int,
                 tile_t: int = 8, n_frag: int = 1) -> None:
    """out[t] = sum over this token's (non-negative) slots of yw rows.

    yw (M, d) bf16/fp16 k-major; slots (T*topk,) int32 (-1 = dropped);
    out (T, d) same dtype as yw. Deterministic (fixed gather order, fp32
    register accumulation, single store). tile_t / n_frag are scheduling
    knobs only - the output is bitwise identical for any setting (the
    per-token fp32 add order never changes).
    """
    assert yw.stride(-1) == 1 and out.stride(-1) == 1
    assert slots.dtype == torch.int32 and slots.numel() == out.shape[0] * topk
    dtype = torch2cute_dtype_map[yw.dtype]
    vec = 128 // dtype.width
    assert out.shape[1] % vec == 0, f"d must be a multiple of {vec}"
    assert yw.stride(0) % vec == 0 and out.stride(0) % vec == 0, (
        "row strides must keep 128-bit vector alignment"
    )
    MoEFinalizeKernel.compile(dtype, topk, tile_t, 256, n_frag)(yw, slots, out)


class MoEFinalizeBwdKernel:
    """finalize backward: dY_M[s, :] = probs[s] * dY[tok[s], :].

    Pure per-slot gather with a per-row scale - no reduction, no atomics,
    trivially deterministic. tok is the forward gather_idx (slot -> source
    token); probs the sorted routing weights. Replaces the torch
    index-select + fp32 mul pair (12.8% of the Q8 bwd step GPU).

    emit_amax: fold each thread's |stored bf16 value| max and tree-reduce to
    ONE f32 partial per CTA - amax over exactly the values this kernel
    stores, so torch.amax over the partials is bitwise the aminmax the layer
    used to run as a separate full read of dY_M (max is exact, any order).

    For bf16 the fold is an abs-masked max.bf16x2 tree over the store
    fragment's raw bits (2 elements per int-pipe op, one final u32
    halves-merge) - bitwise the per-element XU cvt + 2 f32 max chain it
    replaces (comparisons do not round; non-negative finite f32 bit order ==
    unsigned int order; bf16 -> f32 as bits << 16 is exact), and NOT free to
    skip: Q18 measured the f32 chain at +16.4us (+25.8%) on this DRAM-bound
    kernel - the ALU did not hide under the loads. Other dtypes keep the f32
    chain.

    emit_dot: additionally read the forward's yw (the SAME (M, d) row
    addresses this kernel writes) and emit per-slot fp32 dot partials
    dot[s] = <yw[s], dY[tok[s]]> - the routing-weight gradient's numerator
    (dprobs[s] = dot[s] / probs[s], since yw = bf16(probs * y2)). The dY
    fragment is already in registers for the main store, so the feature
    costs exactly one extra (M, d) read. Reduction order is FIXED (lane
    fold in ascending v, warp butterfly, lane-0 store, warp-ascending smem
    fold), no atomics - bitwise deterministic run-to-run. Partial layout:
    mDot[cta_col * M + s], caller folds column tiles with torch.sum (fixed
    shape, deterministic) - one entry per (slot, d-tile)."""

    def __init__(self, dtype: type[cutlass.Numeric], tile_s: int = 8,
                 num_threads: int = 256, emit_amax: bool = False,
                 emit_dot: bool = False):
        self.dtype = dtype
        self.tile_s = tile_s
        self.num_threads = num_threads
        self.vec = 128 // dtype.width
        self.cols_per_cta = num_threads * self.vec
        assert not emit_amax or num_threads == 256  # reduce tree below
        assert num_threads % 32 == 0
        self.emit_amax = emit_amax
        self.emit_dot = emit_dot
        # the bf16x2 amax trick is bf16-only; other dtypes keep the f32 chain
        self.packed = dtype is cutlass.BFloat16

    @cute.jit
    def __call__(
        self,
        mDY: cute.Tensor,  # (T, d) dtype
        mTok: cute.Tensor,  # (M,) Int32
        mPs: cute.Tensor,  # (M,) Float32
        mOut: cute.Tensor,  # (M, d) dtype
        mAmax,  # Optional (>= grid_x * grid_y,) Float32 per-CTA partials
        mYw,  # Optional (M, d) dtype - the forward's stored yw (emit_dot)
        mDot,  # Optional (>= grid_y * M,) Float32 per-(slot, d-tile) partials
        stream: cuda.CUstream,
    ):
        Mrows, d = mOut.shape[0], mOut.shape[1]
        self.kernel(mDY, mTok, mPs, mOut, mAmax, mYw, mDot).launch(
            grid=[cute.ceil_div(Mrows, self.tile_s),
                  cute.ceil_div(d, self.cols_per_cta), 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mDY: cute.Tensor, mTok: cute.Tensor, mPs: cute.Tensor,
               mOut: cute.Tensor, mAmax, mYw, mDot):
        bx, by, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        Mrows, d = mOut.shape[0], mOut.shape[1]
        dy_stride = mDY.stride[0]
        out_stride = mOut.stride[0]
        col0 = by * self.cols_per_cta + tidx * self.vec
        lay_vec = cute.make_layout(self.vec)
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=128
        )
        smem = cutlass.utils.SmemAllocator()
        n_warp = self.num_threads // 32
        loc_amax = Float32(0.0)
        loc_pk = cutlass.Uint32(0)
        if const_expr(self.emit_amax):
            sRed = smem.allocate_tensor(
                Float32, cute.make_layout(self.num_threads), byte_alignment=16
            )
        if const_expr(self.emit_dot):
            sDot = smem.allocate_tensor(
                Float32, cute.make_layout(n_warp * self.tile_s),
                byte_alignment=16,
            )
            # per-thread fp32 dot accumulators, one per slot in this tile
            # (defined for EVERY thread - idle columns contribute 0)
            dacc = cute.make_rmem_tensor(
                cute.make_layout(self.tile_s), Float32)
            dacc.fill(0.0)
        if col0 < d:
            frag = cute.make_rmem_tensor(lay_vec, self.dtype)
            outf = cute.make_rmem_tensor(lay_vec, self.dtype)
            if const_expr(self.emit_dot):
                ywf = cute.make_rmem_tensor(lay_vec, self.dtype)
                yw_stride = mYw.stride[0]
            # raw-bits pair view of the store fragment (packed emit path)
            outu = cute.make_tensor(
                cute.recast_ptr(outf.iterator, dtype=cutlass.Uint32),
                cute.make_layout(self.vec // 2),
            )
            for it in cutlass.range_constexpr(self.tile_s):
                sslot = bx * self.tile_s + it
                if sslot < Mrows:
                    tok = mTok[sslot]
                    ps = Float32(mPs[sslot])
                    src = cute.make_tensor(
                        mDY.iterator + cute.assume(tok * dy_stride + col0, divby=self.vec),
                        lay_vec,
                    )
                    cute.copy(atom, src, frag)
                    outf.store((frag.load().to(Float32) * ps).to(self.dtype))
                    if const_expr(self.emit_dot):
                        # dot with the RAW dY fragment (already loaded for
                        # the store) in fixed ascending lane order
                        ysrc = cute.make_tensor(
                            mYw.iterator
                            + cute.assume(sslot * yw_stride + col0,
                                          divby=self.vec),
                            lay_vec,
                        )
                        cute.copy(atom, ysrc, ywf)
                        for v in cutlass.range_constexpr(self.vec):
                            dacc[it] = dacc[it] + (
                                ywf[v].to(Float32) * frag[v].to(Float32))
                    if const_expr(self.emit_amax):
                        # amax of the ROUNDED stores (what aminmax re-read)
                        if const_expr(self.packed):
                            # abs-masked bf16x2 tree over the raw store bits
                            # (bitwise the f32 chain below; int pipe, 2/op)
                            for v2 in cutlass.range_constexpr(self.vec // 2):
                                loc_pk = _max_bf16x2(
                                    loc_pk,
                                    outu[v2] & cutlass.Uint32(_ABSP))
                        else:
                            for v in cutlass.range_constexpr(self.vec):
                                av = outf[v].to(Float32)
                                loc_amax = cutlass.max(
                                    loc_amax, cutlass.max(av, -av))
                    dst = cute.make_tensor(
                        mOut.iterator
                        + cute.assume(sslot * out_stride + col0, divby=self.vec),
                        lay_vec,
                    )
                    cute.copy(atom, outf, dst)
        if const_expr(self.emit_amax):
            if const_expr(self.packed):
                # unpack: each abs bf16 half's f32 bits are half << 16;
                # non-negative f32 order == u32 order, so one integer max
                # merges the halves exactly (the Q17 rht_amax idiom)
                loc_amax = _bits_f32(_max_u32(
                    loc_pk << 16, loc_pk & cutlass.Uint32(_HI16)))
            # unconditional per-CTA store (idle threads/tiles contribute 0):
            # the caller reduces the exact grid slice, no zero-fill. abs
            # clears a possible -0.0 from the max(x,-x) chain (PTX max.f32
            # zero-sign is undefined; aminmax yields +0.0).
            sRed[tidx] = loc_amax
            cute.arch.sync_threads()
            for off in (128, 64, 32, 16, 8, 4, 2, 1):
                if tidx < off:
                    sRed[tidx] = cutlass.max(sRed[tidx], sRed[tidx + off])
                cute.arch.sync_threads()
            if tidx == 0:
                gy = cute.ceil_div(d, self.cols_per_cta)
                mAmax[bx * gy + by] = _abs_f32(sRed[0])
        if const_expr(self.emit_dot):
            # fixed-order per-slot fold: warp butterfly (fixed shuffle
            # schedule), lane-0 stores, then one thread per slot sums the
            # warp partials in ascending warp order - deterministic, no
            # atomics. fp32 add is not associative, but the ORDER is static
            # so the result is bitwise reproducible run-to-run.
            wi = tidx // 32
            lane = tidx % 32
            for it in cutlass.range_constexpr(self.tile_s):
                wsum = warp_reduce(dacc[it], operator.add, dtype=Float32)
                if lane == 0:
                    sDot[wi * self.tile_s + it] = wsum
            cute.arch.sync_threads()
            if tidx < self.tile_s:
                tot = Float32(0.0)
                for w in cutlass.range_constexpr(n_warp):
                    tot = tot + sDot[w * self.tile_s + tidx]
                dslot = bx * self.tile_s + tidx
                if dslot < Mrows:
                    mDot[by * Mrows + dslot] = tot

    @staticmethod
    @jit_cache
    def compile(dtype, tile_s, num_threads, emit_amax=False, emit_dot=False):
        t_sym, m_sym, d_sym, a_sym, dt_sym = (cute.sym_int() for _ in range(5))
        vec = 128 // dtype.width
        dy = fake_tensor(dtype, (t_sym, d_sym), vec)
        tok = fake_tensor(Int32, (m_sym,), 1)
        ps = fake_tensor(Float32, (m_sym,), 1)
        out = fake_tensor(dtype, (m_sym, d_sym), vec)
        amax = fake_tensor(Float32, (a_sym,), 1) if emit_amax else None
        yw = fake_tensor(dtype, (m_sym, d_sym), vec) if emit_dot else None
        dot = fake_tensor(Float32, (dt_sym,), 1) if emit_dot else None
        return cute.compile(
            MoEFinalizeBwdKernel(dtype, tile_s=tile_s, num_threads=num_threads,
                                 emit_amax=emit_amax, emit_dot=emit_dot),
            dy,
            tok,
            ps,
            out,
            amax,
            yw,
            dot,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )


def moe_finalize_bwd(dY: Tensor, tok: Tensor, probs: Tensor, out: Tensor,
                     amax_out: Tensor | None = None,
                     yw: Tensor | None = None,
                     dot_out: Tensor | None = None) -> None:
    """out[s] = probs[s] * dY[tok[s]] (bf16 store, fp32 scale). Deterministic
    (pure gather, no reduction). amax_out ((>= ceil(M/8)*ceil(d/2048),) f32):
    emit per-CTA partials of amax(|out|); torch.amax over the exact grid
    slice is bitwise the aminmax of the stored tensor.

    yw + dot_out ((>= ceil(d/2048) * M,) f32): additionally emit per-slot
    fp32 dot partials dot_out[ct * M + s] = <yw[s], dY[tok[s]]> over this
    kernel's column tile ct (the routing-weight gradient numerator; the dY
    fragment is already loaded, so the cost is one extra read of yw).
    Deterministic: fixed lane/warp/smem fold order, no atomics. Caller folds
    the ceil(d/2048) column tiles with torch.sum and divides by probs."""
    assert dY.stride(-1) == 1 and out.stride(-1) == 1
    assert tok.dtype == torch.int32 and probs.dtype == torch.float32
    dtype = torch2cute_dtype_map[dY.dtype]
    vec = 128 // dtype.width
    assert out.shape[1] % vec == 0 and dY.stride(0) % vec == 0 and out.stride(0) % vec == 0
    if amax_out is not None:
        k = MoEFinalizeBwdKernel(dtype, 8, 256)
        npart = -(-out.shape[0] // 8) * (-(-out.shape[1] // k.cols_per_cta))
        assert amax_out.dtype == torch.float32 and amax_out.numel() >= npart
    if yw is not None:
        assert dot_out is not None, "emit_dot needs the partial buffer"
        assert yw.dtype == dY.dtype and tuple(yw.shape) == tuple(out.shape)
        assert yw.stride(-1) == 1 and yw.stride(0) % vec == 0
        k = MoEFinalizeBwdKernel(dtype, 8, 256)
        nd = -(-out.shape[1] // k.cols_per_cta)
        assert (dot_out.dtype == torch.float32
                and dot_out.numel() >= nd * out.shape[0])
    else:
        assert dot_out is None
    MoEFinalizeBwdKernel.compile(dtype, 8, 256, amax_out is not None,
                                 yw is not None)(
        dY, tok, probs, out, amax_out, yw, dot_out)
