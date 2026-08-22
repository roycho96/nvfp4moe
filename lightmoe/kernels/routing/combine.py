# Copyright (c) 2026, Sung Hyun Cho.
"""Deterministic MoE combine and backward gather kernels.

Forward accumulates each token's weighted expert rows in a fixed FP32 order.
Backward gathers and scales routed gradients without atomic reductions.
"""

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import Float32, Int32, const_expr, cute
from torch import Tensor

from .._common import (
    _bits_f32,
    fake_tensor,
    jit_cache,
    torch2cute_dtype_map,
    warp_sum,
)
from ..quantize.kernel import _ABSP, _HI16, _abs_f32, _max_bf16x2, _max_u32


class MoEFinalizeKernel:
    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        topk: int,
        tile_t: int = 8,
        num_threads: int = 256,
        n_frag: int = 1,
        weighted: bool = False,
        use_pdl: bool = False,
        direct: bool = False,
        broadcast_slots: bool = False,
    ):
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
        self.weighted = weighted
        self.use_pdl = use_pdl
        self.direct = direct
        self.broadcast_slots = broadcast_slots

    @cute.jit
    def __call__(
        self,
        mYw: cute.Tensor,  # (M, d) dtype, k-major
        mSlots: cute.Tensor,  # (T * topk,) Int32, -1 = dropped slot
        mOut: cute.Tensor,  # (T, d) dtype
        mWeights,  # Optional (M,) Float32
        stream: cuda.CUstream,
    ):
        T, d = mOut.shape[0], mOut.shape[1]
        self.kernel(mYw, mSlots, mOut, mWeights).launch(
            grid=[cute.ceil_div(T, self.tile_t), cute.ceil_div(d, self.cols_per_cta), 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(self, mYw: cute.Tensor, mSlots: cute.Tensor, mOut: cute.Tensor, mWeights):
        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        bx, by, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        T, d = mOut.shape[0], mOut.shape[1]
        yw_stride = mYw.stride[0]
        out_stride = mOut.stride[0]
        col0 = by * self.cols_per_cta + tidx * self.vec
        lane = tidx % cute.arch.WARP_SIZE
        lay_vec = cute.make_layout(self.vec)
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=128)
        # d % vec == 0 is asserted host-side, so col0 < d implies a full vector.
        if col0 < d:
            for it in cutlass.range_constexpr(self.tile_t):
                t = bx * self.tile_t + it
                if t < T:
                    acc = cute.make_rmem_tensor(lay_vec, Float32)
                    acc.fill(0.0)
                    # Two fragments overlap gathers without the register cost
                    # of keeping one fragment per top-k contribution.
                    if const_expr(self.direct):
                        frags = [
                            cute.make_rmem_tensor(lay_vec, self.dtype) for _ in range(self.n_frag)
                        ]
                        src = cute.make_tensor(
                            mYw.iterator + cute.assume(col0, divby=self.vec),
                            lay_vec,
                        )
                        cute.copy(atom, src, frags[0])
                        for j in cutlass.range_constexpr(self.topk):
                            if j + 1 < self.topk:
                                next_src = cute.make_tensor(
                                    mYw.iterator
                                    + cute.assume((j + 1) * yw_stride + col0, divby=self.vec),
                                    lay_vec,
                                )
                                cute.copy(atom, next_src, frags[(j + 1) % self.n_frag])
                            value = frags[j % self.n_frag].load().to(Float32)
                            weight = Float32(0.0)
                            if lane == 0:
                                weight = Float32(mWeights[j])
                            weight = cute.arch.shuffle_sync(weight, offset=0)
                            acc.store(acc.load() + value * weight)
                    elif const_expr(self.n_frag == 1):
                        frag = cute.make_rmem_tensor(lay_vec, self.dtype)
                        for j in cutlass.range_constexpr(self.topk):
                            if const_expr(self.broadcast_slots):
                                s = Int32(0)
                                if lane == 0:
                                    s = mSlots[t * self.topk + j]
                                s = cute.arch.shuffle_sync(s, offset=0)
                            else:
                                s = mSlots[t * self.topk + j]
                            if s >= 0:  # uniform across CTA: no divergence
                                # row strides are multiples of vec (host
                                # assert): 128-bit alignment is provable
                                src = cute.make_tensor(
                                    mYw.iterator
                                    + cute.assume(s * yw_stride + col0, divby=self.vec),
                                    lay_vec,
                                )
                                cute.copy(atom, src, frag)
                                value = frag.load().to(Float32)
                                if const_expr(self.weighted):
                                    weight = Float32(0.0)
                                    if lane == 0:
                                        weight = Float32(mWeights[s])
                                    weight = cute.arch.shuffle_sync(weight, offset=0)
                                    value = value * weight
                                acc.store(acc.load() + value)
                    else:
                        sl = cute.make_rmem_tensor(cute.make_layout(self.topk), Int32)
                        for j in cutlass.range_constexpr(self.topk):
                            if const_expr(self.broadcast_slots):
                                slot = Int32(0)
                                if lane == 0:
                                    slot = mSlots[t * self.topk + j]
                                sl[j] = cute.arch.shuffle_sync(slot, offset=0)
                            else:
                                sl[j] = mSlots[t * self.topk + j]
                        frags = [
                            cute.make_rmem_tensor(lay_vec, self.dtype) for _ in range(self.n_frag)
                        ]
                        first_slot = cutlass.max(sl[0], Int32(0))
                        srcd = cute.make_tensor(
                            mYw.iterator
                            + cute.assume(first_slot * yw_stride + col0, divby=self.vec),
                            lay_vec,
                        )
                        cute.copy(atom, srcd, frags[0])
                        for j in cutlass.range_constexpr(self.topk):
                            if j + 1 < self.topk:
                                next_slot = cutlass.max(sl[j + 1], Int32(0))
                                srce = cute.make_tensor(
                                    mYw.iterator
                                    + cute.assume(next_slot * yw_stride + col0, divby=self.vec),
                                    lay_vec,
                                )
                                cute.copy(atom, srce, frags[(j + 1) % self.n_frag])
                            value = frags[j % self.n_frag].load().to(Float32)
                            if const_expr(self.weighted):
                                safe_slot = cutlass.max(sl[j], Int32(0))
                                weight = Float32(0.0)
                                if lane == 0:
                                    weight = Float32(mWeights[safe_slot])
                                weight = cute.arch.shuffle_sync(weight, offset=0)
                                value = value * weight
                            if sl[j] >= 0:
                                acc.store(acc.load() + value)
                    out_frag = cute.make_rmem_tensor(lay_vec, self.dtype)
                    out_frag.store(acc.load().to(self.dtype))
                    dst = cute.make_tensor(
                        mOut.iterator + cute.assume(t * out_stride + col0, divby=self.vec),
                        lay_vec,
                    )
                    cute.copy(atom, out_frag, dst)

    @staticmethod
    @jit_cache
    def compile(
        dtype,
        topk,
        tile_t,
        num_threads,
        n_frag=1,
        weighted=False,
        use_pdl=False,
        direct=False,
        broadcast_slots=False,
    ):
        m_sym, t_sym, d_sym, tk_sym = (cute.sym_int() for _ in range(4))
        vec = 128 // dtype.width
        yw = fake_tensor(dtype, (m_sym, d_sym), vec)
        slots = None if direct else fake_tensor(Int32, (tk_sym,), 1)
        out = fake_tensor(dtype, (t_sym, d_sym), vec)
        weights = fake_tensor(Float32, (m_sym,), 1) if weighted else None
        return cute.compile(
            MoEFinalizeKernel(
                dtype,
                topk,
                tile_t=tile_t,
                num_threads=num_threads,
                n_frag=n_frag,
                weighted=weighted,
                use_pdl=use_pdl,
                direct=direct,
                broadcast_slots=broadcast_slots,
            ),
            yw,
            slots,
            out,
            weights,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )


def moe_finalize(
    yw: Tensor,
    slots: Tensor,
    out: Tensor,
    topk: int,
    tile_t: int = 8,
    n_frag: int = 1,
    weights: Tensor | None = None,
    use_pdl: bool = False,
    direct: bool = False,
    num_threads: int = 256,
    broadcast_slots: bool = False,
) -> None:
    """out[t] = sum over this token's (non-negative) slots of yw rows.

    yw (M, d) bf16/fp16 k-major; slots (T*topk,) int32 (-1 = dropped);
    out (T, d) same dtype as yw. Deterministic (fixed gather order, fp32
    register accumulation, single store). tile_t / n_frag are scheduling
    knobs only - the output is bitwise identical for any setting (the
    per-token fp32 add order never changes).
    """
    assert yw.stride(-1) == 1 and out.stride(-1) == 1
    if direct:
        assert out.shape[0] == 1 and yw.shape[0] == topk
        assert weights is not None and n_frag == 2
        slots_arg = None
    else:
        assert slots.dtype == torch.int32 and slots.numel() == out.shape[0] * topk
        slots_arg = slots
    if weights is not None:
        assert weights.dtype == torch.float32 and weights.numel() == yw.shape[0]
        assert weights.is_contiguous()
    dtype = torch2cute_dtype_map[yw.dtype]
    if num_threads not in (128, 256, 512, 768):
        raise ValueError("num_threads must be 128, 256, 512, or 768")
    vec = 128 // dtype.width
    assert out.shape[1] % vec == 0, f"d must be a multiple of {vec}"
    assert yw.stride(0) % vec == 0 and out.stride(0) % vec == 0, (
        "row strides must keep 128-bit vector alignment"
    )
    MoEFinalizeKernel.compile(
        dtype,
        topk,
        tile_t,
        num_threads,
        n_frag,
        weights is not None,
        use_pdl,
        direct,
        broadcast_slots,
    )(yw, slots_arg, out, weights)


class MoEFinalizeBwdKernel:
    """Gather routed gradients and optionally emit amax and router-dot partials.

    BF16 amax uses packed integer comparisons to avoid extra floating-point
    pressure. Dot products use a fixed lane, warp, and shared-memory fold order.
    """

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        tile_s: int = 8,
        num_threads: int = 256,
        emit_amax: bool = False,
        emit_dot: bool = False,
    ):
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
            grid=[cute.ceil_div(Mrows, self.tile_s), cute.ceil_div(d, self.cols_per_cta), 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mDY: cute.Tensor,
        mTok: cute.Tensor,
        mPs: cute.Tensor,
        mOut: cute.Tensor,
        mAmax,
        mYw,
        mDot,
    ):
        bx, by, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        Mrows, d = mOut.shape[0], mOut.shape[1]
        dy_stride = mDY.stride[0]
        out_stride = mOut.stride[0]
        col0 = by * self.cols_per_cta + tidx * self.vec
        lay_vec = cute.make_layout(self.vec)
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=128)
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
                Float32,
                cute.make_layout(n_warp * self.tile_s),
                byte_alignment=16,
            )
            # per-thread fp32 dot accumulators, one per slot in this tile
            # (defined for EVERY thread - idle columns contribute 0)
            dacc = cute.make_rmem_tensor(cute.make_layout(self.tile_s), Float32)
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
                            mYw.iterator + cute.assume(sslot * yw_stride + col0, divby=self.vec),
                            lay_vec,
                        )
                        cute.copy(atom, ysrc, ywf)
                        for v in cutlass.range_constexpr(self.vec):
                            dacc[it] = dacc[it] + (ywf[v].to(Float32) * frag[v].to(Float32))
                    if const_expr(self.emit_amax):
                        # amax of the ROUNDED stores (what aminmax re-read)
                        if const_expr(self.packed):
                            # abs-masked bf16x2 tree over the raw store bits
                            # (bitwise the f32 chain below; int pipe, 2/op)
                            for v2 in cutlass.range_constexpr(self.vec // 2):
                                loc_pk = _max_bf16x2(loc_pk, outu[v2] & cutlass.Uint32(_ABSP))
                        else:
                            for v in cutlass.range_constexpr(self.vec):
                                av = outf[v].to(Float32)
                                loc_amax = cutlass.max(loc_amax, cutlass.max(av, -av))
                    dst = cute.make_tensor(
                        mOut.iterator + cute.assume(sslot * out_stride + col0, divby=self.vec),
                        lay_vec,
                    )
                    cute.copy(atom, outf, dst)
        if const_expr(self.emit_amax):
            if const_expr(self.packed):
                # Non-negative FP32 and uint32 ordering match, so one integer
                # max combines the two absolute BF16 halves exactly.
                loc_amax = _bits_f32(_max_u32(loc_pk << 16, loc_pk & cutlass.Uint32(_HI16)))
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
                wsum = warp_sum(dacc[it])
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
            MoEFinalizeBwdKernel(
                dtype,
                tile_s=tile_s,
                num_threads=num_threads,
                emit_amax=emit_amax,
                emit_dot=emit_dot,
            ),
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


def moe_finalize_bwd(
    dY: Tensor,
    tok: Tensor,
    probs: Tensor,
    out: Tensor,
    amax_out: Tensor | None = None,
    yw: Tensor | None = None,
    dot_out: Tensor | None = None,
) -> None:
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
        assert dot_out.dtype == torch.float32 and dot_out.numel() >= nd * out.shape[0]
    else:
        assert dot_out is None
    MoEFinalizeBwdKernel.compile(dtype, 8, 256, amax_out is not None, yw is not None)(
        dY, tok, probs, out, amax_out, yw, dot_out
    )
