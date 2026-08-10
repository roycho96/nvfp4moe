# Copyright (c) 2026, Sung Hyun Cho.
"""Stable expert-major dispatch for preselected router outputs.

The first kernel counts routing pairs per chunk and expert. The second emits
the stable permutation, offsets, probabilities, and inverse slots without
atomics. Router selection remains outside this module.
"""

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import Float32, Int32, cute
from torch import Tensor

from ._common import fake_tensor, jit_cache

CHUNK = 1024  # Keeps the count table within the SM100 shared-memory budget.
NUM_THREADS = 256
B_MAX = 128


class MoEDispatchKernel:
    def __init__(self, E: int, topk: int):
        assert E <= NUM_THREADS, "one counting thread per expert"
        assert CHUNK * 4 + B_MAX * E * 4 + E * 4 <= 200 * 1024, "smem budget"
        self.E = E
        self.topk = topk

    @cute.jit
    def __call__(
        self,
        mTopI: cute.Tensor,  # (M,) Int32 flat expert ids (router output)
        mTopV: cute.Tensor,  # (M,) Float32 flat routing probs
        mPart: cute.Tensor,  # (B_MAX * E,) Int32 scratch: per-(chunk, expert) counts
        mGi: cute.Tensor,  # (M,) Int32 out
        mCu: cute.Tensor,  # (E + 1,) Int32 out
        mPs: cute.Tensor,  # (M,) Float32 out
        mSlots: cute.Tensor,  # (M,) Int32 out
        mOffPad: cute.Tensor,  # (E,) Int32 out: cumsum(ceil(count/128)*128)
        B: Int32,  # ceil(M / CHUNK)
        stream: cuda.CUstream,
    ):
        self.kernel_hist(mTopI, mPart).launch(
            grid=[B, 1, 1], block=[NUM_THREADS, 1, 1], stream=stream
        )
        self.kernel_scatter(mTopI, mTopV, mPart, mGi, mCu, mPs, mSlots, mOffPad, B).launch(
            grid=[B, 1, 1], block=[NUM_THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel_hist(self, mTopI: cute.Tensor, mPart: cute.Tensor):
        b, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        M = mTopI.shape[0]
        smem = cutlass.utils.SmemAllocator()
        sChunk = smem.allocate_tensor(Int32, cute.make_layout(CHUNK), byte_alignment=16)
        for i in cutlass.range_constexpr(CHUNK // NUM_THREADS):
            j = tidx + i * NUM_THREADS
            p = b * CHUNK + j
            v = Int32(-1)  # pad: matches no expert
            if p < M:
                v = mTopI[p]
            sChunk[j] = v
        cute.arch.sync_threads()
        if tidx < self.E:
            cnt = Int32(0)
            for j in cutlass.range(CHUNK):
                if sChunk[j] == tidx:
                    cnt += 1
            mPart[b * self.E + tidx] = cnt

    @cute.kernel
    def kernel_scatter(
        self,
        mTopI: cute.Tensor,
        mTopV: cute.Tensor,
        mPart: cute.Tensor,
        mGi: cute.Tensor,
        mCu: cute.Tensor,
        mPs: cute.Tensor,
        mSlots: cute.Tensor,
        mOffPad: cute.Tensor,
        B: Int32,
    ):
        b, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        M = mTopI.shape[0]
        smem = cutlass.utils.SmemAllocator()
        sChunk = smem.allocate_tensor(Int32, cute.make_layout(CHUNK), byte_alignment=16)
        sPart = smem.allocate_tensor(Int32, cute.make_layout(B_MAX * self.E), byte_alignment=16)
        sTot = smem.allocate_tensor(Int32, cute.make_layout(self.E), byte_alignment=16)
        # stage this CTA's chunk + the full part matrix
        for i in cutlass.range_constexpr(CHUNK // NUM_THREADS):
            j = tidx + i * NUM_THREADS
            p = b * CHUNK + j
            v = Int32(-1)
            if p < M:
                v = mTopI[p]
            sChunk[j] = v
        n_part = B * self.E
        n_iters = (n_part + NUM_THREADS - 1) // NUM_THREADS
        for i in cutlass.range(n_iters):
            j = tidx + i * NUM_THREADS
            if j < n_part:
                sPart[j] = mPart[j]
        cute.arch.sync_threads()
        if tidx < self.E:
            tot = Int32(0)
            for b2 in cutlass.range(B):
                tot += sPart[b2 * self.E + tidx]
            sTot[tidx] = tot
        cute.arch.sync_threads()
        if tidx < self.E:
            e = tidx
            # exclusive prefix over experts (fixed ascending order) = cu[e];
            # inclusive 128-padded prefix = off_pad[e] (grouped-wgrad contract)
            cu_e = Int32(0)
            op_e = Int32(0)
            for e2 in cutlass.range_constexpr(self.E):
                if e2 < e:
                    cu_e += sTot[e2]
                if e2 <= e:
                    op_e += ((sTot[e2] + 127) // 128) * 128
            if b == 0:
                mCu[e + 1] = cu_e + sTot[e]
                mOffPad[e] = op_e
                if e == 0:
                    mCu[0] = 0
            # + counts of expert e in earlier chunks = this chunk's rank base
            base = cu_e
            for b2 in cutlass.range(B):
                if b2 < b:
                    base += sPart[b2 * self.E + e]
            # stable scatter: walk the chunk in flat order (pads are -1)
            cnt = Int32(0)
            for j in cutlass.range(CHUNK):
                if sChunk[j] == e:
                    p = b * CHUNK + j
                    s = base + cnt
                    mGi[s] = p // self.topk
                    mPs[s] = Float32(mTopV[p])
                    mSlots[p] = s
                    cnt += 1

    @staticmethod
    @jit_cache
    def compile(E, topk):
        m1, m2, m3, m4, m5, c1, p1, e1 = (cute.sym_int() for _ in range(8))
        topi = fake_tensor(Int32, (m1,), 1)
        topv = fake_tensor(Float32, (m2,), 1)
        part = fake_tensor(Int32, (p1,), 1)
        gi = fake_tensor(Int32, (m3,), 1)
        cu = fake_tensor(Int32, (c1,), 1)
        ps = fake_tensor(Float32, (m4,), 1)
        slots = fake_tensor(Int32, (m5,), 1)
        off_pad = fake_tensor(Int32, (e1,), 1)
        return cute.compile(
            MoEDispatchKernel(E, topk),
            topi,
            topv,
            part,
            gi,
            cu,
            ps,
            slots,
            off_pad,
            Int32(1),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )


def moe_dispatch(
    topi: Tensor,
    topv: Tensor,
    E: int,
    gi: Tensor,
    cu: Tensor,
    ps: Tensor,
    slots: Tensor,
    part: Tensor,
    off_pad: Tensor,
) -> None:
    """Fill (gi, cu, ps, slots, off_pad) from router (topi, topv); see module
    docstring. off_pad (E,) int32 = cumsum(ceil(count/128)*128), the padded
    grouped-wgrad offsets contract (emitted here so the layer's bwd needs no
    torch chain for it).

    topi (T, k) int32 contiguous; topv (T, k) float32 contiguous; gi/slots (T*k,)
    int32; cu (E+1,) int32; ps (T*k,) float32; part (B_MAX*E,) int32 scratch.
    Deterministic and bitwise equal to the stable-argsort torch chain.
    """
    assert topi.dtype == torch.int32 and topi.is_contiguous()
    assert topv.dtype == torch.float32 and topv.is_contiguous()
    assert topi.dim() == 2 and topv.shape == topi.shape
    T, k = topi.shape
    M = T * k
    B = -(-M // CHUNK)
    assert B <= B_MAX, f"M={M} exceeds the {B_MAX * CHUNK} dispatch bound"
    assert gi.numel() == M and slots.numel() == M and ps.numel() == M
    assert cu.numel() == E + 1 and part.numel() >= B * E
    assert off_pad.numel() == E and off_pad.dtype == torch.int32
    MoEDispatchKernel.compile(E, k)(
        topi.view(-1), topv.view(-1), part, gi, cu, ps, slots, off_pad, B
    )


class MoEDispatch:
    """Preallocated expert-major dispatch buffers."""

    def __init__(self, T: int, E: int, k: int, device="cuda"):
        M = T * k
        self.T, self.E, self.k = T, E, k
        self.gi = torch.empty(M, dtype=torch.int32, device=device)
        self.cu = torch.empty(E + 1, dtype=torch.int32, device=device)
        self.ps = torch.empty(M, dtype=torch.float32, device=device)
        self.slots = torch.empty(M, dtype=torch.int32, device=device)
        self.part = torch.empty(B_MAX * E, dtype=torch.int32, device=device)
        self.off_pad = torch.empty(E, dtype=torch.int32, device=device)

    def __call__(self, topi: torch.Tensor, topv: torch.Tensor):
        assert topi.shape == (self.T, self.k)
        probs = topv.float() if topv.dtype != torch.float32 else topv
        moe_dispatch(
            topi,
            probs,
            self.E,
            self.gi,
            self.cu,
            self.ps,
            self.slots,
            self.part,
            self.off_pad,
        )
        return self.gi, self.cu, self.ps, self.slots

    def differentiable_probs(self, topv: torch.Tensor) -> torch.Tensor:
        """Return expert-major probabilities while preserving router gradients."""
        assert topv.shape == (self.T, self.k)
        return torch.zeros_like(self.ps).index_put((self.slots.long(),), topv.reshape(-1).float())


__all__ = ["MoEDispatch", "MoEDispatchKernel", "moe_dispatch"]
