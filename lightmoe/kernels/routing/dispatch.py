# Copyright (c) 2026, Sung Hyun Cho.
"""Stable expert-major dispatch for preselected router outputs.

The first kernel counts routing pairs per chunk and expert. The second emits
the stable permutation, offsets, probabilities, and inverse slots without
atomics. Router selection remains outside this module.
"""

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import Float32, Int32, const_expr, cute
from torch import Tensor

from .._common import fake_tensor, jit_cache

CHUNK = 1024  # Keeps the count table within the SM100 shared-memory budget.
NUM_THREADS = 256
B_MAX = 128


class MoEDispatchKernel:
    def __init__(
        self,
        E: int,
        topk: int,
        small: bool = False,
        use_pdl: bool = True,
        gather_experts: bool = False,
        plan_tile_rows: int = 128,
    ):
        assert E <= NUM_THREADS, "one counting thread per expert"
        assert CHUNK * 4 + B_MAX * E * 4 + E * 4 <= 200 * 1024, "smem budget"
        self.E = E
        self.topk = topk
        self.small = small
        self.use_pdl = use_pdl
        self.gather_experts = gather_experts
        self.plan_tile_rows = plan_tile_rows
        self.warp_dispatch = E <= NUM_THREADS

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
        if const_expr(self.small):
            self.kernel_small(mTopI, mTopV, mPart, mGi, mCu, mPs, mSlots, mOffPad).launch(
                grid=[1, 1, 1],
                block=[NUM_THREADS, 1, 1],
                stream=stream,
                use_pdl=self.use_pdl,
            )
        else:
            self.kernel_hist(mTopI, mPart).launch(
                grid=[B, 1, 1],
                block=[NUM_THREADS, 1, 1],
                stream=stream,
                use_pdl=self.use_pdl,
            )
            self.kernel_scatter(mTopI, mTopV, mPart, mGi, mCu, mPs, mSlots, mOffPad, B).launch(
                grid=[B, 1, 1],
                block=[NUM_THREADS, 1, 1],
                stream=stream,
                use_pdl=self.use_pdl,
            )

    @cute.kernel
    def kernel_small(
        self,
        mTopI: cute.Tensor,
        mTopV: cute.Tensor,
        mPlan: cute.Tensor,
        mGi: cute.Tensor,
        mCu: cute.Tensor,
        mPs: cute.Tensor,
        mSlots: cute.Tensor,
        mOffPad: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        M = mTopI.shape[0]
        smem = cutlass.utils.SmemAllocator()
        sChunk = smem.allocate_tensor(Int32, cute.make_layout(CHUNK), byte_alignment=16)
        sBase = smem.allocate_tensor(Int32, cute.make_layout(self.E), byte_alignment=16)
        sTot = smem.allocate_tensor(Int32, cute.make_layout(self.E), byte_alignment=16)
        sPad = smem.allocate_tensor(Int32, cute.make_layout(self.E), byte_alignment=16)
        sPlan = smem.allocate_tensor(Int32, cute.make_layout(self.E), byte_alignment=16)
        sSeg = smem.allocate_tensor(
            Int32,
            cute.make_layout((CHUNK // 32, self.E), stride=(self.E, 1)),
            byte_alignment=16,
        )
        segment_items = (CHUNK // 32) * self.E
        segment_iters = (segment_items + NUM_THREADS - 1) // NUM_THREADS
        for i in cutlass.range_constexpr(segment_iters):
            linear = tidx + i * NUM_THREADS
            if linear < segment_items:
                sSeg[linear // self.E, linear % self.E] = 0
        for i in cutlass.range_constexpr(CHUNK // NUM_THREADS):
            j = tidx + i * NUM_THREADS
            value = Int32(-1)
            if j < M:
                value = mTopI[j]
            sChunk[j] = value
        cute.arch.sync_threads()
        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        warp = tidx // 32
        lane = tidx % 32
        ranks = cute.make_rmem_tensor(cute.make_layout(CHUNK // NUM_THREADS), Int32)
        for i in cutlass.range_constexpr(CHUNK // NUM_THREADS):
            j = tidx + i * NUM_THREADS
            value = sChunk[j]
            segment = i * (NUM_THREADS // 32) + warp
            peers = cute.arch.match_sync(Int32(-1), value)
            rank = Int32(cute.arch.popc(peers & cute.arch.lanemask_lt()))
            if value >= 0 and rank == 0:
                sSeg[segment, value] = cute.arch.popc(peers)
            ranks[i] = rank
        cute.arch.sync_threads()

        count = Int32(0)
        if tidx < self.E:
            for segment in cutlass.range_constexpr(CHUNK // 32):
                segment_count = sSeg[segment, tidx]
                sSeg[segment, tidx] = count
                count += segment_count
            sTot[tidx] = count
            tile_count = (count + self.plan_tile_rows - 1) // self.plan_tile_rows
            sPad[tidx] = tile_count * 128
            sPlan[tidx] = tile_count
        cute.arch.sync_threads()
        for offset in (1, 2, 4, 8, 16, 32, 64, 128):
            add_count = Int32(0)
            add_padded = Int32(0)
            add_plan = Int32(0)
            if tidx < self.E and tidx >= offset:
                add_count = sTot[tidx - offset]
                add_padded = sPad[tidx - offset]
                add_plan = sPlan[tidx - offset]
            cute.arch.sync_threads()
            if tidx < self.E:
                sTot[tidx] += add_count
                sPad[tidx] += add_padded
                sPlan[tidx] += add_plan
            cute.arch.sync_threads()
        if tidx < self.E:
            base = sTot[tidx] - count
            sBase[tidx] = base
            mCu[tidx + 1] = sTot[tidx]
            mOffPad[tidx] = sPad[tidx]
            tile_count = (count + self.plan_tile_rows - 1) // self.plan_tile_rows
            plan_base = sPlan[tidx] - tile_count
            for tile_m in cutlass.range(tile_count):
                mPlan[plan_base + tile_m + 1] = tidx + tile_m * self.E
            mPlan[mPlan.shape[0] - self.E + tidx] = (sPad[tidx] - tile_count * 128) // 128
            if tidx == 0:
                mCu[0] = 0
                mPlan[0] = sPlan[self.E - 1]
        cute.arch.sync_threads()

        for i in cutlass.range_constexpr(CHUNK // NUM_THREADS):
            j = tidx + i * NUM_THREADS
            expert = sChunk[j]
            if j < M:
                segment = i * (NUM_THREADS // 32) + warp
                dst = sBase[expert] + sSeg[segment, expert] + ranks[i]
                mGi[dst] = expert if const_expr(self.gather_experts) else j // self.topk
                mPs[dst] = Float32(mTopV[j])
                mSlots[j] = dst

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
        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()
        if const_expr(self.warp_dispatch):
            warp = tidx // 32
            lane = tidx % 32
            for group in cutlass.range_constexpr((self.E + 7) // 8):
                expert = warp + group * 8
                if expert < self.E:
                    cnt = Int32(0)
                    for i in cutlass.range_constexpr(CHUNK // 32):
                        mask = cute.arch.vote_ballot_sync(sChunk[i * 32 + lane] == expert)
                        if lane == 0:
                            cnt += cute.arch.popc(mask)
                    if lane == 0:
                        mPart[b * self.E + expert] = cnt
        elif tidx < self.E:
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
        if const_expr(self.warp_dispatch):
            sBase = smem.allocate_tensor(Int32, cute.make_layout(self.E), byte_alignment=16)
            sSeg = smem.allocate_tensor(
                Int32,
                cute.make_layout((CHUNK // 32, self.E), stride=(self.E, 1)),
                byte_alignment=16,
            )
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
        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
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
            # Build stable and padded expert offsets.
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
            # Add this chunk's stable rank base.
            base = cu_e
            for b2 in cutlass.range(B):
                if b2 < b:
                    base += sPart[b2 * self.E + e]
            if const_expr(self.warp_dispatch):
                sBase[e] = base
            else:
                cnt = Int32(0)
                for j in cutlass.range(CHUNK):
                    if sChunk[j] == e:
                        p = b * CHUNK + j
                        s = base + cnt
                        mGi[s] = e if const_expr(self.gather_experts) else p // self.topk
                        mPs[s] = Float32(mTopV[p])
                        mSlots[p] = s
                        cnt += 1
        if const_expr(self.warp_dispatch):
            warp = tidx // 32
            lane = tidx % 32
            ranks = cute.make_rmem_tensor(cute.make_layout(CHUNK // NUM_THREADS), Int32)
            for i in cutlass.range_constexpr(CHUNK // NUM_THREADS):
                j = tidx + i * NUM_THREADS
                value = sChunk[j]
                rank = Int32(0)
                segment = i * (NUM_THREADS // 32) + warp
                for e in cutlass.range_constexpr(self.E):
                    mask = cute.arch.vote_ballot_sync(value == e)
                    if lane == 0:
                        sSeg[segment, e] = cute.arch.popc(mask)
                    if value == e:
                        rank = Int32(cute.arch.popc(cutlass.Uint32(mask) & cute.arch.lanemask_lt()))
                ranks[i] = rank
            cute.arch.sync_threads()
            for i in cutlass.range_constexpr(CHUNK // NUM_THREADS):
                j = tidx + i * NUM_THREADS
                p = b * CHUNK + j
                e = sChunk[j]
                if p < M:
                    segment = i * (NUM_THREADS // 32) + warp
                    dst = sBase[e] + ranks[i]
                    for earlier in cutlass.range_constexpr(CHUNK // 32):
                        if earlier < segment:
                            dst += sSeg[earlier, e]
                    mGi[dst] = e if const_expr(self.gather_experts) else p // self.topk
                    mPs[dst] = Float32(mTopV[p])
                    mSlots[p] = dst

    @staticmethod
    @jit_cache
    def compile(
        E,
        topk,
        small=False,
        use_pdl=True,
        gather_experts=False,
        plan_tile_rows=128,
    ):
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
            MoEDispatchKernel(E, topk, small, use_pdl, gather_experts, plan_tile_rows),
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
    *,
    use_pdl: bool = True,
    gather_experts: bool = False,
    plan_tile_rows: int = 128,
) -> None:
    """Build a stable expert-major permutation from preselected routes."""
    assert topi.dtype == torch.int32 and topi.is_contiguous()
    assert topv.dtype == torch.float32 and topv.is_contiguous()
    assert topi.dim() == 2 and topv.shape == topi.shape
    T, k = topi.shape
    M = T * k
    B = -(-M // CHUNK)
    assert B <= B_MAX, f"M={M} exceeds the {B_MAX * CHUNK} dispatch bound"
    assert plan_tile_rows > 0 and 128 % plan_tile_rows == 0
    assert gi.numel() == M and slots.numel() == M and ps.numel() == M
    assert cu.numel() == E + 1 and part.numel() >= B * E
    assert off_pad.numel() == E and off_pad.dtype == torch.int32
    MoEDispatchKernel.compile(E, k, B == 1, use_pdl, gather_experts, plan_tile_rows)(
        topi.view(-1), topv.view(-1), part, gi, cu, ps, slots, off_pad, B
    )


class MoEDispatch:
    """Preallocated expert-major dispatch buffers."""

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        device="cuda",
    ):
        num_assignments = num_tokens * top_k
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.gi = torch.empty(num_assignments, dtype=torch.int32, device=device)
        self.cu = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        self.ps = torch.empty(num_assignments, dtype=torch.float32, device=device)
        self.slots = torch.empty(num_assignments, dtype=torch.int32, device=device)
        self.part = torch.empty(B_MAX * num_experts, dtype=torch.int32, device=device)
        self.off_pad = torch.empty(num_experts, dtype=torch.int32, device=device)

    def __call__(self, topi: torch.Tensor, topv: torch.Tensor):
        assert topi.shape == (self.num_tokens, self.top_k)
        probs = topv.float() if topv.dtype != torch.float32 else topv
        moe_dispatch(
            topi,
            probs,
            self.num_experts,
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
        assert topv.shape == (self.num_tokens, self.top_k)
        return torch.zeros_like(self.ps).index_put((self.slots.long(),), topv.reshape(-1).float())


__all__ = ["MoEDispatch", "MoEDispatchKernel", "moe_dispatch"]
