# Copyright (c) 2026, Sung Hyun Cho.
"""Decode quantization that writes one token to all selected experts."""

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import Float32, Int32, cute
from torch import Tensor

from ._common import _bits_f32, fake_tensor, jit_cache
from .epilogue import gated_postact_value, quantize_postact_fragment, validate_gated_activation
from .quantize import (
    _ABSP,
    _HI16,
    E4M3_EPS,
    E4M3_MAX,
    INV6,
    _cvt_e2m1_pair_rn,
    _cvt_e4m3_rn,
    _div_rn,
    _max_bf16x2,
    _max_u32,
)

FEATURE_ALIGNMENT = 256
NUM_THREADS = 32
DISPATCH_THREADS = 256
DISPATCH_CHUNK = 1024


class NVFP4DecodeQuantKernel:
    def __init__(
        self,
        features: int,
        experts: int,
        topk: int,
        use_pdl: bool = True,
        sf_tile_rows: int = 128,
        token_major_q: bool = False,
    ):
        self.features = features
        self.experts = experts
        self.topk = topk
        self.use_pdl = use_pdl
        self.sf_tile_rows = sf_tile_rows
        self.token_major_q = token_major_q
        self.features_per_cta = 512 if features % 512 == 0 else FEATURE_ALIGNMENT

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mTopI: cute.Tensor,
        mSlots: cute.Tensor,
        mCu: cute.Tensor,
        mTileEnds: cute.Tensor,
        mPts: cute.Tensor,
        mQOut: cute.Tensor,
        mSFOut: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(mX, mTopI, mSlots, mCu, mTileEnds, mPts, mQOut, mSFOut).launch(
            grid=[self.features // self.features_per_cta, mX.shape[0], 1],
            block=[NUM_THREADS, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mTopI: cute.Tensor,
        mSlots: cute.Tensor,
        mCu: cute.Tensor,
        mTileEnds: cute.Tensor,
        mPts: cute.Tensor,
        mQOut: cute.Tensor,
        mSFOut: cute.Tensor,
    ):
        feature_tile, token, _ = cute.arch.block_idx()
        lane, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()
        feature_group = feature_tile * (self.features_per_cta // 16) + lane
        if lane < self.features_per_cta // 16:
            layout8 = cute.make_layout(8)
            copy16 = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
            )
            fragment = cute.make_rmem_tensor(cute.make_layout(16), cutlass.BFloat16)
            fragment_u32 = cute.make_tensor(
                cute.recast_ptr(fragment.iterator, dtype=cutlass.Uint32),
                cute.make_layout(8),
            )
            values = cute.make_rmem_tensor(cute.make_layout(16), Float32)
            qbytes = cute.make_rmem_tensor(layout8, cutlass.Uint8)
            feature = feature_group * 16
            for half in cutlass.range_constexpr(2):
                x_source = cute.make_tensor(
                    mX.iterator + cute.assume(token * self.features + feature + half * 8, divby=8),
                    layout8,
                )
                fragment_target = cute.make_tensor(fragment.iterator + half * 8, layout8)
                cute.copy(copy16, x_source, fragment_target)
            for pair in cutlass.range_constexpr(8):
                word = fragment_u32[pair]
                values[2 * pair] = _bits_f32(word << 16)
                values[2 * pair + 1] = _bits_f32(word & cutlass.Uint32(_HI16))

            t0 = _max_bf16x2(
                fragment_u32[0] & cutlass.Uint32(_ABSP),
                fragment_u32[1] & cutlass.Uint32(_ABSP),
            )
            t1 = _max_bf16x2(
                fragment_u32[2] & cutlass.Uint32(_ABSP),
                fragment_u32[3] & cutlass.Uint32(_ABSP),
            )
            t2 = _max_bf16x2(
                fragment_u32[4] & cutlass.Uint32(_ABSP),
                fragment_u32[5] & cutlass.Uint32(_ABSP),
            )
            t3 = _max_bf16x2(
                fragment_u32[6] & cutlass.Uint32(_ABSP),
                fragment_u32[7] & cutlass.Uint32(_ABSP),
            )
            packed_max = _max_bf16x2(_max_bf16x2(t0, t1), _max_bf16x2(t2, t3))
            amax = _bits_f32(
                _max_u32(
                    packed_max << 16,
                    packed_max & cutlass.Uint32(_HI16),
                )
            )
            pts = Float32(mPts[0])
            inv_pts = Float32(mPts[1])
            scaled = amax * (inv_pts * Float32(INV6))
            scaled = cutlass.max(cutlass.min(scaled, Float32(E4M3_MAX)), Float32(E4M3_EPS))
            sf = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Float8E4M3FN)
            sf_u8 = cute.make_tensor(
                cute.recast_ptr(sf.iterator, dtype=cutlass.Uint8), cute.make_layout(1)
            )
            sf_u8[0] = cutlass.Uint8(_cvt_e4m3_rn(scaled) & 0xFF)
            reciprocal = _div_rn(Float32(1.0), sf[0].to(Float32) * pts)
            for pair in cutlass.range_constexpr(0, 16, 2):
                values[pair], values[pair + 1] = cute.arch.mul_packed_f32x2(
                    (values[pair], values[pair + 1]),
                    (reciprocal, reciprocal),
                    rnd="rn",
                    ftz=False,
                )
            for byte in cutlass.range_constexpr(8):
                qbytes[byte] = cutlass.Uint8(
                    _cvt_e2m1_pair_rn(values[2 * byte + 1], values[2 * byte]) & 0xFF
                )

            copy8 = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.Uint8, num_bits_per_copy=64
            )
            q_row_bytes = self.features // 2
            sf_groups = self.features // 64
            sf_row_groups = self.features // 16
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.token_major_q):
                q_target = cute.make_tensor(
                    mQOut.iterator + cute.assume(token * q_row_bytes + feature_group * 8, divby=8),
                    layout8,
                )
                cute.copy(copy8, qbytes, q_target)
                mSFOut[token * sf_row_groups + feature_group] = sf_u8[0]
            else:
                for route in cutlass.range_constexpr(self.topk):
                    flat_slot = token * self.topk + route
                    dst_row = mSlots[flat_slot]
                    q_target = cute.make_tensor(
                        mQOut.iterator
                        + cute.assume(dst_row * q_row_bytes + feature_group * 8, divby=8),
                        layout8,
                    )
                    cute.copy(copy8, qbytes, q_target)

                    expert = mTopI[token, route]
                    expert_begin = mCu[expert]
                    previous_tiles = Int32(0)
                    if expert > 0:
                        previous_tiles = mTileEnds[expert - 1] // 128
                    row_in_expert = dst_row - expert_begin
                    tile = previous_tiles + row_in_expert // self.sf_tile_rows
                    row_in_tile = row_in_expert % self.sf_tile_rows
                    sf_byte = (
                        tile * sf_groups * 512
                        + (feature_group // 4) * 512
                        + (row_in_tile % 32) * 16
                        + (row_in_tile // 32) * 4
                        + feature_group % 4
                    )
                    mSFOut[sf_byte] = sf_u8[0]

    @staticmethod
    @jit_cache
    def compile(
        features: int,
        experts: int,
        topk: int,
        use_pdl: bool = True,
        sf_tile_rows: int = 128,
        token_major_q: bool = False,
    ):
        tokens, rows, sf_bytes = (cute.sym_int() for _ in range(3))
        x = fake_tensor(cutlass.BFloat16, (tokens, features), 8)
        topi = fake_tensor(Int32, (tokens, topk), 1)
        slots = fake_tensor(Int32, (rows,), 1)
        cu = fake_tensor(Int32, (experts + 1,), 1)
        tile_ends = fake_tensor(Int32, (experts,), 1)
        pts = fake_tensor(Float32, (2,), 1)
        q_rows = tokens if token_major_q else rows
        q_out = fake_tensor(cutlass.Uint8, (q_rows, features // 2), 8)
        sf_out = fake_tensor(cutlass.Uint8, (sf_bytes,), 1)
        return cute.compile(
            NVFP4DecodeQuantKernel(
                features,
                experts,
                topk,
                use_pdl,
                sf_tile_rows,
                token_major_q,
            ),
            x,
            topi,
            slots,
            cu,
            tile_ends,
            pts,
            q_out,
            sf_out,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )


class NVFP4DecodeDispatchQuantKernel:
    def __init__(
        self,
        features: int,
        experts: int,
        topk: int,
        plan_tile_rows: int,
        use_pdl: bool = True,
    ):
        self.features = features
        self.experts = experts
        self.topk = topk
        self.plan_tile_rows = plan_tile_rows
        self.use_pdl = use_pdl
        self.threads = DISPATCH_THREADS
        self.features_per_cta = 512 if features % 512 == 0 else FEATURE_ALIGNMENT

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mTopI: cute.Tensor,
        mTopV: cute.Tensor,
        mPlan: cute.Tensor,
        mGather: cute.Tensor,
        mCu: cute.Tensor,
        mProbs: cute.Tensor,
        mSlots: cute.Tensor,
        mOffPad: cute.Tensor,
        mPts: cute.Tensor,
        mQOut: cute.Tensor,
        mSFOut: cute.Tensor,
        stream: cuda.CUstream,
    ):
        quant_blocks = mX.shape[0] * (self.features // self.features_per_cta)
        self.kernel(
            mX,
            mTopI,
            mTopV,
            mPlan,
            mGather,
            mCu,
            mProbs,
            mSlots,
            mOffPad,
            mPts,
            mQOut,
            mSFOut,
        ).launch(
            grid=[quant_blocks + 1, 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mTopI: cute.Tensor,
        mTopV: cute.Tensor,
        mPlan: cute.Tensor,
        mGather: cute.Tensor,
        mCu: cute.Tensor,
        mProbs: cute.Tensor,
        mSlots: cute.Tensor,
        mOffPad: cute.Tensor,
        mPts: cute.Tensor,
        mQOut: cute.Tensor,
        mSFOut: cute.Tensor,
    ):
        block, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        if block == 0:
            routes = mTopI.shape[0]
            smem = cutlass.utils.SmemAllocator()
            sChunk = smem.allocate_tensor(
                Int32, cute.make_layout(DISPATCH_CHUNK), byte_alignment=16
            )
            sBase = smem.allocate_tensor(Int32, cute.make_layout(self.experts), byte_alignment=16)
            sTot = smem.allocate_tensor(Int32, cute.make_layout(self.experts), byte_alignment=16)
            sPad = smem.allocate_tensor(Int32, cute.make_layout(self.experts), byte_alignment=16)
            sPlan = smem.allocate_tensor(Int32, cute.make_layout(self.experts), byte_alignment=16)
            sSeg = smem.allocate_tensor(
                Int32,
                cute.make_layout((DISPATCH_CHUNK // 32, self.experts), stride=(self.experts, 1)),
                byte_alignment=16,
            )
            segment_items = (DISPATCH_CHUNK // 32) * self.experts
            segment_iters = (segment_items + self.threads - 1) // self.threads
            for index in cutlass.range_constexpr(segment_iters):
                linear = tidx + index * self.threads
                if linear < segment_items:
                    sSeg[linear // self.experts, linear % self.experts] = 0
            for index in cutlass.range_constexpr(DISPATCH_CHUNK // self.threads):
                route = tidx + index * self.threads
                expert = Int32(-1)
                if route < routes:
                    expert = mTopI[route]
                sChunk[route] = expert
            cute.arch.sync_threads()

            warp = tidx // 32
            lane = tidx % 32
            ranks = cute.make_rmem_tensor(cute.make_layout(DISPATCH_CHUNK // self.threads), Int32)
            for index in cutlass.range_constexpr(DISPATCH_CHUNK // self.threads):
                route = tidx + index * self.threads
                expert = sChunk[route]
                segment = index * (self.threads // 32) + warp
                peers = cute.arch.match_sync(Int32(-1), expert)
                rank = Int32(cute.arch.popc(peers & cute.arch.lanemask_lt()))
                if expert >= 0 and rank == 0:
                    sSeg[segment, expert] = cute.arch.popc(peers)
                ranks[index] = rank
            cute.arch.sync_threads()

            count = Int32(0)
            if tidx < self.experts:
                for segment in cutlass.range_constexpr(DISPATCH_CHUNK // 32):
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
                if tidx < self.experts and tidx >= offset:
                    add_count = sTot[tidx - offset]
                    add_padded = sPad[tidx - offset]
                    add_plan = sPlan[tidx - offset]
                cute.arch.sync_threads()
                if tidx < self.experts:
                    sTot[tidx] += add_count
                    sPad[tidx] += add_padded
                    sPlan[tidx] += add_plan
                cute.arch.sync_threads()
            if tidx < self.experts:
                base = sTot[tidx] - count
                sBase[tidx] = base
                mCu[tidx + 1] = sTot[tidx]
                mOffPad[tidx] = sPad[tidx]
                tile_count = (count + self.plan_tile_rows - 1) // self.plan_tile_rows
                plan_base = sPlan[tidx] - tile_count
                for tile_m in cutlass.range(tile_count):
                    mPlan[plan_base + tile_m + 1] = tidx + tile_m * self.experts
                mPlan[mPlan.shape[0] - self.experts + tidx] = (sPad[tidx] - tile_count * 128) // 128
                if tidx == 0:
                    mCu[0] = 0
                    mPlan[0] = sPlan[self.experts - 1]
            cute.arch.sync_threads()

            for index in cutlass.range_constexpr(DISPATCH_CHUNK // self.threads):
                route = tidx + index * self.threads
                expert = sChunk[route]
                if route < routes:
                    segment = index * (self.threads // 32) + warp
                    dst = sBase[expert] + sSeg[segment, expert] + ranks[index]
                    mGather[dst] = route // self.topk
                    mProbs[dst] = Float32(mTopV[route])
                    mSlots[route] = dst
        else:
            feature_tiles = self.features // self.features_per_cta
            quant_block = block - 1
            token = quant_block // feature_tiles
            feature_tile = quant_block % feature_tiles
            if tidx < self.features_per_cta // 16:
                feature_group = feature_tile * (self.features_per_cta // 16) + tidx
                if feature_group < self.features // 16:
                    layout8 = cute.make_layout(8)
                    copy16 = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        cutlass.BFloat16,
                        num_bits_per_copy=128,
                    )
                    fragment = cute.make_rmem_tensor(cute.make_layout(16), cutlass.BFloat16)
                    fragment_u32 = cute.make_tensor(
                        cute.recast_ptr(fragment.iterator, dtype=cutlass.Uint32),
                        cute.make_layout(8),
                    )
                    values = cute.make_rmem_tensor(cute.make_layout(16), Float32)
                    qbytes = cute.make_rmem_tensor(layout8, cutlass.Uint8)
                    feature = feature_group * 16
                    for half in cutlass.range_constexpr(2):
                        source = cute.make_tensor(
                            mX.iterator
                            + cute.assume(
                                token * self.features + feature + half * 8,
                                divby=8,
                            ),
                            layout8,
                        )
                        target = cute.make_tensor(fragment.iterator + half * 8, layout8)
                        cute.copy(copy16, source, target)
                    for pair in cutlass.range_constexpr(8):
                        word = fragment_u32[pair]
                        values[2 * pair] = _bits_f32(word << 16)
                        values[2 * pair + 1] = _bits_f32(word & cutlass.Uint32(_HI16))

                    t0 = _max_bf16x2(
                        fragment_u32[0] & cutlass.Uint32(_ABSP),
                        fragment_u32[1] & cutlass.Uint32(_ABSP),
                    )
                    t1 = _max_bf16x2(
                        fragment_u32[2] & cutlass.Uint32(_ABSP),
                        fragment_u32[3] & cutlass.Uint32(_ABSP),
                    )
                    t2 = _max_bf16x2(
                        fragment_u32[4] & cutlass.Uint32(_ABSP),
                        fragment_u32[5] & cutlass.Uint32(_ABSP),
                    )
                    t3 = _max_bf16x2(
                        fragment_u32[6] & cutlass.Uint32(_ABSP),
                        fragment_u32[7] & cutlass.Uint32(_ABSP),
                    )
                    packed_max = _max_bf16x2(_max_bf16x2(t0, t1), _max_bf16x2(t2, t3))
                    amax = _bits_f32(
                        _max_u32(
                            packed_max << 16,
                            packed_max & cutlass.Uint32(_HI16),
                        )
                    )
                    pts = Float32(mPts[0])
                    inv_pts = Float32(mPts[1])
                    scaled = amax * (inv_pts * Float32(INV6))
                    scaled = cutlass.max(cutlass.min(scaled, Float32(E4M3_MAX)), Float32(E4M3_EPS))
                    sf = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Float8E4M3FN)
                    sf_u8 = cute.make_tensor(
                        cute.recast_ptr(sf.iterator, dtype=cutlass.Uint8),
                        cute.make_layout(1),
                    )
                    sf_u8[0] = cutlass.Uint8(_cvt_e4m3_rn(scaled) & 0xFF)
                    reciprocal = _div_rn(Float32(1.0), sf[0].to(Float32) * pts)
                    for pair in cutlass.range_constexpr(0, 16, 2):
                        values[pair], values[pair + 1] = cute.arch.mul_packed_f32x2(
                            (values[pair], values[pair + 1]),
                            (reciprocal, reciprocal),
                            rnd="rn",
                            ftz=False,
                        )
                    for byte in cutlass.range_constexpr(8):
                        qbytes[byte] = cutlass.Uint8(
                            _cvt_e2m1_pair_rn(values[2 * byte + 1], values[2 * byte]) & 0xFF
                        )

                    if cutlass.const_expr(self.use_pdl):
                        cute.arch.griddepcontrol_wait()
                    q_target = cute.make_tensor(
                        mQOut.iterator
                        + cute.assume(
                            token * (self.features // 2) + feature_group * 8,
                            divby=8,
                        ),
                        layout8,
                    )
                    copy8 = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        cutlass.Uint8,
                        num_bits_per_copy=64,
                    )
                    cute.copy(copy8, qbytes, q_target)
                    mSFOut[token * (self.features // 16) + feature_group] = sf_u8[0]

    @staticmethod
    @jit_cache
    def compile(
        features: int,
        experts: int,
        topk: int,
        plan_tile_rows: int,
        use_pdl: bool = True,
    ):
        tokens, routes, plan_rows, sf_bytes = (cute.sym_int() for _ in range(4))
        return cute.compile(
            NVFP4DecodeDispatchQuantKernel(features, experts, topk, plan_tile_rows, use_pdl),
            fake_tensor(cutlass.BFloat16, (tokens, features), 8),
            fake_tensor(Int32, (routes,), 1),
            fake_tensor(Float32, (routes,), 1),
            fake_tensor(Int32, (plan_rows,), 1),
            fake_tensor(Int32, (routes,), 1),
            fake_tensor(Int32, (experts + 1,), 1),
            fake_tensor(Float32, (routes,), 1),
            fake_tensor(Int32, (routes,), 1),
            fake_tensor(Int32, (experts,), 1),
            fake_tensor(Float32, (2,), 1),
            fake_tensor(cutlass.Uint8, (tokens, features // 2), 8),
            fake_tensor(cutlass.Uint8, (sf_bytes,), 1),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )


class NVFP4BatchOneDispatchQuantKernel:
    def __init__(
        self,
        features: int,
        experts: int,
        topk: int,
        use_pdl: bool = True,
        direct: bool = False,
    ):
        self.features = features
        self.experts = experts
        self.topk = topk
        self.use_pdl = use_pdl
        self.direct = direct
        self.features_per_cta = 512 if features % 512 == 0 else FEATURE_ALIGNMENT

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mTopI: cute.Tensor,
        mTopV: cute.Tensor,
        mGather: cute.Tensor,
        mSlots: cute.Tensor,
        mCu: cute.Tensor,
        mProbs: cute.Tensor,
        mTileEnds: cute.Tensor,
        mPts: cute.Tensor,
        mQOut: cute.Tensor,
        mSFOut: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            mX,
            mTopI,
            mTopV,
            mGather,
            mSlots,
            mCu,
            mProbs,
            mTileEnds,
            mPts,
            mQOut,
            mSFOut,
        ).launch(
            grid=[self.features // self.features_per_cta, mX.shape[0], 1],
            block=[NUM_THREADS, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mTopI: cute.Tensor,
        mTopV: cute.Tensor,
        mGather: cute.Tensor,
        mSlots: cute.Tensor,
        mCu: cute.Tensor,
        mProbs: cute.Tensor,
        mTileEnds: cute.Tensor,
        mPts: cute.Tensor,
        mQOut: cute.Tensor,
        mSFOut: cute.Tensor,
    ):
        feature_tile, token, _ = cute.arch.block_idx()
        lane, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        smem = cutlass.utils.SmemAllocator()
        sRoute = smem.allocate_tensor(Int32, cute.make_layout(self.topk), byte_alignment=16)
        sTile = smem.allocate_tensor(Int32, cute.make_layout(self.topk), byte_alignment=16)
        sRow = smem.allocate_tensor(Int32, cute.make_layout(self.topk), byte_alignment=16)

        if lane == 0:
            for route in cutlass.range_constexpr(self.topk):
                expert = mTopI[0, route]
                if cutlass.const_expr(self.direct):
                    sRoute[route] = 0
                    sTile[route] = 0
                    sRow[route] = 0
                    if feature_tile == 0:
                        mSlots[route] = route
                        mCu[route] = expert
                        mProbs[route] = Float32(mTopV[0, route])
                    continue
                dst = Int32(0)
                row = Int32(0)
                tile = Int32(0)
                for other in cutlass.range_constexpr(self.topk):
                    other_expert = mTopI[0, other]
                    if other_expert < expert:
                        dst += 1
                    if other < route and other_expert == expert:
                        dst += 1
                        row += 1
                    first = Int32(1)
                    for previous in cutlass.range_constexpr(self.topk):
                        if previous < other and mTopI[0, previous] == other_expert:
                            first = 0
                    if other_expert < expert and first == 1:
                        count = Int32(0)
                        for candidate in cutlass.range_constexpr(self.topk):
                            if mTopI[0, candidate] == other_expert:
                                count += 1
                        tile += (count + 127) // 128
                tile += row // 128
                sRoute[route] = dst
                sTile[route] = tile
                sRow[route] = row % 128
                if feature_tile == 0:
                    mGather[dst] = 0
                    mSlots[route] = dst
                    mProbs[dst] = Float32(mTopV[0, route])
            if cutlass.const_expr(not self.direct) and feature_tile == 0:
                rows = Int32(0)
                padded = Int32(0)
                mCu[0] = 0
                for expert in cutlass.range_constexpr(self.experts):
                    count = Int32(0)
                    for route in cutlass.range_constexpr(self.topk):
                        if mTopI[0, route] == expert:
                            count += 1
                    rows += count
                    padded += ((count + 127) // 128) * 128
                    mCu[expert + 1] = rows
                    mTileEnds[expert] = padded
        cute.arch.sync_warp()

        feature_group = feature_tile * (self.features_per_cta // 16) + lane
        if lane < self.features_per_cta // 16:
            layout8 = cute.make_layout(8)
            copy16 = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
            )
            fragment = cute.make_rmem_tensor(cute.make_layout(16), cutlass.BFloat16)
            fragment_u32 = cute.make_tensor(
                cute.recast_ptr(fragment.iterator, dtype=cutlass.Uint32),
                cute.make_layout(8),
            )
            values = cute.make_rmem_tensor(cute.make_layout(16), Float32)
            qbytes = cute.make_rmem_tensor(layout8, cutlass.Uint8)
            feature = feature_group * 16
            for half in cutlass.range_constexpr(2):
                x_source = cute.make_tensor(
                    mX.iterator + cute.assume(token * self.features + feature + half * 8, divby=8),
                    layout8,
                )
                fragment_target = cute.make_tensor(fragment.iterator + half * 8, layout8)
                cute.copy(copy16, x_source, fragment_target)
            for pair in cutlass.range_constexpr(8):
                word = fragment_u32[pair]
                values[2 * pair] = _bits_f32(word << 16)
                values[2 * pair + 1] = _bits_f32(word & cutlass.Uint32(_HI16))

            t0 = _max_bf16x2(
                fragment_u32[0] & cutlass.Uint32(_ABSP),
                fragment_u32[1] & cutlass.Uint32(_ABSP),
            )
            t1 = _max_bf16x2(
                fragment_u32[2] & cutlass.Uint32(_ABSP),
                fragment_u32[3] & cutlass.Uint32(_ABSP),
            )
            t2 = _max_bf16x2(
                fragment_u32[4] & cutlass.Uint32(_ABSP),
                fragment_u32[5] & cutlass.Uint32(_ABSP),
            )
            t3 = _max_bf16x2(
                fragment_u32[6] & cutlass.Uint32(_ABSP),
                fragment_u32[7] & cutlass.Uint32(_ABSP),
            )
            packed_max = _max_bf16x2(_max_bf16x2(t0, t1), _max_bf16x2(t2, t3))
            amax = _bits_f32(
                _max_u32(
                    packed_max << 16,
                    packed_max & cutlass.Uint32(_HI16),
                )
            )
            pts = Float32(mPts[0])
            inv_pts = Float32(mPts[1])
            scaled = amax * (inv_pts * Float32(INV6))
            scaled = cutlass.max(cutlass.min(scaled, Float32(E4M3_MAX)), Float32(E4M3_EPS))
            sf = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Float8E4M3FN)
            sf_u8 = cute.make_tensor(
                cute.recast_ptr(sf.iterator, dtype=cutlass.Uint8), cute.make_layout(1)
            )
            sf_u8[0] = cutlass.Uint8(_cvt_e4m3_rn(scaled) & 0xFF)
            reciprocal = _div_rn(Float32(1.0), sf[0].to(Float32) * pts)
            for pair in cutlass.range_constexpr(0, 16, 2):
                values[pair], values[pair + 1] = cute.arch.mul_packed_f32x2(
                    (values[pair], values[pair + 1]),
                    (reciprocal, reciprocal),
                    rnd="rn",
                    ftz=False,
                )
            for byte in cutlass.range_constexpr(8):
                qbytes[byte] = cutlass.Uint8(
                    _cvt_e2m1_pair_rn(values[2 * byte + 1], values[2 * byte]) & 0xFF
                )

            copy8 = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.Uint8, num_bits_per_copy=64
            )
            q_row_bytes = self.features // 2
            sf_groups = self.features // 64
            output_routes = self.topk
            if cutlass.const_expr(self.direct):
                output_routes = 1
            for route in cutlass.range_constexpr(output_routes):
                dst_row = sRoute[route]
                tile = sTile[route]
                row_in_tile = sRow[route]
                if cutlass.const_expr(self.direct):
                    dst_row = Int32(0)
                    tile = Int32(0)
                    row_in_tile = Int32(0)
                q_target = cute.make_tensor(
                    mQOut.iterator
                    + cute.assume(dst_row * q_row_bytes + feature_group * 8, divby=8),
                    layout8,
                )
                cute.copy(copy8, qbytes, q_target)

                sf_byte = (
                    tile * sf_groups * 512
                    + (feature_group // 4) * 512
                    + (row_in_tile % 32) * 16
                    + (row_in_tile // 32) * 4
                    + feature_group % 4
                )
                mSFOut[sf_byte] = sf_u8[0]

    @staticmethod
    @jit_cache
    def compile(
        features: int,
        experts: int,
        topk: int,
        use_pdl: bool = True,
        direct: bool = False,
    ):
        sf_bytes = (experts + 1) * (features // 64) * 512
        x = fake_tensor(cutlass.BFloat16, (1, features), 8)
        topi = fake_tensor(Int32, (1, topk), 1)
        topv = fake_tensor(Float32, (1, topk), 1)
        gather = fake_tensor(Int32, (topk,), 1)
        slots = fake_tensor(Int32, (topk,), 1)
        cu = fake_tensor(Int32, (topk if direct else experts + 1,), 1)
        probs = fake_tensor(Float32, (topk,), 1)
        tile_ends = fake_tensor(Int32, (experts,), 1)
        pts = fake_tensor(Float32, (2,), 1)
        q_out = fake_tensor(cutlass.Uint8, (topk, features // 2), 8)
        sf_out = fake_tensor(cutlass.Uint8, (sf_bytes,), 1)
        return cute.compile(
            NVFP4BatchOneDispatchQuantKernel(features, experts, topk, use_pdl, direct),
            x,
            topi,
            topv,
            gather,
            slots,
            cu,
            probs,
            tile_ends,
            pts,
            q_out,
            sf_out,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )


class NVFP4GatedQuantKernel:
    def __init__(
        self,
        features: int,
        experts: int,
        activation: str,
        use_pdl: bool = True,
    ):
        self.features = features
        self.experts = experts
        self.activation = validate_gated_activation(activation)
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        mPreact: cute.Tensor,
        mRowExperts: cute.Tensor,
        mCu: cute.Tensor,
        mTileEnds: cute.Tensor,
        mPts: cute.Tensor,
        mQOut: cute.Tensor,
        mSFOut: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(mPreact, mRowExperts, mCu, mTileEnds, mPts, mQOut, mSFOut).launch(
            grid=[cute.ceil_div(self.features, NUM_THREADS * 16), mPreact.shape[0], 1],
            block=[NUM_THREADS, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mPreact: cute.Tensor,
        mRowExperts: cute.Tensor,
        mCu: cute.Tensor,
        mTileEnds: cute.Tensor,
        mPts: cute.Tensor,
        mQOut: cute.Tensor,
        mSFOut: cute.Tensor,
    ):
        feature_tile, row, _ = cute.arch.block_idx()
        lane, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()

        expert = mRowExperts[row]
        expert_row = row - mCu[expert]
        previous_tiles = Int32(0)
        if expert > 0:
            previous_tiles = mTileEnds[expert - 1] // 128

        feature_group = feature_tile * NUM_THREADS + lane
        if feature_group < self.features // 16:
            feature = feature_group * 16
            postact = cute.make_rmem_tensor(cute.make_layout(16), Float32)
            for index in cutlass.range_constexpr(16):
                gate = Float32(mPreact[row, 2 * (feature + index)])
                up = Float32(mPreact[row, 2 * (feature + index) + 1])
                postact[index] = gated_postact_value(gate, up, self.activation, 0.0)
            scaled, scale_factors = quantize_postact_fragment(postact, Float32(mPts[1]))
            packed = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Uint8)
            for pair in cutlass.range_constexpr(8):
                packed[pair] = cutlass.Uint8(
                    _cvt_e2m1_pair_rn(scaled[pair * 2 + 1], scaled[pair * 2]) & 0xFF
                )

            copy8 = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.Uint8, num_bits_per_copy=64
            )
            q_target = cute.make_tensor(
                mQOut.iterator
                + cute.assume(row * (self.features // 2) + feature_group * 8, divby=8),
                cute.make_layout(8),
            )
            cute.copy(copy8, packed, q_target)

            tile = previous_tiles + expert_row // 128
            row_in_tile = expert_row % 128
            sf_offset = (
                tile * (self.features // 64) * 512
                + (feature_group // 4) * 512
                + (row_in_tile % 32) * 16
                + (row_in_tile // 32) * 4
                + feature_group % 4
            )
            scale_bytes = cute.make_tensor(
                cute.recast_ptr(scale_factors.iterator, dtype=cutlass.Uint8),
                cute.make_layout(1),
            )
            mSFOut[sf_offset] = scale_bytes[0]

    @staticmethod
    @jit_cache
    def compile(
        features: int,
        experts: int,
        activation: str,
        use_pdl: bool = True,
    ):
        rows, sf_bytes = (cute.sym_int() for _ in range(2))
        preact = fake_tensor(Float32, (rows, 2 * features), 4)
        row_experts = fake_tensor(Int32, (rows,), 1)
        cu = fake_tensor(Int32, (experts + 1,), 1)
        tile_ends = fake_tensor(Int32, (experts,), 1)
        pts = fake_tensor(Float32, (2,), 1)
        q_out = fake_tensor(cutlass.Uint8, (rows, features // 2), 8)
        sf_out = fake_tensor(cutlass.Uint8, (sf_bytes,), 1)
        return cute.compile(
            NVFP4GatedQuantKernel(features, experts, activation, use_pdl),
            preact,
            row_experts,
            cu,
            tile_ends,
            pts,
            q_out,
            sf_out,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )


def nvfp4_decode_prepare(
    x: Tensor,
    topk_ids: Tensor,
    topk_weights: Tensor,
    gather: Tensor,
    slots: Tensor,
    cu: Tensor,
    probs: Tensor,
    padded_offsets: Tensor,
    scale: Tensor,
    q_out: Tensor,
    sf_out: Tensor,
    *,
    use_pdl: bool = True,
) -> None:
    """Build batch-one expert rows and quantize the input in one launch."""
    if x.ndim != 2 or topk_ids.ndim != 2:
        raise ValueError("decode input and expert ids must be matrices")
    tokens, features = x.shape
    topk = topk_ids.shape[1]
    experts = cu.numel() - 1
    rows = tokens * topk
    if tokens != 1 or experts <= 0 or experts > 256 or topk <= 0 or topk > min(experts, 32):
        raise ValueError("fused decode requires one token, 1-256 experts, and top-k at most 32")
    expected = (
        x.dtype == torch.bfloat16
        and features % FEATURE_ALIGNMENT == 0
        and topk_ids.dtype == torch.int32
        and tuple(topk_ids.shape) == (tokens, topk)
        and topk_weights.dtype == torch.float32
        and tuple(topk_weights.shape) == (tokens, topk)
        and gather.dtype == torch.int32
        and gather.numel() == rows
        and slots.dtype == torch.int32
        and slots.numel() == rows
        and cu.dtype == torch.int32
        and tuple(cu.shape) == (experts + 1,)
        and probs.dtype == torch.float32
        and probs.numel() == rows
        and padded_offsets.dtype == torch.int32
        and tuple(padded_offsets.shape) == (experts,)
        and scale.dtype == torch.float32
        and tuple(scale.shape) == (2,)
        and q_out.dtype in (torch.uint8, torch.float4_e2m1fn_x2)
        and q_out.numel() == rows * features // 2
        and sf_out.dtype == torch.float8_e4m3fn
    )
    if not expected:
        raise ValueError("fused decode tensors do not match the configured shape")
    tensors = (
        x,
        topk_ids,
        topk_weights,
        gather,
        slots,
        cu,
        probs,
        padded_offsets,
        scale,
        q_out,
        sf_out,
    )
    device = x.device
    if any(
        not tensor.is_cuda or not tensor.is_contiguous() or tensor.device != device
        for tensor in tensors
    ):
        raise ValueError("fused decode tensors must be contiguous on one CUDA device")
    NVFP4BatchOneDispatchQuantKernel.compile(features, experts, topk, use_pdl)(
        x,
        topk_ids,
        topk_weights,
        gather,
        slots,
        cu,
        probs,
        padded_offsets,
        scale,
        q_out.view(torch.uint8),
        sf_out.view(torch.uint8).view(-1),
    )


def nvfp4_direct_decode_prepare(
    x: Tensor,
    topk_ids: Tensor,
    topk_weights: Tensor,
    gather: Tensor,
    slots: Tensor,
    route_ids: Tensor,
    probs: Tensor,
    padded_offsets: Tensor,
    scale: Tensor,
    q_out: Tensor,
    sf_out: Tensor,
    *,
    use_pdl: bool = True,
) -> None:
    """Quantize one token and emit direct route metadata."""
    tokens, features = x.shape
    topk = topk_ids.shape[1]
    experts = padded_offsets.numel()
    rows = tokens * topk
    expected = (
        tokens == 1
        and 0 < topk <= min(experts, 32)
        and x.dtype == torch.bfloat16
        and features % FEATURE_ALIGNMENT == 0
        and topk_ids.dtype == torch.int32
        and tuple(topk_ids.shape) == (1, topk)
        and topk_weights.dtype == torch.float32
        and tuple(topk_weights.shape) == (1, topk)
        and gather.dtype == torch.int32
        and gather.numel() == rows
        and slots.dtype == torch.int32
        and slots.numel() == rows
        and route_ids.dtype == torch.int32
        and tuple(route_ids.shape) == (topk,)
        and probs.dtype == torch.float32
        and probs.numel() == rows
        and padded_offsets.dtype == torch.int32
        and scale.dtype == torch.float32
        and tuple(scale.shape) == (2,)
        and q_out.dtype in (torch.uint8, torch.float4_e2m1fn_x2)
        and q_out.numel() == rows * features // 2
        and sf_out.dtype == torch.float8_e4m3fn
    )
    if not expected:
        raise ValueError("direct decode tensors do not match the configured shape")
    tensors = (
        x,
        topk_ids,
        topk_weights,
        gather,
        slots,
        route_ids,
        probs,
        padded_offsets,
        scale,
        q_out,
        sf_out,
    )
    if any(
        not tensor.is_cuda or not tensor.is_contiguous() or tensor.device != x.device
        for tensor in tensors
    ):
        raise ValueError("direct decode tensors must be contiguous on one CUDA device")
    NVFP4BatchOneDispatchQuantKernel.compile(features, experts, topk, use_pdl, True)(
        x,
        topk_ids,
        topk_weights,
        gather,
        slots,
        route_ids,
        probs,
        padded_offsets,
        scale,
        q_out.view(torch.uint8),
        sf_out.view(torch.uint8).view(-1),
    )


def nvfp4_quantize_decode(
    x: Tensor,
    topk_ids: Tensor,
    slots: Tensor,
    cu: Tensor,
    padded_offsets: Tensor,
    scale: Tensor,
    q_out: Tensor,
    sf_out: Tensor,
    *,
    use_pdl: bool = True,
    sf_tile_rows: int = 128,
    token_major_q: bool = False,
) -> None:
    """Quantize decode activations for routed or token-major consumption."""
    tokens, features = x.shape
    topk = topk_ids.shape[1]
    experts = cu.numel() - 1
    rows = tokens * topk
    if x.dtype != torch.bfloat16 or features % FEATURE_ALIGNMENT:
        raise ValueError("decode input must be BF16 with features aligned to 256")
    if sf_tile_rows <= 0 or 128 % sf_tile_rows:
        raise ValueError("scale-factor tile rows must divide 128")
    expected = (
        topk_ids.dtype == torch.int32
        and slots.dtype == torch.int32
        and slots.numel() == rows
        and cu.dtype == torch.int32
        and tuple(cu.shape) == (experts + 1,)
        and padded_offsets.dtype == torch.int32
        and tuple(padded_offsets.shape) == (experts,)
        and scale.dtype == torch.float32
        and tuple(scale.shape) == (2,)
        and q_out.numel() == (tokens if token_major_q else rows) * features // 2
        and sf_out.dtype == torch.float8_e4m3fn
    )
    if not expected:
        raise ValueError("decode quantization tensors do not match the configured shape")
    tensors = (x, topk_ids, slots, cu, padded_offsets, scale, q_out, sf_out)
    if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("decode quantization tensors must be contiguous CUDA tensors")
    NVFP4DecodeQuantKernel.compile(
        features,
        experts,
        topk,
        use_pdl,
        sf_tile_rows,
        token_major_q,
    )(
        x,
        topk_ids,
        slots,
        cu,
        padded_offsets,
        scale,
        q_out.view(torch.uint8),
        sf_out.view(torch.uint8).view(-1),
    )


def nvfp4_dispatch_quantize_decode(
    x: Tensor,
    topk_ids: Tensor,
    topk_weights: Tensor,
    plan: Tensor,
    gather: Tensor,
    cu: Tensor,
    probs: Tensor,
    slots: Tensor,
    padded_offsets: Tensor,
    scale: Tensor,
    q_out: Tensor,
    sf_out: Tensor,
    *,
    plan_tile_rows: int,
    use_pdl: bool = True,
) -> None:
    """Build decode routing metadata and token-major NVFP4 in one launch."""
    tokens, features = x.shape
    topk = topk_ids.shape[1]
    experts = cu.numel() - 1
    routes = tokens * topk
    expected = (
        x.dtype == torch.bfloat16
        and features % FEATURE_ALIGNMENT == 0
        and 0 < routes <= DISPATCH_CHUNK
        and 0 < experts <= DISPATCH_THREADS
        and topk_ids.dtype == torch.int32
        and tuple(topk_ids.shape) == (tokens, topk)
        and topk_weights.dtype == torch.float32
        and tuple(topk_weights.shape) == (tokens, topk)
        and plan.dtype == torch.int32
        and gather.dtype == torch.int32
        and gather.numel() == routes
        and cu.dtype == torch.int32
        and tuple(cu.shape) == (experts + 1,)
        and probs.dtype == torch.float32
        and probs.numel() == routes
        and slots.dtype == torch.int32
        and slots.numel() == routes
        and padded_offsets.dtype == torch.int32
        and tuple(padded_offsets.shape) == (experts,)
        and scale.dtype == torch.float32
        and tuple(scale.shape) == (2,)
        and q_out.numel() == tokens * features // 2
        and sf_out.dtype == torch.float8_e4m3fn
        and plan_tile_rows > 0
        and 128 % plan_tile_rows == 0
    )
    if not expected:
        raise ValueError("fused decode preparation tensors do not match the configured shape")
    tensors = (
        x,
        topk_ids,
        topk_weights,
        plan,
        gather,
        cu,
        probs,
        slots,
        padded_offsets,
        scale,
        q_out,
        sf_out,
    )
    if any(
        not tensor.is_cuda or not tensor.is_contiguous() or tensor.device != x.device
        for tensor in tensors
    ):
        raise ValueError("fused decode preparation tensors must be contiguous on one CUDA device")
    NVFP4DecodeDispatchQuantKernel.compile(
        features,
        experts,
        topk,
        plan_tile_rows,
        use_pdl,
    )(
        x,
        topk_ids.view(-1),
        topk_weights.view(-1),
        plan,
        gather,
        cu,
        probs,
        slots,
        padded_offsets,
        scale,
        q_out.view(torch.uint8),
        sf_out.view(torch.uint8).view(-1),
    )


def nvfp4_gated_quantize(
    preact: Tensor,
    row_experts: Tensor,
    cu: Tensor,
    padded_offsets: Tensor,
    scale: Tensor,
    q_out: Tensor,
    sf_out: Tensor,
    *,
    activation: str = "swiglu",
    use_pdl: bool = True,
) -> None:
    """Apply a gated activation and quantize FP32 expert rows to NVFP4."""
    if preact.ndim != 2 or preact.shape[1] % 256:
        raise ValueError("preactivation must have shape [rows, 2I] with I aligned to 128")
    rows, paired_features = preact.shape
    features = paired_features // 2
    experts = cu.numel() - 1
    expected_sf_rows = -(-rows // 128) + experts
    expected = (
        preact.dtype == torch.float32
        and 0 < experts <= 256
        and row_experts.dtype == torch.int32
        and tuple(row_experts.shape) == (rows,)
        and cu.dtype == torch.int32
        and tuple(cu.shape) == (experts + 1,)
        and padded_offsets.dtype == torch.int32
        and tuple(padded_offsets.shape) == (experts,)
        and scale.dtype == torch.float32
        and tuple(scale.shape) == (2,)
        and q_out.dtype in (torch.uint8, torch.float4_e2m1fn_x2)
        and q_out.numel() == rows * features // 2
        and sf_out.dtype == torch.float8_e4m3fn
        and sf_out.numel() == expected_sf_rows * (features // 64) * 512
    )
    if not expected:
        raise ValueError("gated quantization tensors do not match the configured shape")
    tensors = (preact, row_experts, cu, padded_offsets, scale, q_out, sf_out)
    if any(
        not tensor.is_cuda or not tensor.is_contiguous() or tensor.device != preact.device
        for tensor in tensors
    ):
        raise ValueError("gated quantization tensors must be contiguous on one CUDA device")
    NVFP4GatedQuantKernel.compile(features, experts, activation, use_pdl)(
        preact,
        row_experts,
        cu,
        padded_offsets,
        scale,
        q_out.view(torch.uint8),
        sf_out.view(torch.uint8).view(-1),
    )


__all__ = [
    "nvfp4_decode_prepare",
    "nvfp4_direct_decode_prepare",
    "nvfp4_dispatch_quantize_decode",
    "nvfp4_gated_quantize",
    "nvfp4_quantize_decode",
]
