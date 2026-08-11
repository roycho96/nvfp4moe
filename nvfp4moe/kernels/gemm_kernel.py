# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Persistent grouped NVFP4 GEMM for Blackwell SM100.

Derived from NVIDIA CUTLASS's grouped block-scaled CuTe DSL example. Runtime
routing metadata remains on the GPU and expert weights use batched descriptors.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import cute, pipeline, utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .epilogue import (
    gated_backward_values,
    gated_postact_fragment,
    gated_sf_u32_word_count,
    quantize_postact_fragment,
    swiglu_backward_pair,
    validate_gated_activation,
)
from .quantize import _cvt_e2m1_pair_rn
from .scheduler import (
    MoEPersistentTileScheduler,
    MoESchedulerParams,
    MoEWorkTileInfo,
)


@dsl_user_op
def _tma_evict_first_policy(*, loc=None, ip=None) -> cutlass.Int64:
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;",
            "=l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _tma_evict_last_policy(*, loc=None, ip=None) -> cutlass.Int64:
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "createpolicy.fractional.L2::evict_last.b64 $0, 1.0;",
            "=l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class Sm100GroupedBlockScaledGemmKernel:
    """This example demonstrates an implementation of grouped blockscaled GEMM using a TMA plus Blackwell SM100 TensorCore
    warp-specialized persistent kernel.

    :param sf_vec_size: Scalefactor vector size.
    :type sf_vec_size: int
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

    :note: In current version, A and B tensors must have the same data type
        - i.e., Float8E4M3FN for A and Float8E5M2 for B is not supported

    :note: Supported combinations of A/B data types, SF data typs and SF vector size:
        - MXF8: A/B: Float8E5M2/Float8E4M3FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - MXF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - NVF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU/Float8E4M3FN + sf_vec_size: 16

    :note: Supported accumulator data types:
        - Float32

    :note: Supported C data types:
        - Float32
        - Float16/BFloat16
        - Float8E4M3FN/Float8E5M2
    :note: Constraints:
        - MMA tiler M must be 128 or 256 (use_2cta_instrs)
        - MMA tiler N must be 128/256
        - Cluster shape M must be multiple of 2 if Mma tiler M is 256
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
        - Cluster shape M/N must be <= 4 for scale factor multicasts due to limited size of scale factors
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        activation: str | None = None,
        dactivation: str | None = None,
        mma_inst_tile_k: int = 4,
    ):
        """Initializes the configuration for a Blackwell grouped blockscaled GEMM kernel.

        Besides configurations for dense persistent blockscaled GEMM, there is an extra config specific to grouped blockscaled GEMM:

        :param sf_vec_size: Scalefactor vector size.
        :type sf_vec_size: int
        :param mma_tiler_mn: tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: tuple[int, int]
        :param cluster_shape_mn: tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: tuple[int, int]
        """
        if activation is not None and dactivation is not None:
            raise ValueError("activation and dactivation are mutually exclusive")
        if mma_inst_tile_k not in (2, 4):
            raise ValueError("mma_inst_tile_k must be 2 or 4")
        self.acc_dtype = cutlass.Float32
        self.activation = None if activation is None else validate_gated_activation(activation)
        self.dactivation = None if dactivation is None else validate_gated_activation(dactivation)
        self.gated = self.activation is not None
        self.dgrad = self.dactivation is not None
        self.mma_inst_tile_k = mma_inst_tile_k
        self.sf_vec_size = sf_vec_size
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)

        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.occupancy = 1
        # Set specialized warp ids
        self.epilog_warp_id = (
            0,
            1,
            2,
            3,
        )
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.sched_warp_id = 6
        self.num_sched_stage = 3
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, self.sched_warp_id, *self.epilog_warp_id)
        )
        # Set barrier for epilogue sync and tmem ptr sync
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

    # Set up configurations that dependent on gemm inputs.
    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B/SFA/SFB
        - Computing epilogue subtile
        - Setting up A/B/SFA/SFB/C stage counts in shared memory
        - Computing A/B/SFA/SFB/C shared memory layout
        - Checking reserved smem bytes size capacity for mbar, tensor memory management and tensormap updates utilization
        """
        # Compute mma instruction shapes
        # (MMA_Tile_Shape_M, MMA_Tile_Shape_N, MMA_Inst_Shape_K)
        self.mma_inst_shape_mn = (
            self.mma_tiler[0],
            self.mma_tiler[1],
        )
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = self.mma_inst_tile_k
        self.mma_tiler = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_tile_shape_mnk = tuple(
            x * y for x, y in zip(self.cta_tile_shape_mnk, (*self.cluster_shape_mn, 1))
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Compute epilogue subtile
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )
        self.store_epi_tile = self.epi_tile
        if self.gated:
            epi_n = self.epi_tile[1]
            store_epi_n = (
                cute.recast_layout(2, 1, epi_n) if isinstance(epi_n, cute.Layout) else epi_n // 2
            )
            self.store_epi_tile = (self.epi_tile[0], store_epi_n)
        self.gated_sf_u32_words = 0
        if self.gated and self.is_nvfp4_output:
            self.gated_sf_u32_words = gated_sf_u32_word_count(
                self.cta_tile_shape_mnk[0],
                self.cta_tile_shape_mnk[1],
                cute.size(self.epi_tile[0]),
                cute.size(self.epi_tile[1]),
            )

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.store_epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
        )
        self.num_preact_stage = 0
        self.num_aux_stage = 0
        if self.dgrad:
            ab_stage_reduction = (
                2 if self.use_2cta_instrs and self.mma_inst_shape_mn[1] == 256 else 1
            )
            if self.mma_inst_tile_k == 2:
                ab_stage_reduction += 1
                if self.use_2cta_instrs and self.mma_inst_shape_mn[1] == 128:
                    ab_stage_reduction += 1
            self.num_ab_stage = max(2, self.num_ab_stage - ab_stage_reduction)
            self.num_c_stage = 2
            self.num_preact_stage = 2
            self.num_aux_stage = 2

        # A one-stage accumulator normally serializes MMA with the epilogue.
        # For the fused gated tile, alternate two TMEM banks: one holds the
        # accumulator while the other temporarily holds its scale factors.
        self.interleave_scale_tmem = self.gated and self.num_acc_stage == 1
        self.tmem_release_subtile = -1
        if self.interleave_scale_tmem:
            scale_columns = (
                (
                    cute.ceil_div(self.cta_tile_shape_mnk[0], 128)
                    + cute.ceil_div(self.cta_tile_shape_mnk[1], 128)
                )
                * 4
                * (mma_inst_shape_k // self.sf_vec_size)
            )
            self.tmem_release_subtile = scale_columns // cute.size(self.epi_tile[1])

        # Compute A/B/SFA/SFB/C shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.store_epi_tile,
            self.num_c_stage,
        )
        self.aux_smem_layout_staged = self.c_smem_layout_staged
        if self.dgrad:
            self.aux_smem_layout_staged = sm100_utils.make_smem_layout_epi(
                cutlass.BFloat16,
                self.c_layout,
                self.epi_tile,
                self.num_aux_stage,
            )

        mbar_smem_bytes = self._get_mbar_smem_bytes(
            num_acc_stage=self.num_acc_stage,
            num_ab_stage=self.num_ab_stage,
            num_c_stage=self.num_c_stage,
            num_preact_stage=self.num_preact_stage,
        )

        if (
            mbar_smem_bytes + Sm100GroupedBlockScaledGemmKernel.tensor_memory_management_bytes
            > self.reserved_smem_bytes
        ):
            raise ValueError(
                f"smem consumption for barriers {mbar_smem_bytes} exceeds the "
                f"reserved smem bytes {self.reserved_smem_bytes}"
            )

    @cute.jit
    def __call__(
        self,
        initial_a: cute.Tensor,
        initial_b: cute.Tensor,
        initial_c: cute.Tensor,
        initial_sfa: cute.Tensor,
        initial_sfb: cute.Tensor,
        data_a: cute.Tensor,
        data_b: cute.Tensor,
        data_c: cute.Tensor,
        data_preact: cute.Tensor,
        data_aux: cute.Tensor,
        data_sfa: cute.Tensor,
        data_sfb: cute.Tensor,
        cu_seqlens_m: cute.Tensor,
        group_count: cutlass.Constexpr[int],
        problem_n: cutlass.Constexpr[int],
        problem_k: cutlass.Constexpr[int],
        alpha: cute.Tensor,
        output_sf: cute.Tensor,
        output_scale: cute.Tensor,
        total_num_clusters: cutlass.Constexpr[int],
        max_active_clusters: cutlass.Constexpr[int],
        stream: cuda.CUstream,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        The initial tensors provide compile-time dtype and majorness. Runtime
        tensors and routing offsets supply addresses and expert sizes.

        :param initial_a: Initial tensor A, used for data type and majorness information.
        :type initial_a: cute.Tensor
        :param initial_b: Initial tensor B, used for data type and majorness information.
        :type initial_b: cute.Tensor
        :param initial_c: Initial tensor C, used for data type and majorness information.
        :type initial_c: cute.Tensor
        :param initial_sfa: Initial tensor SFA, used for data type and majorness information.
        :type initial_sfa: cute.Tensor
        :param initial_sfb: Initial tensor SFB, used for data type and majorness information.
        :type initial_sfb: cute.Tensor
        :param group_count: The number of GEMM groups.
        :type group_count: cutlass.Constexpr[int]
        :param total_num_clusters: Total number of clusters needed for all groups.
        :type total_num_clusters: cutlass.Constexpr[int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr[int]
        :param stream: CUDA stream for asynchronous execution.
        :type stream: cuda.CUstream
        :raises TypeError: If A and B data types do not match.
        """
        self.a_dtype = initial_a.element_type
        self.b_dtype = initial_b.element_type
        self.sf_dtype = initial_sfa.element_type
        self.c_dtype = initial_c.element_type
        self.is_nvfp4_output = self.c_dtype is cutlass.Float4E2M1FN
        self.a_major_mode = utils.LayoutEnum.from_tensor(initial_a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(initial_b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(initial_c)
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()
        store_n = problem_n
        if cutlass.const_expr(self.gated):
            store_n = problem_n // 2

        # A and SFA use global descriptors with expert-local offsets. B and SFB
        # use batched descriptors. The output layout encodes an expert-local
        # logical window so one static TMA descriptor can suppress tail stores.
        total_rows = data_a.shape[0]
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)
        runtime_a_ptr = cute.make_ptr(
            self.a_dtype,
            data_a.iterator.toint(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        initial_a = cute.make_tensor(
            runtime_a_ptr,
            cute.make_layout(
                (total_rows, problem_k, c1),
                stride=(problem_k, c1, c0),
            ),
        )
        runtime_b_ptr = cute.make_ptr(
            self.b_dtype,
            data_b.iterator.toint(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        initial_b = cute.make_tensor(
            runtime_b_ptr,
            cute.make_layout(
                (problem_n, problem_k, group_count),
                stride=(problem_k, c1, problem_n * problem_k),
            ),
        )
        runtime_c_ptr = cute.make_ptr(
            self.c_dtype,
            data_c.iterator.toint(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        initial_c = cute.make_tensor(
            runtime_c_ptr - total_rows * store_n,
            cute.make_layout(
                (total_rows, store_n, (c1, total_rows + c1)),
                stride=(store_n, c1, (c0, store_n)),
            ),
        )

        runtime_preact_ptr = cute.make_ptr(
            data_preact.element_type,
            data_preact.iterator.toint(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        preact = cute.make_tensor(
            cute.recast_ptr(runtime_preact_ptr, dtype=cutlass.Int32),
            cute.make_layout(
                (total_rows, problem_n, c1),
                stride=(problem_n, c1, c0),
            ),
        )
        runtime_aux_ptr = cute.make_ptr(
            data_aux.element_type,
            data_aux.iterator.toint(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        aux = cute.make_tensor(
            runtime_aux_ptr - total_rows * problem_n,
            cute.make_layout(
                (total_rows, problem_n, (c1, total_rows + c1)),
                stride=(problem_n, c1, (c0, problem_n)),
            ),
        )

        padded_sfa_rows = data_sfa.shape[1] * 128
        runtime_sfa_ptr = cute.make_ptr(
            self.sf_dtype,
            data_sfa.iterator.toint(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            (padded_sfa_rows, problem_k, c1), self.sf_vec_size
        )
        initial_sfa = cute.make_tensor(runtime_sfa_ptr, sfa_layout)

        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            (problem_n, problem_k, group_count), self.sf_vec_size
        )
        runtime_sfb_ptr = cute.make_ptr(
            self.sf_dtype,
            data_sfb.iterator.toint(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        initial_sfb = cute.make_tensor(runtime_sfb_ptr, sfb_layout)

        if cutlass.const_expr(self.gated and self.is_nvfp4_output):
            padded_output_rows = output_sf.shape[1] * 128
            output_sf_ptr = cute.make_ptr(
                output_sf.element_type,
                output_sf.iterator.toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            output_sf_layout = blockscaled_utils.tile_atom_to_shape_SF(
                (padded_output_rows, problem_n, c1),
                self.sf_vec_size * 2,
            )
            output_sf = cute.make_tensor(output_sf_ptr, output_sf_layout)

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            initial_a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            initial_b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for SFA
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            initial_sfa,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Setup TMA load for SFB
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            initial_sfb,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        # Setup TMA store for C
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            initial_c,
            epi_smem_layout,
            self.store_epi_tile,
        )
        tma_atom_preact = tma_atom_c
        tma_tensor_preact = tma_tensor_c
        tma_atom_aux = tma_atom_c
        tma_tensor_aux = tma_tensor_c
        if cutlass.const_expr(self.dgrad):
            preact_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
            tma_atom_preact, tma_tensor_preact = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                preact,
                preact_smem_layout,
                self.epi_tile,
            )
            aux_smem_layout = cute.slice_(self.aux_smem_layout_staged, (None, None, 0))
            tma_atom_aux, tma_tensor_aux = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                aux,
                aux_smem_layout,
                self.epi_tile,
            )

        # Compute grid size
        _, grid = self._compute_grid(total_num_clusters, self.cluster_shape_mn, max_active_clusters)
        moe_sched_params = MoESchedulerParams(
            scenario="2Dx3D",
            expert_shape=(group_count, problem_n, problem_k),
            cta_tile_shape_mnk=self.cta_tile_shape_mnk,
            cluster_shape_mn=self.cluster_shape_mn,
        )

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        SchedulerStorage = MoEPersistentTileScheduler.make_storage_struct(
            self.num_sched_stage,
            use_dynamic_sched=False,
        )
        PreactBarriers = cute.struct.MemRange[cutlass.Int32, 0]
        PreactStorage = cute.struct.MemRange[cutlass.Int32, 0]
        AuxStorage = cute.struct.MemRange[cutlass.Int32, 0]
        if cutlass.const_expr(self.dgrad):
            PreactBarriers = cute.struct.MemRange[
                cutlass.Int64,
                self.num_preact_stage * 2,
            ]
            PreactStorage = cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            AuxStorage = cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16,
                    cute.cosize(self.aux_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            preact_mbar_ptr: PreactBarriers
            scheduler: SchedulerStorage
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            sP: PreactStorage
            sAux: AuxStorage
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            tma_atom_preact,
            tma_tensor_preact,
            tma_atom_aux,
            tma_tensor_aux,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.aux_smem_layout_staged,
            self.epi_tile,
            self.store_epi_tile,
            moe_sched_params,
            group_count,
            cu_seqlens_m,
            alpha,
            output_sf,
            output_scale,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tma_atom_preact: cute.CopyAtom,
        mPreact_mnl: cute.Tensor,
        tma_atom_aux: cute.CopyAtom,
        mAux_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: cute.Layout | cute.ComposedLayout,
        aux_smem_layout_staged: cute.Layout | cute.ComposedLayout,
        epi_tile: cute.Tile,
        store_epi_tile: cute.Tile,
        moe_sched_params: MoESchedulerParams,
        group_count: cutlass.Constexpr,
        cu_seqlens_m: cute.Tensor,
        alpha: cute.Tensor,
        output_sf: cute.Tensor,
        output_scale: cute.Tensor,
    ):
        """
        GPU device kernel performing the grouped GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        if warp_idx == self.tma_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sfa)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sfb)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)
            if cutlass.const_expr(self.dgrad):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_preact)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_aux)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        k_tile_count = cutlass.Int32(
            cute.ceil_div(
                cute.size(mA_mkl, mode=[1]),
                self.cta_tile_shape_mnk[2],
            )
        )

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, _bidy, _bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        # Allocate pipeline barriers and operand storage.
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr
        tmem_holding_buf = storage.tmem_holding_buf

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        sched_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32)
        sched_consumer_threads = 32 * len(
            (self.tma_warp_id, self.mma_warp_id, *self.epilog_warp_id)
        )
        sched_pipeline = pipeline.PipelineAsync.create(
            num_stages=self.num_sched_stage,
            producer_group=sched_producer_group,
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                sched_consumer_threads,
            ),
            barrier_storage=storage.scheduler.tile_info_mbar.data_ptr(),
            defer_sync=True,
        )
        preact_pipeline = None
        if cutlass.const_expr(self.dgrad):
            preact_pipeline = pipeline.PipelineTmaAsync.create(
                num_stages=self.num_preact_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    len(self.epilog_warp_id),
                ),
                tx_count=cute.size_in_bytes(
                    self.c_dtype,
                    cute.slice_(c_smem_layout_staged, (None, None, 0)),
                ),
                barrier_storage=storage.preact_mbar_ptr.data_ptr(),
                defer_sync=True,
            )

        # Tensor memory dealloc barrier init
        if use_2cta_instrs:  # noqa: SIM102
            if warp_idx == self.tma_warp_id:
                num_tmem_dealloc_threads = 32
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads)

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        #
        # Setup smem tensor A/B/SFA/SFB/C
        #
        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        sP = None
        sAux = None
        if cutlass.const_expr(self.dgrad):
            sP = storage.sP.get_tensor(
                c_smem_layout_staged.outer,
                swizzle=c_smem_layout_staged.inner,
            )
            sAux = storage.sAux.get_tensor(
                aux_smem_layout_staged.outer,
                swizzle=aux_smem_layout_staged.inner,
            )
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        # (MMA, MMA_M, MMA_K, STAGE)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (MMA, MMA_N, MMA_K, STAGE)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        sched_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Int32,
            num_bits_per_copy=128,
        )
        sched_buf = cute.make_tensor(
            storage.scheduler.sInfo.data_ptr(),
            cute.make_layout((4, self.num_sched_stage), stride=(1, 4)),
        )

        #
        # Compute multicast mask for A/B/SFA/SFB buffer full
        #
        b_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_b_mcast or use_2cta_instrs):
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        #
        # Partition global tensors for TiledMMA A/B
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # TMA load scaled factor B partition_S/D
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        physical_acc_stages = 2 if self.interleave_scale_tmem else self.num_acc_stage
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, physical_acc_stages))

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        grid_dim = cute.arch.grid_dim()
        #
        # Persistent tile scheduling loop
        #
        # When the problem shapes are on device, we launch one CTA per SM.
        # The if condition later prevents the warps from extra CTAs from doing any work.
        expert_ends = cute.make_tensor(
            cu_seqlens_m.iterator + 1,
            cute.make_layout(group_count),
        )
        tile_sched = MoEPersistentTileScheduler.create(
            moe_sched_params,
            expert_ends,
            cute.arch.block_idx(),
            grid_dim,
        )

        if warp_idx == self.sched_warp_id:
            sched_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.num_sched_stage,
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                sched_pipeline.producer_acquire(sched_producer_state)
                cute.copy(
                    sched_copy_atom,
                    work_tile.to_rmem_tensor(),
                    sched_buf[(None, sched_producer_state.index)],
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                sched_pipeline.producer_commit(sched_producer_state)
                sched_producer_state.advance()
                work_tile = tile_sched.advance_to_next_work()

            sched_pipeline.producer_acquire(sched_producer_state)
            sentinel = MoEWorkTileInfo(
                cutlass.Int32(-1),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
            )
            cute.copy(
                sched_copy_atom,
                sentinel.to_rmem_tensor(),
                sched_buf[(None, sched_producer_state.index)],
            )
            cute.arch.fence_proxy("async.shared", space="cta")
            sched_pipeline.producer_commit(sched_producer_state)
            sched_producer_state.advance()
            sched_pipeline.producer_tail(sched_producer_state)

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            #
            # Persistent tile scheduling loop
            #
            sched_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_sched_stage,
            )
            sched_pipeline.consumer_wait(sched_consumer_state)
            sched_rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
            cute.copy(
                sched_copy_atom,
                sched_buf[(None, sched_consumer_state.index)],
                sched_rmem,
            )
            work_tile = MoEWorkTileInfo.from_rmem_tensor(sched_rmem)
            cute.arch.fence_acq_rel_cta()
            sched_pipeline.consumer_release(sched_consumer_state)
            sched_consumer_state.advance()

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            preact_producer_state = None
            if cutlass.const_expr(self.dgrad):
                preact_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    self.num_preact_stage,
                )
            reuse_cache_policy = None
            stream_cache_policy = None
            if cutlass.const_expr(self.mma_inst_tile_k == 2):
                # Small shards reuse A across output tiles; B remains streaming.
                if cutlass.const_expr(group_count <= 128):
                    reuse_cache_policy = _tma_evict_last_policy()
                stream_cache_policy = _tma_evict_first_policy()

            while work_tile.is_valid_tile:
                cur_k_tile_cnt = k_tile_count
                cur_group_idx = work_tile.expert_idx
                is_k_tile_cnt_zero = cur_k_tile_cnt == 0
                sched_pipeline.consumer_wait(sched_consumer_state)
                cute.copy(
                    sched_copy_atom,
                    sched_buf[(None, sched_consumer_state.index)],
                    sched_rmem,
                )
                next_work_tile = MoEWorkTileInfo.from_rmem_tensor(sched_rmem)
                cute.arch.fence_acq_rel_cta()
                sched_pipeline.consumer_release(sched_consumer_state)
                sched_consumer_state.advance()
                # Do not load any data if cur_k_tile_cnt is 0
                if not is_k_tile_cnt_zero:
                    mma_tile_coord_mnl = (
                        work_tile.tile_m_idx // cute.size(tiled_mma.thr_id.shape),
                        work_tile.tile_n_idx,
                        cur_group_idx,
                    )

                    expert_row = cu_seqlens_m[cur_group_idx]
                    mA_mk = cute.domain_offset(
                        (expert_row, None),
                        mA_mkl[(None, None, 0)],
                    )
                    gA_mk = cute.local_tile(
                        mA_mk,
                        cute.select(self.mma_tiler, [0, 2]),
                        (mma_tile_coord_mnl[0], None),
                    )
                    tCgA_tile = thr_mma.partition_A(gA_mk)
                    tCgA_staged = cute.group_modes(
                        tCgA_tile,
                        0,
                        cute.rank(tCgA_tile) - 1,
                    )
                    sA_staged = cute.group_modes(
                        sA,
                        0,
                        cute.rank(sA) - 1,
                    )

                    sfa_tile = expert_row // 128 + cur_group_idx
                    mSFA_mk = cute.domain_offset(
                        ((None, sfa_tile), None),
                        mSFA_mkl[(None, None, 0)],
                    )
                    gSFA_mk = cute.local_tile(
                        mSFA_mk,
                        cute.select(self.mma_tiler, [0, 2]),
                        (mma_tile_coord_mnl[0], None),
                    )
                    tCgSFA_tile = thr_mma.partition_A(gSFA_mk)
                    tCgSFA_staged = cute.group_modes(
                        tCgSFA_tile,
                        0,
                        cute.rank(tCgSFA_tile) - 1,
                    )
                    sSFA_staged = cute.group_modes(
                        sSFA,
                        0,
                        cute.rank(sSFA) - 1,
                    )

                    #
                    # Slice to per mma tile index
                    #
                    # ((atom_v, rest_v), RestK)
                    tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

                    # ((atom_v, rest_v), RestK)
                    tBgSFB_slice = tBgSFB[
                        (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                    ]

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                    ab_producer_state.reset_count()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < cur_k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                    #
                    # Tma load loop
                    #
                    for k_tile in cutlass.range(0, cur_k_tile_cnt, 1, unroll=1):
                        # Conditionally wait for AB buffer empty
                        ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)

                        # TMA load A/B/SFA/SFB
                        utils.block_copy(
                            tma_atom_a,
                            tCgA_staged[(None, ab_producer_state.count)],
                            sA_staged[(None, ab_producer_state.index)],
                            tma_multicast={
                                "cluster_shape": self.cluster_shape_mn,
                                "multicast_dim": "M",
                                "use_2cta_mma_inst": use_2cta_instrs,
                            },
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            cache_policy=reuse_cache_policy,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_slice[(None, ab_producer_state.count)],
                            tBsB[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            mcast_mask=b_full_mcast_mask,
                            cache_policy=stream_cache_policy,
                        )
                        utils.block_copy(
                            tma_atom_sfa,
                            tCgSFA_staged[(None, ab_producer_state.count)],
                            sSFA_staged[(None, ab_producer_state.index)],
                            tma_multicast={
                                "cluster_shape": self.cluster_shape_mn,
                                "multicast_dim": "M",
                                "use_2cta_mma_inst": use_2cta_instrs,
                            },
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            cache_policy=reuse_cache_policy,
                        )
                        cute.copy(
                            tma_atom_sfb,
                            tBgSFB_slice[(None, ab_producer_state.count)],
                            tBsSFB[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            mcast_mask=sfb_full_mcast_mask,
                            cache_policy=stream_cache_policy,
                        )

                        # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                        ab_producer_state.advance()
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if ab_producer_state.count < cur_k_tile_cnt:
                            peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                                ab_producer_state
                            )
                    if cutlass.const_expr(self.dgrad):
                        expert_preact = cute.domain_offset(
                            (expert_row, None, None),
                            mPreact_mnl,
                        )
                        preact_tile = cute.local_tile(
                            expert_preact,
                            (*self.cta_tile_shape_mnk[:2], 1),
                            (work_tile.tile_m_idx, work_tile.tile_n_idx, 0),
                        )
                        preact_subtiles = cute.zipped_divide(
                            preact_tile,
                            (*epi_tile, 1),
                        )
                        preact_gmem = cute.group_modes(
                            preact_subtiles,
                            0,
                            cute.rank(preact_subtiles) - 1,
                        )
                        preact_smem = cute.group_modes(
                            sP,
                            0,
                            cute.rank(sP) - 1,
                        )
                        preact_subtile_count = cutlass.const_expr(
                            (self.cta_tile_shape_mnk[0] // cute.size(epi_tile[0]))
                            * (self.cta_tile_shape_mnk[1] // cute.size(epi_tile[1]))
                        )
                        for preact_subtile in cutlass.range_constexpr(preact_subtile_count):
                            preact_pipeline.producer_acquire(preact_producer_state)
                            utils.block_copy(
                                tma_atom_preact,
                                preact_gmem[(None, preact_subtile)],
                                preact_smem[(None, preact_producer_state.index)],
                                tma_bar_ptr=preact_pipeline.producer_get_barrier(
                                    preact_producer_state
                                ),
                            )
                            preact_producer_state.advance()
                #
                # Advance to next tile
                #
                work_tile = next_work_tile

            #
            # Wait A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)
            if cutlass.const_expr(self.dgrad):
                preact_pipeline.producer_tail(preact_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            self.tmem_alloc_barrier.arrive_and_wait()

            #
            # Retrieving tensor memory ptr and make accumulator/SFA/SFB tensor
            #
            # Make accumulator tmem tensor
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Make SFA tmem tensor
            acc_bank = (
                tCtAcc_base[(None, None, None, 0)]
                if cutlass.const_expr(self.interleave_scale_tmem)
                else tCtAcc_base
            )
            acc_tmem_col_offset = tcgen05.find_tmem_tensor_col_offset(acc_bank)
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + acc_tmem_col_offset,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA_base = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # Make SFB tmem tensor
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr
                + acc_tmem_col_offset
                + tcgen05.find_tmem_tensor_col_offset(tCtSFA_base),
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB_base = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            #
            # Partition for S2T copy of SFA/SFB
            #
            tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t_base = (
                self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA_base)
            )
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t_base = (
                self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB_base)
            )

            #
            # Persistent tile scheduling loop
            #
            sched_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_sched_stage,
            )
            sched_pipeline.consumer_wait(sched_consumer_state)
            sched_rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
            cute.copy(
                sched_copy_atom,
                sched_buf[(None, sched_consumer_state.index)],
                sched_rmem,
            )
            work_tile = MoEWorkTileInfo.from_rmem_tensor(sched_rmem)
            cute.arch.fence_acq_rel_cta()
            sched_pipeline.consumer_release(sched_consumer_state)
            sched_consumer_state.advance()

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            while work_tile.is_valid_tile:
                cur_k_tile_cnt = k_tile_count
                is_k_tile_cnt_zero = cur_k_tile_cnt == 0
                sched_pipeline.consumer_wait(sched_consumer_state)
                cute.copy(
                    sched_copy_atom,
                    sched_buf[(None, sched_consumer_state.index)],
                    sched_rmem,
                )
                next_work_tile = MoEWorkTileInfo.from_rmem_tensor(sched_rmem)
                cute.arch.fence_acq_rel_cta()
                sched_pipeline.consumer_release(sched_consumer_state)
                sched_consumer_state.advance()

                if cutlass.const_expr(self.interleave_scale_tmem):
                    scale_bank_delta = cute.assume(
                        acc_tmem_col_offset * (acc_producer_state.phase - 1),
                        divby=acc_tmem_col_offset,
                    )
                    tCtSFA = cute.make_tensor(
                        cute.recast_ptr(
                            cute.recast_ptr(tCtSFA_base.iterator, dtype=self.acc_dtype)
                            + scale_bank_delta,
                            dtype=self.sf_dtype,
                        ),
                        tCtSFA_base.layout,
                    )
                    tCtSFB = cute.make_tensor(
                        cute.recast_ptr(
                            cute.recast_ptr(tCtSFB_base.iterator, dtype=self.acc_dtype)
                            + scale_bank_delta,
                            dtype=self.sf_dtype,
                        ),
                        tCtSFB_base.layout,
                    )
                    tCtSFA_compact_s2t = cute.make_tensor(
                        cute.recast_ptr(
                            cute.recast_ptr(
                                tCtSFA_compact_s2t_base.iterator,
                                dtype=self.acc_dtype,
                            )
                            + scale_bank_delta,
                            dtype=self.sf_dtype,
                        ),
                        tCtSFA_compact_s2t_base.layout,
                    )
                    tCtSFB_compact_s2t = cute.make_tensor(
                        cute.recast_ptr(
                            cute.recast_ptr(
                                tCtSFB_compact_s2t_base.iterator,
                                dtype=self.acc_dtype,
                            )
                            + scale_bank_delta,
                            dtype=self.sf_dtype,
                        ),
                        tCtSFB_compact_s2t_base.layout,
                    )
                    acc_stage = acc_producer_state.phase ^ 1
                else:
                    tCtSFA = tCtSFA_base
                    tCtSFB = tCtSFB_base
                    tCtSFA_compact_s2t = tCtSFA_compact_s2t_base
                    tCtSFB_compact_s2t = tCtSFB_compact_s2t_base
                    acc_stage = acc_producer_state.index

                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage)]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < cur_k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta and not is_k_tile_cnt_zero:
                    acc_pipeline.producer_acquire(acc_producer_state)

                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # Mma mainloop
                #
                for k_tile in range(cur_k_tile_cnt):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)

                        #  Copy SFA/SFB from smem to tmem
                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_consumer_state.index,
                        )
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                        tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t_staged,
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t_staged,
                            tCtSFB_compact_s2t,
                        )

                        # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )

                            # Set SFA/SFB tensor to tiled_mma
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kblock_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB[sf_kblock_coord].iterator,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )

                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < cur_k_tile_cnt:  # noqa: SIM102
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                #
                # Async arrive accumulator buffer full
                #
                if not is_k_tile_cnt_zero:
                    if is_leader_cta:
                        acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()

                #
                # Advance to next tile
                #
                work_tile = next_work_tile

            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # Alloc tensor memory buffer
            #
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    self.num_tmem_alloc_cols,
                    tmem_holding_buf,
                    is_two_cta=use_2cta_instrs,
                )

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            self.tmem_alloc_barrier.arrive_and_wait()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Partition for epilogue
            #
            epi_tidx = tidx
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc, tTR_cAcc = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, epi_tile, use_2cta_instrs
            )

            tiled_copy_r2s = None
            tRS_rC = None
            tRS_sC = None
            aux_tiled_copy_r2s = None
            tRS_rAux = None
            tRS_sAux = None
            tiled_copy_s2r = None
            tSR_rP = None
            tSR_sP = None
            if cutlass.const_expr(self.gated):
                copy_atom_r2s = sm100_utils.get_smem_store_op(
                    self.c_layout,
                    self.c_dtype,
                    self.acc_dtype,
                    tiled_copy_t2r,
                )
                full_tiled_copy_r2s = cute.make_tiled_copy_D(
                    copy_atom_r2s,
                    tiled_copy_t2r,
                )
                tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s,
                    full_tiled_copy_r2s,
                )
                tRS_sC = tiled_copy_r2s.get_slice(epi_tidx).partition_D(sC)
            else:
                tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
                tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                    tiled_copy_t2r, tTR_rC, epi_tidx, sC
                )
                if cutlass.const_expr(self.dgrad):
                    copy_atom_s2r = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        self.c_dtype,
                    )
                    tiled_copy_s2r = cute.make_tiled_copy_D(
                        copy_atom_s2r,
                        tiled_copy_t2r,
                    )
                    thr_copy_s2r = tiled_copy_s2r.get_slice(epi_tidx)
                    tSR_sP = thr_copy_s2r.partition_D(sP)
                    tSR_rP = tiled_copy_s2r.retile(tTR_rC)
                    aux_copy_atom_r2s = sm100_utils.get_smem_store_op(
                        self.c_layout,
                        cutlass.BFloat16,
                        self.acc_dtype,
                        tiled_copy_t2r,
                    )
                    aux_tiled_copy_r2s = cute.make_tiled_copy_D(
                        aux_copy_atom_r2s,
                        tiled_copy_t2r,
                    )
                    tTR_rAux = cute.make_rmem_tensor(tTR_rAcc.shape, cutlass.BFloat16)
                    tRS_rAux = aux_tiled_copy_r2s.retile(tTR_rAux)
                    tRS_sAux = aux_tiled_copy_r2s.get_slice(epi_tidx).partition_D(sAux)
            c_tma_smem = cute.group_modes(sC, 0, cute.rank(sC) - 1)
            aux_tma_smem = None
            if cutlass.const_expr(self.dgrad):
                aux_tma_smem = cute.group_modes(sAux, 0, cute.rank(sAux) - 1)

            #
            # Persistent tile scheduling loop
            #
            sched_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_sched_stage,
            )
            sched_pipeline.consumer_wait(sched_consumer_state)
            sched_rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
            cute.copy(
                sched_copy_atom,
                sched_buf[(None, sched_consumer_state.index)],
                sched_rmem,
            )
            work_tile = MoEWorkTileInfo.from_rmem_tensor(sched_rmem)
            cute.arch.fence_acq_rel_cta()
            sched_pipeline.consumer_release(sched_consumer_state)
            sched_consumer_state.advance()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            preact_consumer_state = None
            if cutlass.const_expr(self.dgrad):
                preact_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer,
                    self.num_preact_stage,
                )

            # Threads/warps participating in tma store pipeline
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_store_stages = self.num_c_stage
            if cutlass.const_expr(self.dgrad):
                c_store_stages = self.num_aux_stage
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=c_store_stages,
                producer_group=c_producer_group,
            )
            num_tiles_executed = cutlass.Int32(0)
            output_inv_pts = cutlass.Float32(1.0)
            if cutlass.const_expr(self.is_nvfp4_output):
                output_inv_pts = cutlass.Float32(output_scale[1])
            alpha_value = cutlass.Float32(alpha[0])

            while work_tile.is_valid_tile:
                cur_group_idx = work_tile.expert_idx
                cur_k_tile_cnt = k_tile_count
                is_k_tile_cnt_zero = cur_k_tile_cnt == 0
                sched_pipeline.consumer_wait(sched_consumer_state)
                cute.copy(
                    sched_copy_atom,
                    sched_buf[(None, sched_consumer_state.index)],
                    sched_rmem,
                )
                next_work_tile = MoEWorkTileInfo.from_rmem_tensor(sched_rmem)
                cute.arch.fence_acq_rel_cta()
                sched_pipeline.consumer_release(sched_consumer_state)
                sched_consumer_state.advance()
                expert_row = cutlass.Int64(cu_seqlens_m[cur_group_idx])
                expert_rows = cu_seqlens_m[cur_group_idx + 1] - cu_seqlens_m[cur_group_idx]
                c_window = cute.domain_offset(
                    (
                        mC_mnl.shape[0] - expert_rows,
                        0,
                        (0, expert_row + expert_rows),
                    ),
                    mC_mnl,
                )
                output_cta_n = self.cta_tile_shape_mnk[1]
                if cutlass.const_expr(self.gated):
                    output_cta_n //= 2
                c_tile = cute.local_tile(
                    c_window,
                    (self.cta_tile_shape_mnk[0], output_cta_n, 1),
                    (work_tile.tile_m_idx, work_tile.tile_n_idx, (0, 0)),
                )
                c_subtiles = cute.zipped_divide(c_tile, (*store_epi_tile, 1))
                c_tma_gmem = cute.group_modes(
                    c_subtiles,
                    0,
                    cute.rank(c_subtiles) - 1,
                )
                aux_tma_gmem = None
                if cutlass.const_expr(self.dgrad):
                    aux_window = cute.domain_offset(
                        (
                            mAux_mnl.shape[0] - expert_rows,
                            0,
                            (0, expert_row + expert_rows),
                        ),
                        mAux_mnl,
                    )
                    aux_tile = cute.local_tile(
                        aux_window,
                        (*self.cta_tile_shape_mnk[:2], 1),
                        (work_tile.tile_m_idx, work_tile.tile_n_idx, (0, 0)),
                    )
                    aux_subtiles = cute.zipped_divide(aux_tile, (*epi_tile, 1))
                    aux_tma_gmem = cute.group_modes(
                        aux_subtiles,
                        0,
                        cute.rank(aux_subtiles) - 1,
                    )

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
                acc_stage = (
                    acc_consumer_state.phase
                    if cutlass.const_expr(self.interleave_scale_tmem)
                    else acc_consumer_state.index
                )
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage)]

                #
                # Wait for accumulator buffer full
                #
                if not is_k_tile_cnt_zero:
                    acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                tTR_cAcc_subtiles = cute.group_modes(tTR_cAcc, 3, cute.rank(tTR_cAcc))

                #
                # Store accumulator to global memory in subtiles
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = num_tiles_executed * subtile_cnt
                sf_tile_base = cutlass.Int64(0)
                use_logical_sf_store = cutlass.const_expr(
                    self.gated_sf_u32_words > 0
                    and (mC_mnl.shape[1] * 2) % self.cta_tile_shape_mnk[1] == 0
                )
                if cutlass.const_expr(self.is_nvfp4_output and not use_logical_sf_store):
                    sf_atoms_per_row = mC_mnl.shape[1] // 64
                    sf_tile = expert_row // 128 + cur_group_idx + work_tile.tile_m_idx
                    sf_tile_base = (
                        sf_tile * sf_atoms_per_row * 512
                        + work_tile.tile_n_idx * (self.cta_tile_shape_mnk[1] // 128) * 512
                    )
                sf_i16_staging = None
                if cutlass.const_expr(self.gated_sf_u32_words > 0):
                    assert subtile_cnt == self.gated_sf_u32_words * 2
                    sf_i16_staging = cute.make_rmem_tensor(
                        cute.make_layout(subtile_cnt),
                        cutlass.Int16,
                    )
                    if cutlass.const_expr(use_logical_sf_store):
                        sf_expert_row = (expert_row // 128 + cur_group_idx) * 128
                        sf_cta_in_bounds = (
                            work_tile.tile_m_idx * self.cta_tile_shape_mnk[0]
                            < cute.ceil_div(expert_rows, 128) * 128
                        )
                        expert_output_sf = cute.domain_offset(
                            (sf_expert_row, 0),
                            output_sf[(None, None, 0)],
                        )
                        sf_tile = cute.local_tile(
                            expert_output_sf,
                            self.cta_tile_shape_mnk[:2],
                            (work_tile.tile_m_idx, work_tile.tile_n_idx),
                        )
                        sf_tile_epi = cute.flat_divide(sf_tile, epi_tile)
                        tTR_gSF = tiled_copy_t2r.get_slice(epi_tidx).partition_D(sf_tile_epi)
                    else:
                        first_coords = tTR_cAcc_subtiles[(None, None, None, 0)]
                        first_output_pairs = cute.flat_divide(
                            first_coords,
                            cute.make_layout(2),
                        )
                        first_output_coords = first_output_pairs[
                            (0,) + (None,) * (cute.rank(first_output_pairs) - 1)
                        ]
                        first_coord = first_output_coords[0]
                        sf_local_row = (
                            work_tile.tile_m_idx * self.cta_tile_shape_mnk[0] + first_coord[0]
                        )
                        sf_row_in_bounds = sf_local_row < expert_rows
                        sf_row_offset = (
                            sf_tile_base + (sf_local_row % 32) * 16 + ((sf_local_row // 32) % 4) * 4
                        )
                for subtile_idx in range(subtile_cnt):
                    if not is_k_tile_cnt_zero:
                        #
                        # Load accumulator from tensor memory buffer to register
                        #
                        tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                        if cutlass.const_expr(self.interleave_scale_tmem):  # noqa: SIM102
                            if subtile_idx == self.tmem_release_subtile:
                                cute.arch.fence_view_async_tmem_load()
                                with cute.arch.elect_one():
                                    acc_pipeline.consumer_release(acc_consumer_state)

                    elif cutlass.const_expr(self.gated or self.dgrad):
                        tTR_rAcc.fill(0)

                    if cutlass.const_expr(self.gated):
                        postact = gated_postact_fragment(
                            tTR_rAcc,
                            alpha_value,
                            self.activation,
                        )
                        if cutlass.const_expr(self.is_nvfp4_output):
                            scaled_postact, output_sf_values = quantize_postact_fragment(
                                postact,
                                output_inv_pts,
                            )
                            packed_postact = cute.make_rmem_tensor(
                                cute.make_layout(cute.size(scaled_postact) // 2),
                                cutlass.Uint8,
                            )
                            for pair_idx in cutlass.range_constexpr(cute.size(packed_postact)):
                                packed_postact[pair_idx] = cutlass.Uint8(
                                    _cvt_e2m1_pair_rn(
                                        scaled_postact[pair_idx * 2 + 1],
                                        scaled_postact[pair_idx * 2],
                                    )
                                    & 0xFF
                                )
                            postact_out = cute.make_tensor(
                                cute.recast_ptr(
                                    packed_postact.iterator,
                                    dtype=self.c_dtype,
                                ),
                                scaled_postact.layout,
                            )
                            coords = tTR_cAcc_subtiles[(None, None, None, subtile_idx)]
                            output_pairs = cute.flat_divide(
                                coords,
                                cute.make_layout(2),
                            )
                            output_coords = output_pairs[
                                (0,) + (None,) * (cute.rank(output_pairs) - 1)
                            ]
                            sf_group_count = cute.size(output_sf_values)
                            if cutlass.const_expr(sf_group_count == 2):
                                sf_values_i16 = cute.recast_tensor(
                                    output_sf_values,
                                    cutlass.Int16,
                                )
                                if cutlass.const_expr(self.gated_sf_u32_words > 0):
                                    sf_i16_staging[subtile_idx] = sf_values_i16[0]
                                else:
                                    coord = output_coords[0]
                                    local_row = (
                                        work_tile.tile_m_idx * self.cta_tile_shape_mnk[0] + coord[0]
                                    )
                                    output_col = (
                                        work_tile.tile_n_idx * (self.cta_tile_shape_mnk[1] // 2)
                                        + coord[1] // 2
                                    )
                                    if (
                                        local_row < expert_rows
                                        and output_col + 16 < mC_mnl.shape[1]
                                    ):
                                        sf_offset = (
                                            sf_tile_base
                                            + (coord[1] // 128) * 512
                                            + (local_row % 32) * 16
                                            + ((local_row // 32) % 4) * 4
                                            + (coord[1] // 32) % 4
                                        )
                                        sf_ptr = cute.make_tensor(
                                            cute.recast_ptr(
                                                output_sf.iterator + sf_offset,
                                                dtype=cutlass.Int16,
                                            ).align(2),
                                            cute.make_layout(1),
                                        )
                                        sf_ptr[0] = sf_values_i16[0]
                            else:
                                for sf_group in cutlass.range_constexpr(sf_group_count):
                                    coord = output_coords[sf_group * 16]
                                    local_row = (
                                        work_tile.tile_m_idx * self.cta_tile_shape_mnk[0] + coord[0]
                                    )
                                    output_col = (
                                        work_tile.tile_n_idx * (self.cta_tile_shape_mnk[1] // 2)
                                        + coord[1] // 2
                                    )
                                    if local_row < expert_rows and output_col < mC_mnl.shape[1]:
                                        sf_tile = (
                                            expert_row // 128 + cur_group_idx + local_row // 128
                                        )
                                        sf_col = output_col // 16
                                        sf_atoms_per_row = mC_mnl.shape[1] // 64
                                        sf_offset = (
                                            sf_tile * sf_atoms_per_row * 512
                                            + (sf_col // 4) * 512
                                            + (local_row % 32) * 16
                                            + ((local_row // 32) % 4) * 4
                                            + sf_col % 4
                                        )
                                        sf_ptr = cute.make_tensor(
                                            output_sf.iterator + sf_offset,
                                            cute.make_layout(1),
                                        )
                                        sf_ptr[0] = output_sf_values[sf_group]
                        else:
                            postact_out = cute.make_rmem_tensor(postact.layout, self.c_dtype)
                            for value_idx in cutlass.range_constexpr(cute.size(postact)):
                                postact_out[value_idx] = postact[value_idx].to(self.c_dtype)
                        c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                        if warp_idx == self.epilog_warp_id[0]:
                            c_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()
                        cute.copy(
                            tiled_copy_r2s,
                            tiled_copy_r2s.retile(postact_out),
                            tRS_sC[(None, None, None, c_buffer)],
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        if warp_idx == self.epilog_warp_id[0]:
                            utils.block_copy(
                                tma_atom_c,
                                c_tma_smem[(None, c_buffer)],
                                c_tma_gmem[(None, subtile_idx)],
                            )
                            c_pipeline.producer_commit()
                    elif cutlass.const_expr(self.dgrad):
                        retiled_acc = tiled_copy_r2s.retile(tTR_rAcc)
                        flat_acc = cute.group_modes(retiled_acc, 0, cute.rank(retiled_acc))
                        flat_out = cute.group_modes(tRS_rC, 0, cute.rank(tRS_rC))
                        flat_aux = cute.group_modes(tRS_rAux, 0, cute.rank(tRS_rAux))
                        preact_pipeline.consumer_wait(preact_consumer_state)
                        cute.copy(
                            tiled_copy_s2r,
                            tSR_sP[(None, None, None, preact_consumer_state.index)],
                            tSR_rP,
                        )
                        flat_preact = cute.group_modes(tSR_rP, 0, cute.rank(tSR_rP))
                        assert cute.size(flat_preact) == cute.size(flat_acc)
                        assert cute.size(flat_out) == cute.size(flat_acc)
                        assert cute.size(flat_aux) == cute.size(flat_acc)
                        self.epilog_sync_barrier.arrive_and_wait()
                        preact_pipeline.consumer_release(preact_consumer_state)
                        preact_consumer_state.advance()
                        preact_pair = cute.make_rmem_tensor(
                            cute.make_layout(1),
                            cutlass.Int32,
                        )
                        if cutlass.const_expr(
                            self.dactivation == "swiglu" and self.mma_inst_tile_k == 4
                        ):
                            preact_pairs = cute.make_rmem_tensor(
                                cute.make_layout(2),
                                cutlass.Int32,
                            )
                            grad_pairs = cute.make_rmem_tensor(
                                cute.make_layout(4),
                                cutlass.BFloat16,
                            )
                            for pair_idx in cutlass.range_constexpr(cute.size(flat_acc) // 2):
                                value_idx = pair_idx * 2
                                preact_pairs[0] = flat_preact[value_idx]
                                preact_pairs[1] = flat_preact[value_idx + 1]
                                bf16_pairs = cute.make_tensor(
                                    cute.recast_ptr(
                                        preact_pairs.iterator,
                                        dtype=cutlass.BFloat16,
                                    ),
                                    cute.make_layout(4),
                                )
                                dgate, dup, postact = swiglu_backward_pair(
                                    (
                                        bf16_pairs[0].to(cutlass.Float32),
                                        bf16_pairs[2].to(cutlass.Float32),
                                    ),
                                    (
                                        bf16_pairs[1].to(cutlass.Float32),
                                        bf16_pairs[3].to(cutlass.Float32),
                                    ),
                                    (
                                        flat_acc[value_idx] * alpha_value,
                                        flat_acc[value_idx + 1] * alpha_value,
                                    ),
                                )
                                grad_pairs[0] = dgate[0].to(cutlass.BFloat16)
                                grad_pairs[1] = dup[0].to(cutlass.BFloat16)
                                grad_pairs[2] = dgate[1].to(cutlass.BFloat16)
                                grad_pairs[3] = dup[1].to(cutlass.BFloat16)
                                packed_grads = cute.recast_tensor(grad_pairs, cutlass.Int32)
                                flat_out[value_idx] = packed_grads[0]
                                flat_out[value_idx + 1] = packed_grads[1]
                                flat_aux[value_idx] = postact[0].to(cutlass.BFloat16)
                                flat_aux[value_idx + 1] = postact[1].to(cutlass.BFloat16)
                        else:
                            grad_pair = cute.make_rmem_tensor(
                                cute.make_layout(2),
                                cutlass.BFloat16,
                            )
                            for value_idx in cutlass.range_constexpr(cute.size(flat_acc)):
                                preact_pair[0] = flat_preact[value_idx]
                                bf16_pair = cute.make_tensor(
                                    cute.recast_ptr(
                                        preact_pair.iterator,
                                        dtype=cutlass.BFloat16,
                                    ),
                                    cute.make_layout(2),
                                )
                                dgate, dup, postact = gated_backward_values(
                                    bf16_pair[0].to(cutlass.Float32),
                                    bf16_pair[1].to(cutlass.Float32),
                                    flat_acc[value_idx] * alpha_value,
                                    self.dactivation,
                                )
                                grad_pair[0] = dgate.to(cutlass.BFloat16)
                                grad_pair[1] = dup.to(cutlass.BFloat16)
                                flat_out[value_idx] = cute.recast_tensor(
                                    grad_pair,
                                    cutlass.Int32,
                                )[0]
                                flat_aux[value_idx] = postact.to(cutlass.BFloat16)
                        c_buffer = (num_prev_subtiles + subtile_idx) % c_store_stages
                        aux_buffer = (num_prev_subtiles + subtile_idx) % self.num_aux_stage
                        if warp_idx == self.epilog_warp_id[0]:
                            c_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rC,
                            tRS_sC[(None, None, None, c_buffer)],
                        )
                        cute.copy(
                            aux_tiled_copy_r2s,
                            tRS_rAux,
                            tRS_sAux[(None, None, None, aux_buffer)],
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        if warp_idx == self.epilog_warp_id[0]:
                            utils.block_copy(
                                tma_atom_c,
                                c_tma_smem[(None, c_buffer)],
                                c_tma_gmem[(None, subtile_idx)],
                            )
                            utils.block_copy(
                                tma_atom_aux,
                                aux_tma_smem[(None, aux_buffer)],
                                aux_tma_gmem[(None, subtile_idx)],
                            )
                            c_pipeline.producer_commit()
                    else:
                        if not is_k_tile_cnt_zero:
                            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                            tRS_rC.store((acc_vec * alpha[0]).to(self.c_dtype))
                        elif cutlass.const_expr(self.is_nvfp4_output):
                            zeros_i8 = cute.make_rmem_tensor(
                                cute.recast_layout(
                                    cutlass.Int8.width,
                                    self.c_dtype.width,
                                    tRS_rC.layout,
                                ),
                                cutlass.Int8,
                            )
                            zeros_i8.fill(0)
                            tRS_rC.store(cute.recast_tensor(zeros_i8, self.c_dtype).load())
                        else:
                            tRS_rC.fill(0)

                        c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                        if warp_idx == self.epilog_warp_id[0]:
                            c_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rC,
                            tRS_sC[(None, None, None, c_buffer)],
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        if warp_idx == self.epilog_warp_id[0]:
                            utils.block_copy(
                                tma_atom_c,
                                c_tma_smem[(None, c_buffer)],
                                c_tma_gmem[(None, subtile_idx)],
                            )
                            c_pipeline.producer_commit()
                if cutlass.const_expr(self.gated_sf_u32_words > 0):
                    sf_i32_staging = cute.recast_tensor(sf_i16_staging, cutlass.Int32)
                    if cutlass.const_expr(use_logical_sf_store):
                        tTR_gSF_filtered = cute.coalesce(cute.filter_zeros(tTR_gSF))
                        tTR_gSF_filtered = cute.group_modes(
                            tTR_gSF_filtered,
                            0,
                            cute.rank(tTR_gSF_filtered),
                        )
                        assert cute.size(tTR_gSF_filtered) == self.gated_sf_u32_words * 4
                        tTR_gSF_aligned = cute.make_tensor(
                            tTR_gSF_filtered.iterator.align(4),
                            tTR_gSF_filtered.layout,
                        )
                        if sf_cta_in_bounds:
                            cute.autovec_copy(
                                sf_i32_staging,
                                cute.recast_tensor(tTR_gSF_aligned, cutlass.Int32),
                            )
                    else:
                        for sf_word in cutlass.range_constexpr(self.gated_sf_u32_words):
                            first_subtile = sf_word * 2
                            coords = tTR_cAcc_subtiles[(None, None, None, first_subtile)]
                            output_pairs = cute.flat_divide(
                                coords,
                                cute.make_layout(2),
                            )
                            output_coords = output_pairs[
                                (0,) + (None,) * (cute.rank(output_pairs) - 1)
                            ]
                            coord = output_coords[0]
                            output_col = (
                                work_tile.tile_n_idx * (self.cta_tile_shape_mnk[1] // 2)
                                + coord[1] // 2
                            )
                            if sf_row_in_bounds and output_col + 48 < mC_mnl.shape[1]:
                                sf_offset = (
                                    sf_row_offset + (coord[1] // 128) * 512 + (coord[1] // 32) % 4
                                )
                                sf_ptr = cute.make_tensor(
                                    cute.recast_ptr(
                                        output_sf.iterator + sf_offset,
                                        dtype=cutlass.Int32,
                                    ).align(4),
                                    cute.make_layout(1),
                                )
                                sf_ptr[0] = sf_i32_staging[sf_word]
                #
                # Async arrive accumulator buffer empty
                #
                if not is_k_tile_cnt_zero and not cutlass.const_expr(self.interleave_scale_tmem):
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                if not is_k_tile_cnt_zero:
                    acc_consumer_state.advance()

                #
                # Advance to next tile
                #
                num_tiles_executed += 1
                work_tile = next_work_tile

            #
            # Dealloc the tensor memory buffer
            #
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=use_2cta_instrs)
            self.epilog_sync_barrier.arrive_and_wait()
            if warp_idx == self.epilog_warp_id[0]:
                if use_2cta_instrs:
                    cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1)
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                cute.arch.dealloc_tmem(
                    acc_tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=use_2cta_instrs
                )
            #
            # Wait for C store complete
            #
            c_pipeline.producer_tail()

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: cutlass.Boolean | bool,
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc: The accumulated tensor in register used to hold t2r results
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        c_acc = cute.make_identity_tensor((self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]))
        c_acc_epi = cute.flat_divide(c_acc, epi_tile)
        tTR_cAcc = thr_copy_t2r.partition_D(c_acc_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_cAcc[(None, None, None, 0, 0)].shape,
            self.acc_dtype,
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc, tTR_cAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The partitioned accumulator tensor
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor
        :type sepi: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rC, tRS_sC) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rC: The partitioned tensor C (register source)
            - tRS_sC: The partitioned tensor C (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param c_dtype: Data type of operand C (output).
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout enum of operand C.
        :type c_layout: utils.LayoutEnum
        :param sf_dtype: Data type of Scale factor.
        :type sf_dtype: type[cutlass.Numeric]
        :param sf_vec_size: Scale factor vector size.
        :type sf_vec_size: int
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        # ACC stages
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

        # Default C stages
        num_c_stage = 2

        # Calculate smem layout and size for one stage of A, B, SFA, SFB and C
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )

        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        # Calculate A/B/SFA/SFB stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B/SFA/SFB stage
        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B/SFA/SFB stages and reserved bytes
        # Add remaining unused smem to epilogue
        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)

        return num_acc_stage, num_ab_stage, num_c_stage

    @staticmethod
    def _compute_grid(
        total_num_clusters: int,
        cluster_shape_mn: tuple[int, int],
        max_active_clusters: cutlass.Constexpr[int],
    ) -> tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]:
        """Compute tile scheduler parameters and grid shape for grouped GEMM operations.

        :param total_num_clusters: Total number of clusters to process across all groups.
        :type total_num_clusters: int
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr[int]

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: tuple[utils.PersistentTileSchedulerParams, tuple[int, ...]]
        """
        # Create problem shape with M, N dimensions from cluster shape
        # and L dimension representing the total number of clusters.
        problem_shape_ntile_mnl = (
            cluster_shape_mn[0],
            cluster_shape_mn[1],
            cutlass.Int32(total_num_clusters),
        )

        tile_sched_params = utils.PersistentTileSchedulerParams(
            problem_shape_ntile_mnl, (*cluster_shape_mn, 1)
        )

        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def _get_mbar_smem_bytes(**kwargs_stages: int) -> int:
        """Calculate shared memory consumption for memory barriers based on provided stages.

        Each stage requires 2 barriers, and each barrier consumes 8 bytes of shared memory.
        The total consumption is the sum across all provided stages. This function calculates the total
        shared memory needed for these barriers.

        :param kwargs_stages: Variable keyword arguments where each key is a stage name
                              (e.g., num_acc_stage, num_ab_stage) and each value is the
                              number of stages of that type.
        :type kwargs_stages: int
        :return: Total shared memory bytes required for all memory barriers.
        :rtype: int
        """
        num_barriers_per_stage = 2
        num_bytes_per_barrier = 8
        mbar_smem_consumption = sum(
            [
                num_barriers_per_stage * num_bytes_per_barrier * stage
                for stage in kwargs_stages.values()
            ]
        )
        return mbar_smem_consumption

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        ab_dtype: type[cutlass.Numeric],
        sf_dtype: type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: type[cutlass.Numeric],
    ) -> bool:
        """
        Check if the dtypes and sf_vec_size are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes and sf_vec_size are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        # Check valid ab_dtype
        if ab_dtype not in {
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        # Check valid sf_vec_size
        if sf_vec_size not in {16, 32}:
            is_valid = False

        # Check valid sf_dtype
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False

        # Check valid sf_dtype and sf_vec_size combinations
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            is_valid = False
        if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
            is_valid = False

        # Check valid c_dtype
        if c_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_layouts(
        ab_dtype: type[cutlass.Numeric],
        c_dtype: type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if layouts and dtypes are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major dimension of the A tensor
        :type a_major: str
        :param b_major: The major dimension of the B tensor
        :type b_major: str
        :param c_major: The major dimension of the C tensor
        :type c_major: str

        :return: True if the layouts are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
    ) -> bool:
        """
        Check if the mma tiler and cluster shape are valid

        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]

        :return: True if the mma tiler and cluster shape are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        # Skip invalid mma tile shape
        if mma_tiler_mn[0] not in [128, 256]:
            is_valid = False
        if mma_tiler_mn[1] not in [128, 256]:
            is_valid = False
        # Skip illegal cluster shape
        if cluster_shape_mn[0] % (2 if mma_tiler_mn[0] == 256 else 1) != 0:
            is_valid = False
        # Skip invalid cluster shape
        is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            # Special cluster shape check for scale factor multicasts.
            # Due to limited size of scale factors, we can't multicast among more than 4 CTAs.
            or cluster_shape_mn[0] > 4
            or cluster_shape_mn[1] > 4
            or not is_power_of_2(cluster_shape_mn[0])
            or not is_power_of_2(cluster_shape_mn[1])
        ):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_tensor_alignment(
        problem_sizes_mnkl: list[tuple[int, int, int, int]],
        ab_dtype: type[cutlass.Numeric],
        c_dtype: type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param problem_sizes_mnkl: The problem shape for each group
        :type problem_sizes_mnkl: List[Tuple[int, int, int, int]]
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        for m, n, k, l in problem_sizes_mnkl:
            if (
                not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
                or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
                or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
            ):
                is_valid = False
        return is_valid

    @staticmethod
    def can_implement(
        ab_dtype: type[cutlass.Numeric],
        sf_dtype: type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: type[cutlass.Numeric],
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        problem_sizes_mnkl: list[tuple[int, int, int, int]],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the gemm can be implemented

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor tensor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]

        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the gemm can be implemented, False otherwise
        :rtype: bool
        """
        can_implement = True
        # Skip unsupported types
        if not Sm100GroupedBlockScaledGemmKernel.is_valid_dtypes_and_scale_factor_vec_size(
            ab_dtype, sf_dtype, sf_vec_size, c_dtype
        ):
            can_implement = False
        # Skip unsupported layouts
        if not Sm100GroupedBlockScaledGemmKernel.is_valid_layouts(
            ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        # Skip invalid mma tile shape and cluster shape
        if not Sm100GroupedBlockScaledGemmKernel.is_valid_mma_tiler_and_cluster_shape(
            mma_tiler_mn, cluster_shape_mn
        ):
            can_implement = False
        # Skip illegal problem shape for load/store alignment
        if not Sm100GroupedBlockScaledGemmKernel.is_valid_tensor_alignment(
            problem_sizes_mnkl, ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        return can_implement

    # Size of smem reserved for barriers and tensor-memory management.
    reserved_smem_bytes = 1024
    # size of smem used for tensor memory management
    tensor_memory_management_bytes = 12


GroupedGemmKernel = Sm100GroupedBlockScaledGemmKernel

__all__ = ["GroupedGemmKernel"]
