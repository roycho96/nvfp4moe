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

"""SM100 pipelines used by the native kernels."""

from dataclasses import dataclass

from cutlass import cute, pipeline


@dataclass(frozen=True)
class CpAsyncUmmaPipeline(pipeline.PipelineAsync):
    """LDGSTS producer and UMMA consumer pipeline for one CTA."""

    @staticmethod
    def create(
        *,
        num_stages: int,
        producer_threads: int,
        barrier_storage: cute.Pointer,
    ):
        producer = (
            pipeline.PipelineOp.AsyncLoad,
            pipeline.CooperativeGroup(pipeline.Agent.Thread, producer_threads),
        )
        consumer = (
            pipeline.PipelineOp.TCGen05Mma,
            pipeline.CooperativeGroup(pipeline.Agent.Thread),
        )
        make_sync = pipeline.PipelineTmaUmma._make_sync_object
        full = make_sync(barrier_storage.align(min_align=8), num_stages, producer)
        empty = make_sync(
            barrier_storage.align(min_align=8) + num_stages,
            num_stages,
            consumer,
        )
        return CpAsyncUmmaPipeline(full, empty, num_stages, None, None)

    def consumer_release(self, state, *, loc=None, ip=None):
        self.sync_object_empty.arrive(
            state.index,
            None,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            loc=loc,
            ip=ip,
        )
