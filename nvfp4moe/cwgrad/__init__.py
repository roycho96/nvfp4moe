"""Grouped block-scaled wgrad wrapper around the vendored NVIDIA kernel."""

import torch

from .. import _vendor  # noqa: F401  (EXTRA_SOURCE_DIRS registration)
from quack.cache import jit_cache  # noqa: E402

from .moe_utils import MoEWeightMode, WGradInputOrder  # noqa: E402
from .moe_blockscaled_grouped_gemm_wgrad import (  # noqa: E402
    BlockScaledMoEGroupedGemmWgradKernel,
)


def _kernel_and_ws(m, n, E, mma_m, mma_n, accumulate=False):
    from .moe_utils import WgradSfTensormapConstructor

    kernel = BlockScaledMoEGroupedGemmWgradKernel(
        sf_vec_size=16,
        use_2cta_instrs=(mma_m == 256),
        mma_tiler_mn=(mma_m, mma_n),
        cluster_shape_mn=((2, 1) if mma_m == 256 else (1, 1)),
        accumulate_on_output=accumulate,
        expert_cnt=E,
        weight_mode=MoEWeightMode.DENSE,
        input_order=WGradInputOrder.Tensor2D,
    )
    ws_bytes = max(
        WgradSfTensormapConstructor.get_workspace_size(
            WGradInputOrder.Tensor2D, MoEWeightMode.DENSE, E
        ),
        1,
    )
    return kernel, ws_bytes


@jit_cache
def _compile(m, n, E, mma_m, mma_n, accumulate=False):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack, make_fake_stream

    assert m % 128 == 0 and n % 128 == 0
    kernel, ws_bytes = _kernel_and_ws(m, n, E, mma_m, mma_n, accumulate)

    K = 256  # sample only; the token axis is compiled dynamic
    dev = "cuda"
    a_s = torch.zeros(m, K // 2, dtype=torch.uint8, device=dev).view(
        torch.float4_e2m1fn_x2)
    b_s = torch.zeros(n, K // 2, dtype=torch.uint8, device=dev).view(
        torch.float4_e2m1fn_x2).t()
    sfa_s = torch.zeros(m, K // 16, dtype=torch.float8_e4m3fn, device=dev)
    sfb_s = torch.zeros(n, K // 16, dtype=torch.float8_e4m3fn, device=dev)
    w_s = torch.zeros(E, m, n, dtype=torch.bfloat16, device=dev)
    off_s = torch.zeros(E, dtype=torch.int32, device=dev)
    gs_s = torch.ones(E, dtype=torch.float32, device=dev)
    ws_s = torch.empty(ws_bytes, dtype=torch.uint8, device=dev)

    a_f = from_dlpack(a_s, assumed_align=16, enable_tvm_ffi=True
                      ).mark_compact_shape_dynamic(
        mode=1, stride_order=a_s.dim_order(), divisibility=16)
    b_f = from_dlpack(b_s, assumed_align=16, enable_tvm_ffi=True
                      ).mark_compact_shape_dynamic(
        mode=0, stride_order=b_s.dim_order(), divisibility=16)
    sfa_f = from_dlpack(sfa_s, assumed_align=16, enable_tvm_ffi=True
                        ).mark_compact_shape_dynamic(
        mode=1, stride_order=sfa_s.dim_order(), divisibility=4)
    sfb_f = from_dlpack(sfb_s, assumed_align=16, enable_tvm_ffi=True
                        ).mark_compact_shape_dynamic(
        mode=1, stride_order=sfb_s.dim_order(), divisibility=4)
    w_f = from_dlpack(w_s, assumed_align=16, enable_tvm_ffi=True)
    off_f = from_dlpack(off_s, assumed_align=4, enable_tvm_ffi=True)
    ws_f = from_dlpack(ws_s, assumed_align=128, enable_tvm_ffi=True)
    gsa_f = from_dlpack(gs_s, assumed_align=4, enable_tvm_ffi=True)
    gsb_f = from_dlpack(gs_s.clone(), assumed_align=4, enable_tvm_ffi=True)

    hw = cutlass.utils.HardwareInfo()
    max_active = hw.get_max_active_clusters(
        kernel.cluster_shape_mn[0] * kernel.cluster_shape_mn[1])
    return cute.compile(
        kernel, a_f, b_f, sfa_f, sfb_f, w_f, off_f, ws_f, max_active,
        make_fake_stream(use_tvm_ffi_env_stream=True), gsa_f, gsb_f, None,
        options="--enable-tvm-ffi",
    )


class GroupedWgrad:
    """Compiled grouped-wgrad operator for a fixed output geometry.

    Inputs use K-major NVFP4 with per-expert blocked scales and padded device
    offsets. Accumulation uses one owner CTA per output tile without split-K.
    """

    def __init__(self, m, n, E, mma_tiler=(128, 128), accumulate=False):
        self.m, self.n, self.E = m, n, E
        self.mma = mma_tiler
        self.accumulate = accumulate
        _, ws_bytes = _kernel_and_ws(m, n, E, *mma_tiler, accumulate)
        self._ws = torch.empty(ws_bytes, dtype=torch.uint8, device="cuda")
        self._fn = None

    def __call__(self, a, b_t, sfa, sfb, off_pad, out, gs_a, gs_b):
        if self._fn is None:
            self._fn = _compile(self.m, self.n, self.E, *self.mma,
                                self.accumulate)
        K2 = a.shape[1]
        a_v = a.view(torch.uint8).view(torch.float4_e2m1fn_x2)
        b_v = b_t.view(torch.uint8).view(torch.float4_e2m1fn_x2).t()
        sfa_v = sfa.view(torch.float8_e4m3fn).view(self.m, K2 // 8)
        sfb_v = sfb.view(torch.float8_e4m3fn).view(self.n, K2 // 8)
        self._fn(a_v, b_v, sfa_v, sfb_v, out, off_pad, self._ws, gs_a, gs_b,
                 None)


__all__ = ["GroupedWgrad"]
