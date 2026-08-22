"""Host runtime for dense NVFP4 GEMM."""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import cute, utils
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cute.runtime import make_ptr

from .._common import torch2cute_dtype_map
from .kernel import Sm100BlockScaledPersistentDenseGemmKernel


@cache
def _compile_dense(
    n: int,
    k: int,
    tile_m: int,
    tile_n: int,
    device: int,
    output_dtype: torch.dtype,
):
    with torch.cuda.device(device):
        kernel = Sm100BlockScaledPersistentDenseGemmKernel(
            16,
            (tile_m, tile_n),
            (2, 1) if tile_m == 256 else (1, 1),
        )
        hardware = utils.HardwareInfo()
        max_active = hardware.get_max_active_clusters(2 if tile_m == 256 else 1)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        address_space = cute.AddressSpace.gmem
        compiled = cute.compile(
            kernel,
            make_ptr(cutlass.Float4E2M1FN, 0, address_space, assumed_align=16),
            make_ptr(cutlass.Float4E2M1FN, 0, address_space, assumed_align=16),
            make_ptr(cutlass.Float8E4M3FN, 0, address_space, assumed_align=32),
            make_ptr(cutlass.Float8E4M3FN, 0, address_space, assumed_align=32),
            make_ptr(torch2cute_dtype_map[output_dtype], 0, address_space, assumed_align=16),
            make_ptr(cutlass.Float32, 0, address_space, assumed_align=4),
            (OperandMajorMode.K, OperandMajorMode.K, utils.LayoutEnum.ROW_MAJOR),
            (cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0)),
            max_active,
            stream,
            lambda x: x,
            options="--enable-tvm-ffi --opt-level 2",
        )
        stages = (kernel.num_acc_stage, kernel.num_ab_stage, kernel.num_c_stage)
        return compiled, max_active, stages


class DenseNvfp4Gemm:
    """Dense ``C = A @ B.T`` GEMM for prepacked NVFP4 operands."""

    def __init__(
        self,
        n: int,
        k: int,
        tile_m: int,
        tile_n: int,
        output_dtype: torch.dtype = torch.bfloat16,
    ):
        if n <= 0 or k <= 0 or k % 64:
            raise ValueError("N and K must be positive and K must be aligned to 64")
        if tile_m not in (128, 256) or tile_n not in (64, 128, 192, 256):
            raise ValueError("dense GEMM supports M tiles 128/256 and N tiles 64/128/192/256")
        if output_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("dense GEMM output must use BF16 or FP32")
        self.n = n
        self.k = k
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.output_dtype = output_dtype
        self.device = torch.cuda.current_device()
        self._compiled = None
        self._stream_handle = None
        self._stream = None

    def prepare(self, a, b, out, sfa, sfb, alpha):
        tensors = {"A": a, "B": b, "output": out, "SFA": sfa, "SFB": sfb, "alpha": alpha}
        for name, tensor in tensors.items():
            if not tensor.is_cuda or tensor.device.index != self.device:
                raise ValueError(f"{name} must be on CUDA device {self.device}")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        rows = a.shape[0]
        if a.dtype != torch.float4_e2m1fn_x2 or tuple(a.shape) != (rows, self.k // 2):
            raise ValueError(f"A must have packed shape (M, {self.k // 2})")
        if b.dtype != torch.float4_e2m1fn_x2 or tuple(b.shape) != (self.n, self.k // 2):
            raise ValueError(f"B must have packed shape ({self.n}, {self.k // 2})")
        if out.dtype != self.output_dtype or tuple(out.shape) != (rows, self.n):
            raise ValueError(f"output must have shape ({rows}, {self.n})")
        expected_sfa = (-(-rows // 128), self.k // 64, 32, 4, 4)
        expected_sfb = (-(-self.n // 128), self.k // 64, 32, 4, 4)
        if sfa.dtype != torch.float8_e4m3fn or tuple(sfa.shape) != expected_sfa:
            raise ValueError(f"SFA must have shape {expected_sfa}")
        if sfb.dtype != torch.float8_e4m3fn or tuple(sfb.shape) != expected_sfb:
            raise ValueError(f"SFB must have shape {expected_sfb}")
        if alpha.dtype != torch.float32 or tuple(alpha.shape) != (1,):
            raise ValueError("alpha must be float32 with shape (1,)")
        self._inputs = (a, b, out, sfa, sfb, alpha)

    def launch(self):
        if not hasattr(self, "_inputs"):
            raise RuntimeError("prepare must run before launch")
        if self._compiled is None:
            self._compiled, self._max_active, self.stages = _compile_dense(
                self.n,
                self.k,
                self.tile_m,
                self.tile_n,
                self.device,
                self.output_dtype,
            )
        stream_handle = torch.cuda.current_stream().cuda_stream
        if stream_handle != self._stream_handle:
            self._stream_handle = stream_handle
            self._stream = cuda.CUstream(stream_handle)
        a, b, out, sfa, sfb, alpha = self._inputs
        address_space = cute.AddressSpace.gmem
        self._compiled(
            make_ptr(cutlass.Float4E2M1FN, a.data_ptr(), address_space, assumed_align=16),
            make_ptr(cutlass.Float4E2M1FN, b.data_ptr(), address_space, assumed_align=16),
            make_ptr(cutlass.Float8E4M3FN, sfa.data_ptr(), address_space, assumed_align=32),
            make_ptr(cutlass.Float8E4M3FN, sfb.data_ptr(), address_space, assumed_align=32),
            make_ptr(
                torch2cute_dtype_map[self.output_dtype],
                out.data_ptr(),
                address_space,
                assumed_align=16,
            ),
            make_ptr(cutlass.Float32, alpha.data_ptr(), address_space, assumed_align=4),
            (a.shape[0], self.n, self.k, 1),
            self._stream,
        )

    def __call__(self, a, b, out, sfa, sfb, alpha):
        self.prepare(a, b, out, sfa, sfb, alpha)
        self.launch()

    run = __call__


__all__ = ["DenseNvfp4Gemm"]
