"""Host runtime for the native grouped NVFP4 GEMM kernel."""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.torch as cutlass_torch
import torch
from cutlass import Float32, Int32, cute
from cutlass.cute.runtime import from_dlpack

from ._common import fake_tensor, torch2cute_dtype_map
from .epilogue import GatedBackwardEpilogue, GatedEpilogue, resolve_gemm_epilogue
from .gemm_kernel import (
    DECODE_GRID_DOUBLE,
    DECODE_GRID_FULL,
    DECODE_GRID_PERSISTENT,
    GroupedGemmKernel,
)


def _resolve_dynamic_schedule(use_dynamic_sched: bool | None, k: int) -> bool:
    if use_dynamic_sched is None:
        return k > 2048
    return bool(use_dynamic_sched)


def _seed_tensor(dtype, mode0: int, mode1: int, mode0_major: bool = False):
    ref = cutlass_torch.matrix(1, mode0, mode1, mode0_major, Float32)
    _, backing = cutlass_torch.cute_tensor_like(ref, dtype, True, assumed_align=16)
    tensor = from_dlpack(backing, assumed_align=16, enable_tvm_ffi=True)
    tensor.element_type = dtype
    tensor = tensor.mark_layout_dynamic(leading_dim=cutlass_torch.get_leading_dim(backing))
    return tensor, backing


@cache
def _compile_grouped(
    experts: int,
    n: int,
    k: int,
    tile_m: int,
    tile_n: int,
    device: int,
    output_dtype: torch.dtype,
    activation: str | None,
    dactivation: str | None,
    save_preact: bool,
    use_dynamic_sched: bool,
    per_expert_alpha: bool,
    use_pdl: bool,
    swap_ab: bool,
    fast_decode_sched: bool,
    direct_routes: int,
    broadcast_a: bool,
    gather_b: bool,
    ab_stages: int,
    direct_c_store: bool,
    decode_grid_mode: int,
):
    with torch.cuda.device(device):
        c_dtype = torch2cute_dtype_map[output_dtype]
        output_n = n
        if activation is not None:
            output_n = n // 2
        seed_a = _seed_tensor(cutlass.Float4E2M1FN, 32, 32)
        seed_b = _seed_tensor(cutlass.Float4E2M1FN, 32, 32)
        seed_c = _seed_tensor(c_dtype, 16, 16, swap_ab)
        seed_sfa = _seed_tensor(cutlass.Float8E4M3FN, 16, 16)
        seed_sfb = _seed_tensor(cutlass.Float8E4M3FN, 16, 16)
        seeds = (seed_a, seed_b, seed_c, seed_sfa, seed_sfb)

        rows, c_rows, preact_rows, aux_rows, sfa_rows, plan_rows, gather_rows = (
            cute.sym_int(),
            cute.sym_int(),
            cute.sym_int(),
            cute.sym_int(),
            cute.sym_int(),
            cute.sym_int(),
            cute.sym_int(),
        )
        runtime_inputs = (
            fake_tensor(cutlass.Float4E2M1FN, (rows, k), 32),
            fake_tensor(cutlass.Float4E2M1FN, (experts, n, k), 32),
            fake_tensor(c_dtype, (c_rows, output_n), 128 // c_dtype.width),
            fake_tensor(cutlass.BFloat16, (preact_rows, n * 2), 8),
            fake_tensor(cutlass.BFloat16, (aux_rows, n), 8),
            fake_tensor(
                cutlass.Float8E4M3FN,
                (1, sfa_rows, k // 64, 32, 4, 4),
            ),
            fake_tensor(
                cutlass.Float8E4M3FN,
                (experts, -(-n // 128), k // 64, 32, 4, 4),
            ),
            fake_tensor(Int32, (direct_routes if direct_routes else experts + 1,), 1),
            fake_tensor(Int32, (1,), 1),
            fake_tensor(Int32, (plan_rows,), 1),
            fake_tensor(Int32, (gather_rows,), 1),
        )
        alpha_fake = fake_tensor(Float32, (experts if per_expert_alpha else 1,), 1)
        output_sf_fake = fake_tensor(
            cutlass.Float8E4M3FN,
            (
                1,
                sfa_rows,
                output_n // 64 if output_dtype == torch.float4_e2m1fn_x2 else k // 64,
                32,
                4,
                4,
            ),
        )
        output_scale_fake = fake_tensor(Float32, (2,), 1)

        cluster = (2, 1) if tile_m == 256 else (1, 1)
        kernel = GroupedGemmKernel(
            16,
            (tile_m, tile_n),
            cluster,
            activation=activation,
            dactivation=dactivation,
            save_preact=save_preact,
            mma_inst_tile_k=2 if dactivation is not None and k >= 4096 else 4,
            use_dynamic_sched=use_dynamic_sched,
            use_pdl=use_pdl,
            swap_ab=swap_ab,
            fast_decode_sched=fast_decode_sched,
            direct_routes=direct_routes,
            broadcast_a=broadcast_a,
            gather_b=gather_b,
            ab_stages=ab_stages,
            direct_c_store=direct_c_store,
            decode_grid_mode=decode_grid_mode,
        )
        hardware = cutlass.utils.HardwareInfo()
        max_active = hardware.get_max_active_clusters(cluster[0] * cluster[1])
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = cute.compile(
            kernel,
            *(seed[0] for seed in seeds),
            *runtime_inputs,
            experts,
            n,
            k,
            alpha_fake,
            output_sf_fake,
            output_scale_fake,
            max_active,
            max_active,
            stream,
            options="--enable-tvm-ffi --opt-level 2",
        )
        keepalive = tuple(seed[1] for seed in seeds)
        stages = (kernel.num_acc_stage, kernel.num_ab_stage, kernel.num_c_stage)
        return compiled, tuple(seed[0] for seed in seeds), max_active, keepalive, stages


class GroupedNvfp4Gemm:
    """Grouped NVFP4 GEMM for contiguous expert-major routed rows."""

    def __init__(
        self,
        experts: int,
        n: int,
        k: int,
        tile_m: int,
        tile_n: int,
        output_dtype: torch.dtype = torch.bfloat16,
        activation: str | None = None,
        dactivation: str | None = None,
        epilogue: GatedEpilogue | GatedBackwardEpilogue | None = None,
        use_dynamic_sched: bool | None = None,
        use_pdl: bool = False,
        swap_ab: bool = False,
        fast_decode_sched: bool = False,
        direct_routes: int = 0,
        broadcast_a: bool = False,
        gather_b: bool = False,
    ):
        activation, dactivation = resolve_gemm_epilogue(epilogue, activation, dactivation)
        save_preact = isinstance(epilogue, GatedEpilogue) and epilogue.save_preact
        if epilogue is None:
            if activation is not None:
                epilogue = GatedEpilogue(activation)
            elif dactivation is not None:
                epilogue = GatedBackwardEpilogue(dactivation)
        if experts <= 0 or n <= 0 or k <= 0:
            raise ValueError("experts, N, and K must be positive")
        if experts > 256:
            raise ValueError("native grouped GEMM supports at most 256 local experts")
        if k % 64:
            raise ValueError("native grouped GEMM requires K aligned to 64")
        if tile_m not in (128, 256) or tile_n < 8 or tile_n > 256 or tile_n % 8:
            raise ValueError("native grouped GEMM supports M=128/256 and N=8..256 aligned to 8")
        if output_dtype not in (
            torch.bfloat16,
            torch.float32,
            torch.int32,
            torch.float4_e2m1fn_x2,
        ):
            raise ValueError("native grouped GEMM output must be BF16, FP32, Int32, or NVFP4")
        if activation not in (None, "swiglu", "geglu", "reglu"):
            raise ValueError("activation must be swiglu, geglu, reglu, or None")
        if dactivation not in (None, "swiglu", "geglu", "reglu"):
            raise ValueError("dactivation must be swiglu, geglu, reglu, or None")
        if activation is not None and dactivation is not None:
            raise ValueError("activation and dactivation are mutually exclusive")
        if output_dtype == torch.float4_e2m1fn_x2 and activation is None:
            raise ValueError("native NVFP4 output requires a gated activation")
        if output_dtype == torch.float4_e2m1fn_x2 and n % 128:
            raise ValueError("native gated NVFP4 output requires N aligned to 128")
        gated_tile = tile_m if swap_ab else tile_n
        if activation is not None and (n % 2 or gated_tile % 128):
            axis = "tile_M" if swap_ab else "tile_N"
            raise ValueError(f"native gated GEMM requires even N and {axis} aligned to 128")
        if dactivation is not None and output_dtype != torch.int32:
            raise ValueError("native dgrad2 uses packed BF16 pairs in an Int32 view")
        if dactivation is not None and n % 128:
            raise ValueError("native dgrad2 requires N aligned to 128")
        if save_preact and output_dtype != torch.float4_e2m1fn_x2:
            raise ValueError("saved preactivation requires a gated NVFP4 output")
        if swap_ab and dactivation is not None:
            raise ValueError("operand-swapped grouped GEMM does not support gated backward")
        if swap_ab and activation is not None and tile_n > 64:
            raise ValueError("operand-swapped gated GEMM requires tile_N at most 64")
        self.experts = experts
        self.n = n
        self.k = k
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.output_dtype = output_dtype
        self.activation = activation
        self.dactivation = dactivation
        self.save_preact = save_preact
        self.epilogue = epilogue
        self.use_dynamic_sched = _resolve_dynamic_schedule(use_dynamic_sched, k)
        self.use_pdl = bool(use_pdl)
        self.swap_ab = bool(swap_ab)
        self.fast_decode_sched = bool(fast_decode_sched)
        self.direct_routes = int(direct_routes)
        self.broadcast_a = bool(broadcast_a)
        self.gather_b = bool(gather_b)
        self._direct_c_store = False
        if self.fast_decode_sched and self.use_dynamic_sched:
            raise ValueError("fast decode scheduling requires a static GEMM")
        if self.direct_routes < 0 or self.direct_routes > 32:
            raise ValueError("direct route count must be between 0 and 32")
        if self.direct_routes and (
            self.use_dynamic_sched or self.swap_ab or self.fast_decode_sched
        ):
            raise ValueError("direct routing requires static, non-swapped GEMM")
        if self.direct_routes and self.tile_n not in (128, 256):
            raise ValueError("direct routing requires tile_N 128 or 256")
        if self.broadcast_a and not self.direct_routes:
            raise ValueError("broadcast A requires direct routing")
        if self.gather_b and not (self.swap_ab and self.fast_decode_sched):
            raise ValueError("gather B requires swapped fast decode")
        self.output_n = n
        if activation is not None:
            self.output_n = n // 2
        self.device = torch.cuda.current_device()
        self._compiled = None
        self._stream_handle = None
        self._stream = None
        self._unused_output_scale = torch.ones(2, dtype=torch.float32, device="cuda")
        self._unused_preact = torch.empty(1, 2 * n, dtype=torch.bfloat16, device="cuda")
        self._unused_aux = torch.empty(1, n, dtype=torch.bfloat16, device="cuda")
        self._sched_counter = torch.zeros(1, dtype=torch.int32, device="cuda")

    def prepare(
        self,
        a,
        b,
        out,
        cu,
        sfa,
        sfb,
        alpha,
        output_sf=None,
        output_scale=None,
        preact=None,
        aux=None,
        decode_plan=None,
        gather_idx=None,
    ):
        tensors = {
            "A": a,
            "B": b,
            "output": out,
            "cu": cu,
            "SFA": sfa,
            "SFB": sfb,
            "alpha": alpha,
        }
        for name, tensor in tensors.items():
            if not tensor.is_cuda or tensor.device.index != self.device:
                raise ValueError(f"{name} must be on CUDA device {self.device}")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

        if a.dtype != torch.float4_e2m1fn_x2 or a.ndim != 2 or a.shape[1] * 2 != self.k:
            raise ValueError(f"A must have packed shape (M, {self.k // 2})")
        if b.dtype != torch.float4_e2m1fn_x2 or tuple(b.shape) != (
            self.experts,
            self.n,
            self.k // 2,
        ):
            raise ValueError("B does not match the compiled expert geometry")
        routed_rows = a.shape[0]
        if self.gather_b:
            if (
                gather_idx is None
                or not gather_idx.is_cuda
                or gather_idx.device.index != self.device
                or not gather_idx.is_contiguous()
                or gather_idx.dtype != torch.int32
                or gather_idx.ndim != 1
            ):
                raise ValueError("gather indices must be contiguous CUDA int32")
            routed_rows = gather_idx.numel()
        elif gather_idx is None:
            gather_idx = cu

        output_cols = (
            self.output_n // 2 if self.output_dtype == torch.float4_e2m1fn_x2 else self.output_n
        )
        if tuple(out.shape) != (routed_rows, output_cols) or out.dtype != self.output_dtype:
            raise ValueError(
                f"output must use {self.output_dtype} with shape ({routed_rows}, {output_cols})"
            )
        expected_cu = (self.direct_routes,) if self.direct_routes else (self.experts + 1,)
        if cu.dtype != torch.int32 or tuple(cu.shape) != expected_cu:
            raise ValueError(f"routing metadata must be contiguous int32 with shape {expected_cu}")
        if self.direct_routes and a.shape[0] != self.direct_routes:
            raise ValueError("direct routing requires one output row per route")
        if self.fast_decode_sched:
            if (
                decode_plan is None
                or not decode_plan.is_cuda
                or decode_plan.device.index != self.device
                or not decode_plan.is_contiguous()
                or decode_plan.dtype != torch.int32
                or decode_plan.numel() < self.experts + 1
            ):
                raise ValueError(
                    f"fast decode plan must be contiguous CUDA int32 with at least "
                    f"{self.experts + 1} elements"
                )
        elif decode_plan is None:
            decode_plan = cu

        sf_tile_rows = self.tile_n if self.fast_decode_sched else 128
        expected_sfa = (
            1,
            -(-routed_rows // sf_tile_rows) + self.experts,
            self.k // 64,
            32,
            4,
            4,
        )
        expected_sfb = (
            self.experts,
            -(-self.n // 128),
            self.k // 64,
            32,
            4,
            4,
        )
        if sfa.dtype != torch.float8_e4m3fn or tuple(sfa.shape) != expected_sfa:
            raise ValueError(f"SFA must have shape {expected_sfa}")
        if sfb.dtype != torch.float8_e4m3fn or tuple(sfb.shape) != expected_sfb:
            raise ValueError(f"SFB must have shape {expected_sfb}")
        if alpha.dtype != torch.float32 or tuple(alpha.shape) not in (
            (1,),
            (self.experts,),
        ):
            raise ValueError(
                f"alpha must be contiguous float32 with shape (1,) or ({self.experts},)"
            )

        if self.dactivation is not None:
            for name, tensor, shape in (
                ("preact", preact, (routed_rows, self.n * 2)),
                ("aux", aux, (routed_rows, self.n)),
            ):
                if (
                    tensor is None
                    or not tensor.is_cuda
                    or tensor.device.index != self.device
                    or not tensor.is_contiguous()
                    or tensor.dtype != torch.bfloat16
                    or tuple(tensor.shape) != shape
                ):
                    raise ValueError(f"{name} must be contiguous BF16 with shape {shape}")
        elif self.save_preact:
            preact = self._unused_preact
            expected_aux = (routed_rows, self.n)
            if (
                aux is None
                or not aux.is_cuda
                or aux.device.index != self.device
                or not aux.is_contiguous()
                or aux.dtype != torch.bfloat16
                or tuple(aux.shape) != expected_aux
            ):
                raise ValueError(f"aux must be contiguous BF16 with shape {expected_aux}")
        else:
            preact = self._unused_preact
            aux = self._unused_aux

        if self.output_dtype == torch.float4_e2m1fn_x2:
            output_sf_rows = -(-routed_rows // sf_tile_rows) + self.experts
            expected_output_sf = (
                1,
                output_sf_rows,
                self.output_n // 64,
                32,
                4,
                4,
            )
            if (
                output_sf is None
                or not output_sf.is_cuda
                or output_sf.dtype != torch.float8_e4m3fn
                or tuple(output_sf.shape) != expected_output_sf
                or not output_sf.is_contiguous()
                or output_sf.device.index != self.device
            ):
                raise ValueError(f"output_sf must have shape {expected_output_sf}")
            if (
                output_scale is None
                or not output_scale.is_cuda
                or output_scale.dtype != torch.float32
                or tuple(output_scale.shape) != (2,)
                or not output_scale.is_contiguous()
                or output_scale.device.index != self.device
            ):
                raise ValueError("output_scale must be contiguous float32 [pts, inv_pts]")
        else:
            output_sf = sfa
            output_scale = self._unused_output_scale

        self._runtime_inputs = (
            a,
            b,
            out,
            preact,
            aux,
            sfa,
            sfb,
            cu,
            self._sched_counter,
            decode_plan,
            gather_idx,
        )
        per_expert_alpha = alpha.shape[0] == self.experts
        if getattr(self, "_per_expert_alpha", per_expert_alpha) != per_expert_alpha:
            self._compiled = None
        ab_stages = 0
        if self.gather_b and a.shape[0] > 1:
            ab_stages = 8 if self.k >= 4096 else 5
        elif (
            self.fast_decode_sched
            and self.swap_ab
            and self.n >= 4096
            and self.k >= 2048
            and a.shape[0] > 8
        ):
            ab_stages = 6
        if getattr(self, "_ab_stages", ab_stages) != ab_stages:
            self._compiled = None
        decode_grid_mode = DECODE_GRID_PERSISTENT
        if self.gather_b and self.k >= 4096 and self.n >= 4096:
            if a.shape[0] == 1:
                decode_grid_mode = DECODE_GRID_FULL
            elif a.shape[0] <= (64 if self.experts >= 48 else 16):
                decode_grid_mode = DECODE_GRID_DOUBLE
        elif self.gather_b and self.k >= 3072 and self.n == 3072 and 1 < a.shape[0] <= 32:
            decode_grid_mode = DECODE_GRID_DOUBLE
        if getattr(self, "_decode_grid_mode", decode_grid_mode) != decode_grid_mode:
            self._compiled = None
        self._per_expert_alpha = per_expert_alpha
        self._ab_stages = ab_stages
        self._decode_grid_mode = decode_grid_mode
        self._alpha = alpha
        self._output_sf = output_sf
        self._output_scale = output_scale

    def launch(self):
        if not hasattr(self, "_alpha"):
            raise RuntimeError("prepare must run before launch")

        if self._compiled is None:
            self._compiled, self._seeds, _, self._keepalive, self.stages = _compile_grouped(
                self.experts,
                self.n,
                self.k,
                self.tile_m,
                self.tile_n,
                self.device,
                self.output_dtype,
                self.activation,
                self.dactivation,
                self.save_preact,
                self.use_dynamic_sched,
                self._per_expert_alpha,
                self.use_pdl,
                self.swap_ab,
                self.fast_decode_sched,
                self.direct_routes,
                self.broadcast_a,
                self.gather_b,
                self._ab_stages,
                self._direct_c_store,
                self._decode_grid_mode,
            )
        stream_handle = torch.cuda.current_stream().cuda_stream
        if stream_handle != self._stream_handle:
            self._stream_handle = stream_handle
            self._stream = cuda.CUstream(stream_handle)
        if self.use_dynamic_sched:
            self._sched_counter.zero_()
        self._compiled(
            *self._seeds,
            *self._runtime_inputs,
            self._alpha,
            self._output_sf,
            self._output_scale,
            self._stream,
        )

    def __call__(
        self,
        a,
        b,
        out,
        cu,
        sfa,
        sfb,
        alpha,
        output_sf=None,
        output_scale=None,
        preact=None,
        aux=None,
        decode_plan=None,
        gather_idx=None,
    ):
        self.prepare(
            a,
            b,
            out,
            cu,
            sfa,
            sfb,
            alpha,
            output_sf,
            output_scale,
            preact,
            aux,
            decode_plan,
            gather_idx,
        )
        self.launch()

    run = __call__


def grouped_nvfp4_gemm(
    experts: int,
    n: int,
    k: int,
    tile_m: int,
    tile_n: int,
    output_dtype: torch.dtype = torch.bfloat16,
    activation: str | None = None,
    dactivation: str | None = None,
    epilogue: GatedEpilogue | GatedBackwardEpilogue | None = None,
    use_dynamic_sched: bool | None = None,
    use_pdl: bool = False,
    swap_ab: bool = False,
    fast_decode_sched: bool = False,
    direct_routes: int = 0,
    broadcast_a: bool = False,
):
    return GroupedNvfp4Gemm(
        experts,
        n,
        k,
        tile_m,
        tile_n,
        output_dtype,
        activation,
        dactivation,
        epilogue,
        use_dynamic_sched,
        use_pdl,
        swap_ab,
        fast_decode_sched,
        direct_routes,
        broadcast_a,
    )


__all__ = ["GroupedNvfp4Gemm", "grouped_nvfp4_gemm"]
