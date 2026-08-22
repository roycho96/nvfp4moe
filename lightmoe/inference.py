"""Allocation-free NVFP4 MoE inference plan for SM100."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ._quantization import _DEN, quantize_expert_stack
from .kernels.gated import GatedEpilogue
from .kernels.grouped.runtime import GroupedNvfp4Gemm
from .kernels.quantize.decode import (
    nvfp4_dispatch_quantize_decode,
    nvfp4_quantize_decode,
)
from .kernels.quantize.runtime import nvfp4_quantize_rowwise
from .kernels.routing.combine import moe_finalize
from .kernels.routing.dispatch import B_MAX, CHUNK, moe_dispatch

DECODE_ROWS_PER_EXPERT = 16
FAST_DECODE_ROWS = 64


def _use_swapped_fc2(tokens: int) -> bool:
    return tokens <= 8


def _decode_tile_rows(tokens: int, hidden: int, intermediate: int, experts: int) -> int:
    if hidden >= 4096 and intermediate >= 2048 and experts >= 48:
        if tokens >= 64:
            return 32
        if tokens >= 16:
            return 16
    return 8


def _tile_m(
    token_expert_assignments: int,
    experts: int,
    hidden: int,
    intermediate: int,
) -> tuple[int, int]:
    large = token_expert_assignments >= 256 * experts
    fc1 = 256 if large else 128
    fc2 = 256 if large and hidden > 2048 and intermediate > 1024 else 128
    if intermediate <= 1024:
        fc2 = 128
    return fc1, fc2


def _check_positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(slots=True)
class _FullExecution:
    fc1: GroupedNvfp4Gemm
    fc2: GroupedNvfp4Gemm
    qx: torch.Tensor
    qh: torch.Tensor
    sfx: torch.Tensor
    sfh: torch.Tensor
    routed_out: torch.Tensor


@dataclass(slots=True)
class InferenceWorkspace:
    """Reusable buffers for plans that execute serially on one stream."""

    hidden_size: int
    intermediate_size: int
    experts: int
    topk: int
    max_tokens: int
    device: torch.device
    storage: torch.Tensor
    qx_u8: torch.Tensor
    qh_u8: torch.Tensor
    sfx: torch.Tensor
    sfh: torch.Tensor
    routed_out: torch.Tensor
    out: torch.Tensor
    gather: torch.Tensor
    m_indptr: torch.Tensor
    probs: torch.Tensor
    slots: torch.Tensor
    parts: torch.Tensor
    padded_offsets: torch.Tensor

    @classmethod
    def allocate(
        cls,
        hidden_size: int,
        intermediate_size: int,
        experts: int,
        topk: int,
        max_tokens: int,
        device: torch.device,
    ) -> InferenceWorkspace:
        rows = max_tokens * topk
        prefill_sf_rows = -(-rows // 128) + experts
        decode_rows = min(max_tokens, 64) * topk
        decode_sf_rows = -(-decode_rows // 8) + experts
        sf_rows = max(prefill_sf_rows, decode_sf_rows)
        sfx_shape = (1, sf_rows, hidden_size // 64, 32, 4, 4)
        sfh_shape = (1, sf_rows, intermediate_size // 64, 32, 4, 4)
        qx_bytes = rows * hidden_size // 2
        qh_bytes = rows * intermediate_size // 2
        sfx_bytes = math.prod(sfx_shape)
        sfh_bytes = math.prod(sfh_shape)
        routed_bytes = rows * hidden_size * 2
        out_bytes = max_tokens * hidden_size * 2
        first_bytes = max(qx_bytes + sfx_bytes, routed_bytes)
        second_bytes = max(qh_bytes + sfh_bytes, out_bytes)
        metadata_bytes = 4 * (rows * 3 + experts + 1 + B_MAX * experts + experts)
        storage_bytes = first_bytes + second_bytes + metadata_bytes
        with torch.cuda.device(device):
            storage = torch.empty(storage_bytes, dtype=torch.uint8, device=device)

            def tensor(offset, byte_count, dtype, shape):
                return storage[offset : offset + byte_count].view(dtype).view(shape)

            qx_u8 = tensor(0, qx_bytes, torch.uint8, (rows, hidden_size // 2))
            sfx = tensor(qx_bytes, sfx_bytes, torch.float8_e4m3fn, sfx_shape)
            routed_out = tensor(
                0,
                routed_bytes,
                torch.bfloat16,
                (rows, hidden_size),
            )
            qh_u8 = tensor(
                first_bytes,
                qh_bytes,
                torch.uint8,
                (rows, intermediate_size // 2),
            )
            sfh = tensor(
                first_bytes + qh_bytes,
                sfh_bytes,
                torch.float8_e4m3fn,
                sfh_shape,
            )
            out = tensor(
                first_bytes,
                out_bytes,
                torch.bfloat16,
                (max_tokens, hidden_size),
            )
            cursor = first_bytes + second_bytes

            def int32(shape):
                nonlocal cursor
                elements = math.prod(shape)
                result = tensor(cursor, elements * 4, torch.int32, shape)
                cursor += elements * 4
                return result

            gather = int32((rows,))
            m_indptr = int32((experts + 1,))
            probs = tensor(cursor, rows * 4, torch.float32, (rows,))
            cursor += rows * 4
            slots = int32((rows,))
            parts = int32((B_MAX * experts,))
            padded_offsets = int32((experts,))
            assert cursor == storage_bytes
            return cls(
                hidden_size,
                intermediate_size,
                experts,
                topk,
                max_tokens,
                device,
                storage,
                qx_u8,
                qh_u8,
                sfx,
                sfh,
                routed_out,
                out,
                gather,
                m_indptr,
                probs,
                slots,
                parts,
                padded_offsets,
            )

    @property
    def nbytes(self) -> int:
        return self.storage.numel()

    def matches(
        self,
        hidden_size: int,
        intermediate_size: int,
        experts: int,
        topk: int,
        max_tokens: int,
        device: torch.device,
    ) -> bool:
        return (
            self.hidden_size,
            self.intermediate_size,
            self.experts,
            self.topk,
            self.max_tokens,
            self.device,
        ) == (hidden_size, intermediate_size, experts, topk, max_tokens, device)


class InferenceMoE:
    """Static-scale expert plan with caller-selected router assignments."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        max_tokens: int,
        *,
        activation: str = "swiglu",
        activation_clamp: float | None = None,
        use_dynamic_sched: bool | None = None,
        device: torch.device | str = "cuda",
        workspace: InferenceWorkspace | None = None,
    ):
        self.hidden_size = _check_positive("hidden_size", hidden_size)
        self.intermediate_size = _check_positive("intermediate_size", intermediate_size)
        self.experts = _check_positive("num_experts", num_experts)
        self.topk = _check_positive("top_k", top_k)
        self.max_tokens = _check_positive("max_tokens", max_tokens)
        if self.hidden_size % 256 or self.intermediate_size % 128:
            raise ValueError("hidden_size must align to 256 and intermediate_size to 128")
        if self.experts > 256:
            raise ValueError("at most 256 local experts are supported")
        if self.topk > self.experts:
            raise ValueError("top_k cannot exceed the local expert count")
        if self.max_tokens * self.topk > B_MAX * CHUNK:
            raise ValueError(f"max_tokens * top_k cannot exceed {B_MAX * CHUNK}")
        if activation not in ("swiglu", "geglu", "reglu"):
            raise ValueError("activation must be swiglu, geglu, or reglu")
        if activation_clamp is not None:
            activation_clamp = float(activation_clamp)
            if not math.isfinite(activation_clamp) or activation_clamp <= 0:
                raise ValueError("activation_clamp must be finite and positive")
            if activation != "swiglu":
                raise ValueError("activation_clamp is supported only for swiglu")

        self.activation = activation
        self.activation_clamp = activation_clamp
        self.use_dynamic_sched = use_dynamic_sched
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("InferenceMoE requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

        if workspace is None:
            workspace = InferenceWorkspace.allocate(
                self.hidden_size,
                self.intermediate_size,
                self.experts,
                self.topk,
                self.max_tokens,
                self.device,
            )
        elif not workspace.matches(
            self.hidden_size,
            self.intermediate_size,
            self.experts,
            self.topk,
            self.max_tokens,
            self.device,
        ):
            raise ValueError("workspace configuration must match the inference plan")
        self.workspace = workspace
        self._qx_u8 = workspace.qx_u8
        self._qh_u8 = workspace.qh_u8
        self._sfx = workspace.sfx
        self._sfh = workspace.sfh
        self._routed_out = workspace.routed_out
        self._out = workspace.out
        self._gather = workspace.gather
        self._m_indptr = workspace.m_indptr
        self._probs = workspace.probs
        self._slots = workspace.slots
        self._parts = workspace.parts
        self._padded_offsets = workspace.padded_offsets
        with torch.cuda.device(self.device):
            self._x_scale = torch.empty(2, dtype=torch.float32, device=self.device)
            self._h_scale = torch.empty(2, dtype=torch.float32, device=self.device)
            self._alpha1 = torch.empty(1, dtype=torch.float32, device=self.device)
            self._alpha2 = torch.empty(1, dtype=torch.float32, device=self.device)

        self._weights_ready = False
        self._scales_ready = False
        self._full_executions: dict[tuple[int, bool], _FullExecution] = {}
        self._direct_execution_plan: _FullExecution | None = None
        self._routed_gemms: dict[tuple[str, int], GroupedNvfp4Gemm] = {}
        self._calibration_gemms: dict[int, GroupedNvfp4Gemm] = {}

    @property
    def output(self) -> torch.Tensor:
        """Return the reusable full-MoE output workspace."""
        return self._out

    @property
    def workspace_bytes(self) -> int:
        private = (self._x_scale, self._h_scale, self._alpha1, self._alpha2)
        return self.workspace.nbytes + sum(t.numel() * t.element_size() for t in private)

    def _check_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device != self.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous {dtype} on {self.device} with shape {shape}"
            )

    @torch.no_grad()
    def load_weights(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        down: torch.Tensor,
    ) -> InferenceMoE:
        e, d, i = self.experts, self.hidden_size, self.intermediate_size
        self._check_tensor(gate, "gate", (e, i, d), torch.bfloat16)
        self._check_tensor(up, "up", (e, i, d), torch.bfloat16)
        self._check_tensor(down, "down", (e, d, i), torch.bfloat16)

        gate_up = [torch.stack((gate[x], up[x]), dim=1).reshape(2 * i, d) for x in range(e)]
        amax1 = torch.stack((gate.abs().amax(), up.abs().amax())).amax()
        amax2 = down.abs().amax()
        p1 = (amax1.float() / _DEN).clamp_min(1e-30).reshape(1)
        p2 = (amax2.float() / _DEN).clamp_min(1e-30).reshape(1)
        q1, sf1 = quantize_expert_stack(gate_up, p1)
        q2, sf2 = quantize_expert_stack([down[x] for x in range(e)], p2)
        return self.load_packed_weights(q1, sf1, p1, q2, sf2, p2)

    @torch.no_grad()
    def load_packed_weights(
        self,
        gate_up: torch.Tensor,
        gate_up_sf: torch.Tensor,
        gate_up_scale: torch.Tensor,
        down: torch.Tensor,
        down_sf: torch.Tensor,
        down_scale: torch.Tensor,
    ) -> InferenceMoE:
        e, d, i = self.experts, self.hidden_size, self.intermediate_size
        self._check_tensor(
            gate_up,
            "gate_up",
            (e, 2 * i, d // 2),
            torch.float4_e2m1fn_x2,
        )
        self._check_tensor(
            gate_up_sf,
            "gate_up_sf",
            (e, -(-2 * i // 128), d // 64, 32, 4, 4),
            torch.float8_e4m3fn,
        )
        self._check_tensor(down, "down", (e, d, i // 2), torch.float4_e2m1fn_x2)
        self._check_tensor(
            down_sf,
            "down_sf",
            (e, -(-d // 128), i // 64, 32, 4, 4),
            torch.float8_e4m3fn,
        )
        self._check_weight_scale(gate_up_scale, "gate_up_scale")
        self._check_weight_scale(down_scale, "down_scale")
        if self._alpha1.shape != gate_up_scale.shape:
            self._alpha1 = torch.empty_like(gate_up_scale)
        if self._alpha2.shape != down_scale.shape:
            self._alpha2 = torch.empty_like(down_scale)
        self.qb1, self.sfb1, self.p_w1 = gate_up, gate_up_sf, gate_up_scale
        self.qb2, self.sfb2, self.p_w2 = down, down_sf, down_scale
        self._weights_ready = True
        self._full_executions.clear()
        self._direct_execution_plan = None
        self._routed_gemms.clear()
        self._calibration_gemms.clear()
        if self._scales_ready:
            self._refresh_alphas()
        return self

    def _check_weight_scale(self, scale: torch.Tensor, name: str) -> None:
        if (
            scale.dtype != torch.float32
            or tuple(scale.shape) not in ((1,), (self.experts,))
            or scale.device != self.device
            or not scale.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous float32 on {self.device} "
                f"with shape (1,) or ({self.experts},)"
            )

    @torch.no_grad()
    def set_activation_scales(
        self,
        input_scale: float | torch.Tensor,
        hidden_scale: float | torch.Tensor,
    ) -> InferenceMoE:
        self._set_scale(self._x_scale, input_scale, "input_scale")
        self._set_scale(self._h_scale, hidden_scale, "hidden_scale")
        self._scales_ready = True
        if self._weights_ready:
            self._refresh_alphas()
        return self

    @torch.no_grad()
    def _set_scale(self, target: torch.Tensor, value, name: str) -> None:
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"{name} must contain one value")
            target[:1].copy_(value.detach().to(device=self.device, dtype=torch.float32).reshape(1))
        else:
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            target[0] = value
        torch.reciprocal(target[:1], out=target[1:])

    @torch.no_grad()
    def _refresh_alphas(self) -> None:
        torch.mul(self._x_scale[:1], self.p_w1, out=self._alpha1)
        torch.mul(self._h_scale[:1], self.p_w2, out=self._alpha2)

    def _require_weights(self) -> None:
        if not self._weights_ready:
            raise RuntimeError("load_weights or load_packed_weights must run first")

    def _require_ready(self) -> None:
        self._require_weights()
        if not self._scales_ready:
            raise RuntimeError("calibrate or set_activation_scales must run first")

    def _views(self, rows: int, *, sf_tile_rows: int = 128):
        sf_rows = -(-rows // sf_tile_rows) + self.experts
        qx = self._qx_u8[:rows].view(torch.float4_e2m1fn_x2)
        qh = self._qh_u8[:rows].view(torch.float4_e2m1fn_x2)
        return qx, qh, self._sfx[:, :sf_rows], self._sfh[:, :sf_rows]

    def _new_gemm(
        self,
        projection: str,
        rows: int,
        output_dtype=torch.bfloat16,
        *,
        decode: bool = False,
    ):
        schedule = False if decode and self.use_dynamic_sched is None else self.use_dynamic_sched
        fc1_m, fc2_m = _tile_m(
            rows,
            self.experts,
            self.hidden_size,
            self.intermediate_size,
        )
        if projection == "fc1":
            return GroupedNvfp4Gemm(
                self.experts,
                2 * self.intermediate_size,
                self.hidden_size,
                fc1_m,
                128,
                output_dtype=output_dtype,
                epilogue=(
                    GatedEpilogue(
                        self.activation,
                        save_preact=False,
                        clamp_limit=self.activation_clamp,
                    )
                    if output_dtype == torch.float4_e2m1fn_x2
                    else None
                ),
                use_dynamic_sched=schedule,
                use_pdl=True,
                swap_ab=False,
                fast_decode_sched=(decode and not schedule and self.hidden_size >= 2048),
            )
        return GroupedNvfp4Gemm(
            self.experts,
            self.hidden_size,
            self.intermediate_size,
            128 if decode else fc2_m,
            8 if decode else 128,
            use_dynamic_sched=schedule,
            use_pdl=True,
            swap_ab=decode,
        )

    def _full_execution(self, tokens: int, *, decode: bool) -> _FullExecution:
        key = (tokens, decode)
        execution = self._full_executions.get(key)
        if execution is not None:
            return execution
        rows = tokens * self.topk
        fused_decode = decode and tokens <= 64
        fused_tile_n = _decode_tile_rows(
            tokens,
            self.hidden_size,
            self.intermediate_size,
            self.experts,
        )
        qx, qh, sfx, sfh = self._views(
            rows,
            sf_tile_rows=fused_tile_n if fused_decode else 128,
        )
        routed_out = self._routed_out[:rows]
        with torch.cuda.device(self.device):
            if fused_decode:
                fc1 = GroupedNvfp4Gemm(
                    self.experts,
                    2 * self.intermediate_size,
                    self.hidden_size,
                    128,
                    fused_tile_n,
                    output_dtype=torch.float4_e2m1fn_x2,
                    epilogue=GatedEpilogue(
                        self.activation,
                        save_preact=False,
                        clamp_limit=self.activation_clamp,
                    ),
                    use_dynamic_sched=False,
                    use_pdl=True,
                    swap_ab=True,
                    fast_decode_sched=True,
                    gather_b=True,
                )
            else:
                fc1 = self._new_gemm(
                    "fc1",
                    rows,
                    torch.float4_e2m1fn_x2,
                    decode=decode and rows <= FAST_DECODE_ROWS,
                )
            narrow_fc2 = decode and _use_swapped_fc2(tokens)
            if fused_decode:
                fc2 = GroupedNvfp4Gemm(
                    self.experts,
                    self.hidden_size,
                    self.intermediate_size,
                    128,
                    fused_tile_n,
                    use_dynamic_sched=False,
                    use_pdl=True,
                    swap_ab=True,
                    fast_decode_sched=True,
                )
                fc2._direct_c_store = tokens > 1
            else:
                fc2 = self._new_gemm("fc2", rows, decode=narrow_fc2)
        fc1_input = qx[:tokens] if fc1.gather_b else qx
        gather_idx = self._gather[:rows] if fc1.gather_b else None
        fc1.prepare(
            fc1_input,
            self.qb1,
            qh,
            self._m_indptr,
            sfx,
            self.sfb1,
            self._alpha1,
            output_sf=sfh,
            output_scale=self._h_scale,
            decode_plan=self._parts if fc1.fast_decode_sched else None,
            gather_idx=gather_idx,
        )
        fc2.prepare(
            qh,
            self.qb2,
            routed_out,
            self._m_indptr,
            sfh,
            self.sfb2,
            self._alpha2,
            decode_plan=self._parts if fc2.fast_decode_sched else None,
        )
        execution = _FullExecution(fc1, fc2, qx, qh, sfx, sfh, routed_out)
        self._full_executions[key] = execution
        return execution

    def _direct_execution(self) -> _FullExecution:
        execution = self._direct_execution_plan
        if execution is not None:
            return execution
        rows = self.topk
        qx, qh, sfx, sfh = self._views(rows)
        routed_out = self._routed_out[:rows]
        route_ids = self._m_indptr[: self.topk]
        with torch.cuda.device(self.device):
            fc1 = GroupedNvfp4Gemm(
                self.experts,
                2 * self.intermediate_size,
                self.hidden_size,
                128,
                128,
                output_dtype=torch.float4_e2m1fn_x2,
                epilogue=GatedEpilogue(
                    self.activation,
                    save_preact=False,
                    clamp_limit=self.activation_clamp,
                ),
                use_dynamic_sched=False,
                use_pdl=True,
                direct_routes=self.topk,
                broadcast_a=True,
            )
            fc2 = GroupedNvfp4Gemm(
                self.experts,
                self.hidden_size,
                self.intermediate_size,
                128,
                128,
                use_dynamic_sched=False,
                use_pdl=True,
                direct_routes=self.topk,
            )
        fc1.prepare(
            qx,
            self.qb1,
            qh,
            route_ids,
            sfx,
            self.sfb1,
            self._alpha1,
            output_sf=sfh,
            output_scale=self._h_scale,
        )
        fc2.prepare(
            qh,
            self.qb2,
            routed_out,
            route_ids,
            sfh,
            self.sfb2,
            self._alpha2,
        )
        execution = _FullExecution(fc1, fc2, qx, qh, sfx, sfh, routed_out)
        self._direct_execution_plan = execution
        return execution

    @torch.no_grad()
    def calibrate_routed(
        self,
        x: torch.Tensor,
        m_indptr: torch.Tensor,
        padded_offsets: torch.Tensor,
    ) -> InferenceMoE:
        self._require_weights()
        rows = x.shape[0]
        self._validate_routed(x, m_indptr, padded_offsets)
        input_scale = (x.abs().amax().float() / _DEN).clamp_min(1e-30).reshape(1)
        self._set_scale(self._x_scale, input_scale, "input_scale")
        qx, _, sfx, _ = self._views(rows)
        nvfp4_quantize_rowwise(
            x,
            m_indptr,
            self._x_scale,
            qx,
            sfx,
            padded_offsets=padded_offsets,
            te_math=True,
        )
        preact = torch.empty(
            rows, 2 * self.intermediate_size, dtype=torch.bfloat16, device=self.device
        )
        fc1 = self._calibration_gemms.get(rows)
        if fc1 is None:
            with torch.cuda.device(self.device):
                fc1 = self._new_gemm("fc1", rows)
            self._calibration_gemms[rows] = fc1
        torch.mul(self._x_scale[:1], self.p_w1, out=self._alpha1)
        fc1(qx, self.qb1, preact, m_indptr, sfx, self.sfb1, self._alpha1)
        gate = preact[:, 0::2].float()
        up = preact[:, 1::2].float()
        if self.activation_clamp is not None:
            gate.clamp_(max=self.activation_clamp)
            up.clamp_(min=-self.activation_clamp, max=self.activation_clamp)
        if self.activation == "swiglu":
            hidden = torch.nn.functional.silu(gate).mul_(up)
        elif self.activation == "geglu":
            hidden = torch.nn.functional.gelu(gate, approximate="tanh").mul_(up)
        else:
            hidden = torch.nn.functional.relu(gate).mul_(up)
        hidden_scale = (hidden.abs().amax().float() / _DEN).clamp_min(1e-30).reshape(1)
        self._set_scale(self._h_scale, hidden_scale, "hidden_scale")
        self._scales_ready = True
        self._refresh_alphas()
        return self

    @torch.no_grad()
    def calibrate(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> InferenceMoE:
        tokens = self._validate_full(x, topk_ids, topk_weights)
        rows = tokens * self.topk
        moe_dispatch(
            topk_ids,
            topk_weights,
            self.experts,
            self._gather[:rows],
            self._m_indptr,
            self._probs[:rows],
            self._slots[:rows],
            self._parts,
            self._padded_offsets,
        )
        gathered = x.index_select(0, self._gather[:rows].long())
        return self.calibrate_routed(gathered, self._m_indptr, self._padded_offsets)

    def _validate_routed(self, x, m_indptr, padded_offsets) -> int:
        rows = int(x.shape[0])
        if rows <= 0 or rows > self.max_tokens * self.topk:
            raise ValueError("token-expert assignment rows must be within the configured workspace")
        self._check_tensor(x, "x", (rows, self.hidden_size), torch.bfloat16)
        self._check_tensor(m_indptr, "m_indptr", (self.experts + 1,), torch.int32)
        self._check_tensor(padded_offsets, "padded_offsets", (self.experts,), torch.int32)
        return rows

    def _validate_full(self, x, topk_ids, topk_weights) -> int:
        tokens = int(x.shape[0])
        if tokens <= 0 or tokens > self.max_tokens:
            raise ValueError("token count must be within the configured workspace")
        self._check_tensor(x, "x", (tokens, self.hidden_size), torch.bfloat16)
        self._check_tensor(topk_ids, "topk_ids", (tokens, self.topk), torch.int32)
        self._check_tensor(topk_weights, "topk_weights", (tokens, self.topk), torch.float32)
        return tokens

    def _output_view(self, output, rows: int, name: str):
        if output is None:
            source = self._routed_out if name == "routed_out" else self._out
            return source[:rows]
        self._check_tensor(output, name, (rows, self.hidden_size), torch.bfloat16)
        return output

    @torch.no_grad()
    def run_routed(
        self,
        x: torch.Tensor,
        m_indptr: torch.Tensor,
        padded_offsets: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._require_ready()
        rows = self._validate_routed(x, m_indptr, padded_offsets)
        qx, qh, sfx, sfh = self._views(rows)
        routed_out = self._output_view(out, rows, "routed_out")
        nvfp4_quantize_rowwise(
            x,
            m_indptr,
            self._x_scale,
            qx,
            sfx,
            padded_offsets=padded_offsets,
            te_math=True,
        )
        fc1_m, fc2_m = _tile_m(
            rows,
            self.experts,
            self.hidden_size,
            self.intermediate_size,
        )
        with torch.cuda.device(self.device):
            fc1 = self._routed_gemms.get(("fc1", fc1_m))
            if fc1 is None:
                fc1 = self._new_gemm("fc1", rows, torch.float4_e2m1fn_x2)
                self._routed_gemms[("fc1", fc1_m)] = fc1
            fc2 = self._routed_gemms.get(("fc2", fc2_m))
            if fc2 is None:
                fc2 = self._new_gemm("fc2", rows)
                self._routed_gemms[("fc2", fc2_m)] = fc2
        fc1(
            qx,
            self.qb1,
            qh,
            m_indptr,
            sfx,
            self.sfb1,
            self._alpha1,
            output_sf=sfh,
            output_scale=self._h_scale,
        )
        fc2(qh, self.qb2, routed_out, m_indptr, sfh, self.sfb2, self._alpha2)
        return routed_out

    @torch.no_grad()
    def run_prefill(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._require_ready()
        tokens = self._validate_full(x, topk_ids, topk_weights)
        rows = tokens * self.topk
        output = self._output_view(out, tokens, "out")
        execution = self._full_execution(tokens, decode=False)
        moe_dispatch(
            topk_ids,
            topk_weights,
            self.experts,
            self._gather[:rows],
            self._m_indptr,
            self._probs[:rows],
            self._slots[:rows],
            self._parts,
            self._padded_offsets,
        )
        nvfp4_quantize_rowwise(
            x,
            self._m_indptr,
            self._x_scale,
            execution.qx,
            execution.sfx,
            gather_idx=self._gather[:rows],
            padded_offsets=self._padded_offsets,
            te_math=True,
        )
        execution.fc1.launch()
        execution.fc2.launch()
        moe_finalize(
            execution.routed_out,
            self._slots[:rows],
            output,
            self.topk,
            tile_t=1,
            n_frag=2,
            weights=self._probs[:rows],
            use_pdl=True,
        )
        return output

    @torch.no_grad()
    def run_decode(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._require_ready()
        tokens = self._validate_full(x, topk_ids, topk_weights)
        rows = tokens * self.topk
        output = self._output_view(out, tokens, "out")
        execution = self._full_execution(tokens, decode=True)
        plan_tile_rows = execution.fc1.tile_n if execution.fc1.fast_decode_sched else 128
        if execution.fc1.gather_b:
            nvfp4_dispatch_quantize_decode(
                x,
                topk_ids,
                topk_weights,
                self._parts,
                self._gather[:rows],
                self._m_indptr,
                self._probs[:rows],
                self._slots[:rows],
                self._padded_offsets,
                self._x_scale,
                execution.qx[:tokens],
                execution.sfx,
                plan_tile_rows=plan_tile_rows,
            )
        else:
            moe_dispatch(
                topk_ids,
                topk_weights,
                self.experts,
                self._gather[:rows],
                self._m_indptr,
                self._probs[:rows],
                self._slots[:rows],
                self._parts,
                self._padded_offsets,
                plan_tile_rows=plan_tile_rows,
            )
            nvfp4_quantize_decode(
                x,
                topk_ids,
                self._slots[:rows],
                self._m_indptr,
                self._padded_offsets,
                self._x_scale,
                execution.qx,
                execution.sfx,
                sf_tile_rows=plan_tile_rows,
            )
        execution.fc1.launch()
        execution.fc2.launch()
        moe_finalize(
            execution.routed_out,
            self._slots[:rows],
            output,
            self.topk,
            tile_t=1,
            n_frag=2,
            weights=self._probs[:rows],
            use_pdl=True,
            num_threads=768 if self.hidden_size >= 2048 else 256,
            broadcast_slots=False,
        )
        return output

    @torch.no_grad()
    def run(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rows = x.shape[0] * self.topk
        if rows <= DECODE_ROWS_PER_EXPERT * self.experts:
            return self.run_decode(x, topk_ids, topk_weights, out=out)
        return self.run_prefill(x, topk_ids, topk_weights, out=out)

    __call__ = run

    @torch.no_grad()
    def warmup(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        iterations: int = 2,
    ) -> None:
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        for _ in range(iterations):
            self.run(x, topk_ids, topk_weights)
        torch.cuda.synchronize(self.device)


__all__ = ["InferenceMoE", "InferenceWorkspace"]
