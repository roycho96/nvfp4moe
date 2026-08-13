# Copyright (c) 2026, Sung Hyun Cho.
# SPDX-License-Identifier: Apache-2.0

"""Fused gated activation and NVFP4 output helpers for SM100."""

import math
from dataclasses import dataclass

import cutlass
from cutlass import Boolean, Float32, const_expr, cute
from cutlass._mlir.dialects import arith

from .quantize import F4_MAX, _cvt_e4m3_rn

_ACTIVATIONS = ("swiglu", "geglu", "reglu")
FLT_MAX = 3.4028234663852886e38


def gated_output_n(accumulator_n: int) -> int:
    """Return the half-width output extent for an interleaved accumulator."""
    if accumulator_n <= 0 or accumulator_n % 2:
        raise ValueError("gated accumulator N must be a positive even value")
    return accumulator_n // 2


def validate_gated_tile_n(tile_n: int) -> int:
    """Validate the four-slot blocked-SF alignment of a gated NVFP4 tile."""
    if tile_n <= 0 or tile_n % 128:
        raise ValueError("gated NVFP4 tile_N must be a positive multiple of 128")
    return tile_n


def validate_gated_activation(activation: str) -> str:
    if activation not in _ACTIVATIONS:
        choices = ", ".join(_ACTIVATIONS)
        raise ValueError(f"activation must be one of: {choices}")
    return activation


@dataclass(frozen=True, slots=True)
class GatedEpilogue:
    """Forward gated activation selected at GEMM compile time."""

    activation: str = "swiglu"
    save_preact: bool = False

    def __post_init__(self):
        object.__setattr__(self, "activation", validate_gated_activation(self.activation))


@dataclass(frozen=True, slots=True)
class GatedBackwardEpilogue:
    """Fused gated derivative selected at GEMM compile time."""

    activation: str = "swiglu"

    def __post_init__(self):
        object.__setattr__(self, "activation", validate_gated_activation(self.activation))


def resolve_gemm_epilogue(
    epilogue: GatedEpilogue | GatedBackwardEpilogue | None,
    activation: str | None,
    dactivation: str | None,
) -> tuple[str | None, str | None]:
    if epilogue is not None and (activation is not None or dactivation is not None):
        raise ValueError("epilogue cannot be combined with activation or dactivation")
    if isinstance(epilogue, GatedEpilogue):
        return epilogue.activation, None
    if isinstance(epilogue, GatedBackwardEpilogue):
        return None, epilogue.activation
    if epilogue is not None:
        raise TypeError("epilogue must be GatedEpilogue or GatedBackwardEpilogue")
    return activation, dactivation


def gated_sf_u32_word_count(
    tile_m: int,
    tile_n: int,
    epi_m: int,
    epi_n: int,
) -> int:
    """Return the number of packed SF words when N subtiles form full atoms."""
    if min(tile_m, tile_n, epi_m, epi_n) <= 0:
        return 0
    if epi_m != tile_m or epi_n != 64 or tile_n % (2 * epi_n):
        return 0
    return tile_n // (2 * epi_n)


def gated_postact_shape(accumulator_layout: cute.Layout):
    """Return the register shape after pairing adjacent accumulator values."""
    if cute.size(accumulator_layout) % 2:
        raise ValueError("gated accumulator fragment must contain an even number of values")
    return cute.recast_layout(2, 1, accumulator_layout).shape


@cute.jit
def gated_postact_fragment(
    accumulator: cute.Tensor,
    alpha: Float32,
    activation: cutlass.Constexpr[str],
) -> cute.Tensor:
    """Apply a gated activation to adjacent FP32 accumulator values.

    ``alpha`` is a scalar value loaded by the caller before entering this
    helper. Even accumulator values are gates and odd values are up values.
    """
    validate_gated_activation(activation)
    output = cute.make_rmem_tensor(gated_postact_shape(accumulator.layout), Float32)
    values = cute.make_tensor(
        accumulator.iterator,
        cute.make_layout(cute.size(accumulator)),
    )
    count = cute.size(output)
    for index in cutlass.range_constexpr(count):
        gate = values[index * 2] * alpha
        up = values[index * 2 + 1] * alpha
        if const_expr(activation == "swiglu"):
            sigmoid = cute.arch.rcp_approx(Float32(1.0) + cute.math.exp(-gate, fastmath=True))
            output[index] = gate * sigmoid * up
        elif const_expr(activation == "geglu"):
            root = math.sqrt(2.0 / math.pi)
            argument = root * (gate + Float32(0.044715) * gate * gate * gate)
            gelu = Float32(0.5) * gate * (Float32(1.0) + cute.math.tanh(argument, fastmath=True))
            output[index] = gelu * up
        else:
            output[index] = cute.arch.fmax(gate, Float32(0.0)) * up
    return output


@cute.jit
def gated_backward_values(
    gate: Float32,
    up: Float32,
    dout: Float32,
    activation: cutlass.Constexpr[str],
) -> tuple[Float32, Float32, Float32]:
    """Return gate gradient, up gradient, and the recomputed activation."""
    validate_gated_activation(activation)
    if const_expr(activation == "swiglu"):
        sigmoid = cute.arch.rcp_approx(Float32(1.0) + cute.math.exp(-gate, fastmath=True))
        activated = gate * sigmoid
        derivative = sigmoid + gate * sigmoid * (Float32(1.0) - sigmoid)
        return dout * up * derivative, dout * activated, activated * up
    if const_expr(activation == "geglu"):
        root = math.sqrt(2.0 / math.pi)
        cubic = Float32(0.044715) * gate * gate * gate
        argument = root * (gate + cubic)
        tanh_value = cute.math.tanh(argument, fastmath=True)
        half_sum = Float32(0.5) * (Float32(1.0) + tanh_value)
        activated = gate * half_sum
        argument_grad = root * (Float32(1.0) + Float32(0.134145) * gate * gate)
        activation_grad = half_sum + (
            Float32(0.5) * gate * (Float32(1.0) - tanh_value * tanh_value) * argument_grad
        )
        return dout * up * activation_grad, dout * activated, activated * up

    activated = cute.arch.fmax(gate, Float32(0.0))
    positive = Boolean(gate > 0)
    gate_grad = Float32(
        arith.select(
            positive.ir_value(),
            (dout * up).ir_value(),
            Float32(0.0).ir_value(),
        )
    )
    return gate_grad, dout * activated, activated * up


@cute.jit
def swiglu_backward_pair(
    gates: tuple[Float32, Float32],
    ups: tuple[Float32, Float32],
    douts: tuple[Float32, Float32],
) -> tuple[
    tuple[Float32, Float32],
    tuple[Float32, Float32],
    tuple[Float32, Float32],
]:
    """Evaluate two independent SwiGLU derivatives with packed FP32 ALU ops."""
    sigmoids = (
        cute.arch.rcp_approx(Float32(1.0) + cute.math.exp(-gates[0], fastmath=True)),
        cute.arch.rcp_approx(Float32(1.0) + cute.math.exp(-gates[1], fastmath=True)),
    )
    activated = cute.arch.mul_packed_f32x2(gates, sigmoids, ftz=False, rnd="rn")
    one_minus = cute.arch.add_packed_f32x2(
        (Float32(1.0), Float32(1.0)),
        (-sigmoids[0], -sigmoids[1]),
        ftz=False,
        rnd="rn",
    )
    derivative = cute.arch.fma_packed_f32x2(
        activated,
        one_minus,
        sigmoids,
        ftz=False,
        rnd="rn",
    )
    dout_up = cute.arch.mul_packed_f32x2(douts, ups, ftz=False, rnd="rn")
    dgate = cute.arch.mul_packed_f32x2(dout_up, derivative, ftz=False, rnd="rn")
    dup = cute.arch.mul_packed_f32x2(douts, activated, ftz=False, rnd="rn")
    postact = cute.arch.mul_packed_f32x2(activated, ups, ftz=False, rnd="rn")
    return dgate, dup, postact


@cute.jit
def quantize_postact_fragment(
    postact: cute.Tensor,
    inv_pts: Float32,
) -> tuple[cute.Tensor, cute.Tensor]:
    """Quantize a post-activation fragment in independent 16-value groups."""
    fragment_size = cute.size(postact)
    assert fragment_size % 16 == 0
    group_count = fragment_size // 16
    sf_storage = cute.make_rmem_tensor(
        cute.make_layout(group_count),
        cutlass.Float8E4M3FN,
    )
    sf_bytes = cute.make_tensor(
        cute.recast_ptr(sf_storage.iterator, dtype=cutlass.Uint8),
        cute.make_layout(group_count),
    )
    norm_scaled = inv_pts * Float32(1.0 / F4_MAX)
    for group in cutlass.range_constexpr(group_count):
        source_group = cute.make_tensor(
            postact.iterator + group * 16,
            cute.make_layout(16),
        )
        amax = Float32(0.0)
        for index in cutlass.range_constexpr(16):
            amax = cute.arch.fmax(
                amax,
                cute.math.absf(source_group[index]),
            )
        scaled = amax * norm_scaled
        sf_bytes[group] = cutlass.Uint8(_cvt_e4m3_rn(scaled) & 0xFF)
        rescale = cute.arch.rcp_approx(sf_storage[group].to(Float32))
        rescale = cute.arch.fmin(rescale * inv_pts, Float32(FLT_MAX))
        for index in cutlass.range(16, unroll_full=True, vectorize=True):
            source_group[index] *= rescale
    return postact, sf_storage


__all__ = [
    "GatedBackwardEpilogue",
    "GatedEpilogue",
    "gated_backward_values",
    "gated_output_n",
    "gated_postact_fragment",
    "gated_postact_shape",
    "gated_sf_u32_word_count",
    "quantize_postact_fragment",
    "resolve_gemm_epilogue",
    "swiglu_backward_pair",
    "validate_gated_activation",
    "validate_gated_tile_n",
]
