# Copyright (c) 2026, Sung Hyun Cho.
# SPDX-License-Identifier: Apache-2.0

"""Fused unary activation helpers for SM100 grouped GEMM."""

from dataclasses import dataclass

import cutlass
from cutlass import Boolean, Float32, const_expr, cute
from cutlass._mlir.dialects import arith

_ACTIVATIONS = ("relu2",)


def validate_unary_activation(activation: str) -> str:
    if activation not in _ACTIVATIONS:
        raise ValueError("unary activation must be relu2")
    return activation


@dataclass(frozen=True, slots=True)
class UnaryEpilogue:
    """Forward unary activation selected at GEMM compile time."""

    activation: str = "relu2"
    save_preact: bool = False

    def __post_init__(self):
        object.__setattr__(self, "activation", validate_unary_activation(self.activation))


@dataclass(frozen=True, slots=True)
class UnaryBackwardEpilogue:
    """Unary derivative selected at GEMM compile time."""

    activation: str = "relu2"

    def __post_init__(self):
        object.__setattr__(self, "activation", validate_unary_activation(self.activation))


@cute.jit
def unary_postact_value(
    value: Float32,
    activation: cutlass.Constexpr[str],
) -> Float32:
    validate_unary_activation(activation)
    if const_expr(activation == "relu2"):
        positive = cute.arch.fmax(value, Float32(0.0))
        return positive * positive
    return value


@cute.jit
def unary_postact_fragment(
    accumulator: cute.Tensor,
    alpha: Float32,
    activation: cutlass.Constexpr[str],
) -> cute.Tensor:
    validate_unary_activation(activation)
    values = cute.make_tensor(accumulator.iterator, cute.make_layout(cute.size(accumulator)))
    for index in cutlass.range(cute.size(accumulator), unroll_full=True):
        values[index] = unary_postact_value(values[index] * alpha, activation)
    return accumulator


@cute.jit
def unary_backward_values(
    value: Float32,
    dout: Float32,
    activation: cutlass.Constexpr[str],
) -> tuple[Float32, Float32]:
    validate_unary_activation(activation)
    if const_expr(activation == "relu2"):
        positive = Boolean(value > 0)
        derivative = Float32(
            arith.select(
                positive.ir_value(),
                (Float32(2.0) * value).ir_value(),
                Float32(0.0).ir_value(),
            )
        )
        activated = cute.arch.fmax(value, Float32(0.0))
        return dout * derivative, activated * activated
    return dout, value
