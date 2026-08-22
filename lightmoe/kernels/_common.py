# Copyright (c) 2025-2026, Wentao Guo, Ted Zadouri, Vijay Thakkar, Tri Dao.
# Copyright (c) 2026, Sung Hyun Cho.
# SPDX-License-Identifier: Apache-2.0
"""Small CuTe DSL helpers shared by project kernels."""

import operator
from functools import cache
from typing import get_origin

import cutlass
import torch
from cutlass import Float32, Uint32, Uint64, cute
from cutlass._mlir.dialects import llvm
from cutlass._mlir_helpers.arith import bitcast as _bitcast
from cutlass.base_dsl.tvm_ffi_builder import spec
from cutlass.cutlass_dsl import T, dsl_user_op

# Keep compiled operators alive for the lifetime of the process. CuTe still
# owns its compiler cache; this only avoids retracing identical Python calls.
jit_cache = cache


def fake_tensor(dtype, shape, divisibility=1, leading_dim=-1):
    """Build a fake tensor for TVM-FFI compilation."""
    if dtype is None:
        return None
    if leading_dim is not None and leading_dim < 0:
        leading_dim += len(shape)
    stride = tuple(
        cute.sym_int64(divisibility=divisibility) if i != leading_dim else 1
        for i in range(len(shape))
    )
    alignment = max(divisibility * dtype.width // 8, 1)
    return cute.runtime.make_fake_tensor(dtype, shape, stride=stride, assumed_align=alignment)


torch2cute_dtype_map = {
    torch.uint8: cutlass.Uint8,
    torch.float4_e2m1fn_x2: cutlass.Float4E2M1FN,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float8_e5m2: cutlass.Float8E5M2,
    torch.float8_e8m0fnu: cutlass.Float8E8M0FNU,
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
}


def _install_ffi_converter_patch():
    """Teach the CuTe TVM-FFI converter about Constexpr and NamedTuple args."""
    import cutlass.cute._tvm_ffi_args_spec_converter as converter

    if getattr(converter, "_lightmoe_patched", False):
        return
    original = converter._convert_single_arg

    def convert_single_arg(arg, arg_name, arg_type, ctx):
        if arg_type is not None and get_origin(arg_type) is cutlass.Constexpr:
            return spec.ConstNone(arg_name)
        if (
            isinstance(arg, tuple)
            and hasattr(type(arg), "_fields")
            and (arg_type is None or not hasattr(arg_type, "_fields"))
        ):
            return original(arg, arg_name, type(arg), ctx)
        return original(arg, arg_name, arg_type, ctx)

    converter._convert_single_arg = convert_single_arg
    converter._lightmoe_patched = True


_install_ffi_converter_patch()


def _f32_bits(x: Float32) -> Uint32:
    return Uint32(_bitcast(Float32(x).ir_value(), T.i32()))


def _bits_f32(x: Uint32) -> Float32:
    return Float32(_bitcast(Uint32(x).ir_value(), T.f32()))


def _asm(result_type, ptx: str, constraints: str, args):
    return llvm.inline_asm(
        result_type,
        args,
        ptx,
        constraints,
        has_side_effects=False,
        is_align_stack=False,
    )


@cute.jit
def warp_sum(value):
    return cute.arch.warp_reduction(value, operator.add, threads_in_group=cute.arch.WARP_SIZE)


@dsl_user_op
def cvt_f32x4_e2m1x4_rs(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    rand_bits: Uint32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Uint16:
    """Use Blackwell's packed FP4 stochastic-rounding conversion."""
    values = (v0, v1, v2, v3)
    return cutlass.Uint16(
        _asm(
            T.i16(),
            "cvt.rs.satfinite.e2m1x4.f32 $0, {$4, $3, $2, $1}, $5;",
            "=h,f,f,f,f,r",
            [Float32(v).ir_value(loc=loc, ip=ip) for v in values]
            + [Uint32(rand_bits).ir_value(loc=loc, ip=ip)],
        )
    )


PHILOX_N_ROUNDS = 7
PHILOX_ROUND_A = 0xD2511F53
PHILOX_ROUND_B = 0xCD9E8D57
PHILOX_KEY_A = 0x9E3779B9
PHILOX_KEY_B = 0xBB67AE85


@dsl_user_op
def _mul_wide_u32(a: Uint32, b: Uint32, *, loc=None, ip=None):
    product = cute.arch.mul_wide(Uint32(a), Uint32(b), loc=loc, ip=ip)
    return (
        (product >> 32).to(Uint32, loc=loc, ip=ip),
        product.to(Uint32, loc=loc, ip=ip),
    )


@dsl_user_op
def philox(counter, key, n_rounds: int = PHILOX_N_ROUNDS, *, loc=None, ip=None):
    """Return four Philox 4x32 random words."""
    if cutlass.const_expr(isinstance(counter, Uint64)):
        c0 = (counter & Uint64(0xFFFFFFFF)).to(Uint32)
        c1 = (counter >> Uint64(32)).to(Uint32)
    else:
        c0, c1 = Uint32(counter), Uint32(0)
    c2, c3 = Uint32(0), Uint32(0)

    if cutlass.const_expr(isinstance(key, Uint64)):
        k0 = (key & Uint64(0xFFFFFFFF)).to(Uint32)
        k1 = (key >> Uint64(32)).to(Uint32)
    else:
        k0, k1 = Uint32(key), Uint32(0)

    round_a, round_b = Uint32(PHILOX_ROUND_A), Uint32(PHILOX_ROUND_B)
    key_a, key_b = Uint32(PHILOX_KEY_A), Uint32(PHILOX_KEY_B)
    for _ in range(n_rounds):
        hi_b, lo_b = _mul_wide_u32(c2, round_b, loc=loc, ip=ip)
        hi_a, lo_a = _mul_wide_u32(c0, round_a, loc=loc, ip=ip)
        c0, c2 = hi_b ^ c1 ^ k0, hi_a ^ c3 ^ k1
        c1, c3 = lo_b, lo_a
        k0, k1 = k0 + key_a, k1 + key_b
    return c0, c1, c2, c3
