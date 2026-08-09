"""QuACK GEMM wrappers and fused MoE epilogues."""

import functools

from . import _vendor  # noqa: F401
import cutlass.cute as cute  # noqa: E402
from quack.epilogue.frontend import gemm_epilogue  # noqa: E402
from quack.epilogue.library import (dgate_fn_map, dgated_mod,  # noqa: E402
                                    gate_fn_map)
from quack.epilogue.math import pack, unpack  # noqa: E402
from quack.epilogue.ops import ColVecReduce, Scalar, TileStore  # noqa: E402
from quack.epilogue.quantize_out import BlockScaleFactorStore  # noqa: E402
from quack.gemm import gemm as quack_gemm  # noqa: E402


@functools.lru_cache(maxsize=None)
def fc1_quant_mod(activation: str = "swiglu"):
    """FC1 with gated activation and NVFP4 output quantization."""
    act = gate_fn_map[activation]

    @gemm_epilogue(
        outputs=(
            TileStore(
                "postact",
                gated=True,
                quant=BlockScaleFactorStore("postact_sf", output="postact"),
            ),
        ),
        extra_ops=(Scalar("sfd_norm_const"),),
        mode="acc_pair",
    )
    def fc1_epi(acc, ascale):
        g, u = unpack(acc * ascale)
        return {"postact": act(g, u)}

    return fc1_epi


@functools.lru_cache(maxsize=None)
def fc2_weighted_mod():
    """FC2 with fused routing weights and dequantization scales."""

    @gemm_epilogue(outputs=("yw",))
    def fc2_epi(acc, probs, pscale):
        return {"yw": acc * (probs * pscale)}

    return fc2_epi


@functools.lru_cache(maxsize=None)
def dgrad2_mod(activation: str = "swiglu"):
    """FC2 input gradient with fused activation derivative and recompute."""
    return dgated_mod(activation, has_scale=True, has_reduce=False,
                      scalar_scale=True, scale_aux=False)


def _fmax_abs(a, b):
    """Compute lane-wise ``max(abs(a), abs(b))``."""
    if isinstance(a, tuple):  # F2-style packed lanes (trace-time branch)
        return type(a)(*(cute.arch.fmax(x, y, abs=True)
                         for x, y in zip(a, b)))
    return cute.arch.fmax(a, b, abs=True)


@functools.lru_cache(maxsize=None)
def dgrad2_amax_mod(activation: str = "swiglu"):
    """FC2 input gradient with fused row-amax partials."""
    dgate = dgate_fn_map[activation]

    @gemm_epilogue(
        outputs=("mAuxOut",),
        reduces={"damax": ColVecReduce("damax", combine="max_abs")},
        mode="packed_cd_b16x2",
        # Required by a CUTLASS DSL vectorizer limitation around fmax reuse.
        vectorize=False,
    )
    def dgated_amax_epi(acc, c, gscale):
        x, y = unpack(c)
        dx, dy, out = dgate(x, y, acc * gscale)
        return {"D": pack(dx, dy), "mAuxOut": out,
                "damax": _fmax_abs(dx, dy)}

    return dgated_amax_epi


gemm = quack_gemm  # plain varlen blockscaled GEMM (dgrad1, wgrad, recompute)

__all__ = ["fc1_quant_mod", "fc2_weighted_mod", "dgrad2_mod",
           "dgrad2_amax_mod", "gemm"]
