"""CuTe DSL kernels used by the expert layer."""

from importlib import import_module

__all__ = [
    "DenseNvfp4Gemm",
    "GatedBackwardEpilogue",
    "GatedEpilogue",
    "GroupedGemmKernel",
    "GroupedNvfp4Gemm",
    "GroupedWgrad",
    "MoEDispatch",
    "gated_backward_values",
    "gated_postact_fragment",
    "grouped_nvfp4_gemm",
    "moe_dispatch",
    "moe_finalize",
    "moe_finalize_bwd",
    "nvfp4_decode_prepare",
    "nvfp4_quantize_colwise",
    "nvfp4_quantize_decode",
    "nvfp4_quantize_rowwise",
    "nvfp4_rht_amax",
    "quantize_postact_fragment",
    "rht_matrix",
    "swiglu_backward_pair",
]

_EXPORTS = {
    "DenseNvfp4Gemm": (".dense.runtime", "DenseNvfp4Gemm"),
    "GatedBackwardEpilogue": (".gated", "GatedBackwardEpilogue"),
    "GatedEpilogue": (".gated", "GatedEpilogue"),
    "GroupedGemmKernel": (".grouped.kernel", "GroupedGemmKernel"),
    "GroupedNvfp4Gemm": (".grouped.runtime", "GroupedNvfp4Gemm"),
    "GroupedWgrad": (".grouped.wgrad", "GroupedWgrad"),
    "MoEDispatch": (".routing.dispatch", "MoEDispatch"),
    "gated_backward_values": (".gated", "gated_backward_values"),
    "gated_postact_fragment": (".gated", "gated_postact_fragment"),
    "grouped_nvfp4_gemm": (".grouped.runtime", "grouped_nvfp4_gemm"),
    "moe_dispatch": (".routing.dispatch", "moe_dispatch"),
    "moe_finalize": (".routing.combine", "moe_finalize"),
    "moe_finalize_bwd": (".routing.combine", "moe_finalize_bwd"),
    "nvfp4_decode_prepare": (".quantize.decode", "nvfp4_decode_prepare"),
    "nvfp4_quantize_decode": (".quantize.decode", "nvfp4_quantize_decode"),
    "nvfp4_quantize_colwise": (".quantize.runtime", "nvfp4_quantize_colwise"),
    "nvfp4_quantize_rowwise": (".quantize.runtime", "nvfp4_quantize_rowwise"),
    "nvfp4_rht_amax": (".quantize.runtime", "nvfp4_rht_amax"),
    "quantize_postact_fragment": (".gated", "quantize_postact_fragment"),
    "rht_matrix": (".quantize.runtime", "rht_matrix"),
    "swiglu_backward_pair": (".gated", "swiglu_backward_pair"),
}


def __getattr__(name):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted((*globals(), *_EXPORTS))
