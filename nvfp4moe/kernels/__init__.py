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
    "DenseNvfp4Gemm": (".dense_gemm", "DenseNvfp4Gemm"),
    "GatedBackwardEpilogue": (".epilogue", "GatedBackwardEpilogue"),
    "GatedEpilogue": (".epilogue", "GatedEpilogue"),
    "GroupedGemmKernel": (".gemm_kernel", "GroupedGemmKernel"),
    "GroupedNvfp4Gemm": (".gemm", "GroupedNvfp4Gemm"),
    "GroupedWgrad": (".wgrad", "GroupedWgrad"),
    "MoEDispatch": (".dispatch", "MoEDispatch"),
    "gated_backward_values": (".epilogue", "gated_backward_values"),
    "gated_postact_fragment": (".epilogue", "gated_postact_fragment"),
    "grouped_nvfp4_gemm": (".gemm", "grouped_nvfp4_gemm"),
    "moe_dispatch": (".dispatch", "moe_dispatch"),
    "moe_finalize": (".finalize", "moe_finalize"),
    "moe_finalize_bwd": (".finalize", "moe_finalize_bwd"),
    "nvfp4_decode_prepare": (".decode", "nvfp4_decode_prepare"),
    "nvfp4_quantize_decode": (".decode", "nvfp4_quantize_decode"),
    "nvfp4_quantize_colwise": (".quantize", "nvfp4_quantize_colwise"),
    "nvfp4_quantize_rowwise": (".quantize", "nvfp4_quantize_rowwise"),
    "nvfp4_rht_amax": (".quantize", "nvfp4_rht_amax"),
    "quantize_postact_fragment": (".epilogue", "quantize_postact_fragment"),
    "rht_matrix": (".quantize", "rht_matrix"),
    "swiglu_backward_pair": (".epilogue", "swiglu_backward_pair"),
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
