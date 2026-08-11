"""Training-capable NVFP4 MoE expert kernels for NVIDIA B200."""

from importlib import import_module

__all__ = [
    "GatedBackwardEpilogue",
    "GatedEpilogue",
    "GroupedGemmKernel",
    "GroupedNvfp4Gemm",
    "MoEDispatch",
    "MoEExpertLayer",
    "Qwen3Nvfp4Experts",
    "TensorScale",
    "grouped_nvfp4_gemm",
    "moe_dispatch",
    "moe_finalize",
    "moe_finalize_bwd",
    "nvfp4_quantize_colwise",
    "nvfp4_quantize_rowwise",
]
__version__ = "0.1.0"

_EXPORTS = {
    "GatedBackwardEpilogue": (".kernels.epilogue", "GatedBackwardEpilogue"),
    "GatedEpilogue": (".kernels.epilogue", "GatedEpilogue"),
    "GroupedGemmKernel": (".kernels.gemm_kernel", "GroupedGemmKernel"),
    "GroupedNvfp4Gemm": (".kernels.gemm", "GroupedNvfp4Gemm"),
    "MoEDispatch": (".kernels.dispatch", "MoEDispatch"),
    "MoEExpertLayer": (".layer", "MoEExpertLayer"),
    "Qwen3Nvfp4Experts": (".hf", "Qwen3Nvfp4Experts"),
    "TensorScale": (".recipe", "TensorScale"),
    "grouped_nvfp4_gemm": (".kernels.gemm", "grouped_nvfp4_gemm"),
    "moe_dispatch": (".kernels.dispatch", "moe_dispatch"),
    "moe_finalize": (".kernels.finalize", "moe_finalize"),
    "moe_finalize_bwd": (".kernels.finalize", "moe_finalize_bwd"),
    "nvfp4_quantize_colwise": (".kernels.quantize", "nvfp4_quantize_colwise"),
    "nvfp4_quantize_rowwise": (".kernels.quantize", "nvfp4_quantize_rowwise"),
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
