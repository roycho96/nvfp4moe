"""Training-capable NVFP4 MoE expert kernels for NVIDIA B200."""

from . import _vendor  # noqa: F401  (register sources before kernel imports)
from .recipe import TensorScale
from .quant import nvfp4_quantize_colwise, nvfp4_quantize_rowwise
from .finalize import moe_finalize, moe_finalize_bwd
from .gemm import dgrad2_mod, fc1_quant_mod, fc2_weighted_mod, gemm
from .layer import MoEExpertLayer
from .dispatch import MoEDispatch, moe_dispatch

__all__ = [
    "MoEExpertLayer",
    "MoEDispatch",
    "moe_dispatch",
    "TensorScale",
    "nvfp4_quantize_rowwise",
    "nvfp4_quantize_colwise",
    "moe_finalize",
    "moe_finalize_bwd",
    "fc1_quant_mod",
    "fc2_weighted_mod",
    "dgrad2_mod",
    "gemm",
]
__version__ = "0.1.0"
