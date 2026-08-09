"""Training-capable NVFP4 MoE expert kernels for NVIDIA B200."""

from . import _vendor  # noqa: F401  (register sources before kernel imports)
from .dispatch import MoEDispatch, moe_dispatch
from .finalize import moe_finalize, moe_finalize_bwd
from .gemm import dgrad2_mod, fc1_quant_mod, fc2_weighted_mod, gemm
from .hf import Qwen3Nvfp4Experts
from .layer import MoEExpertLayer
from .quant import nvfp4_quantize_colwise, nvfp4_quantize_rowwise
from .recipe import TensorScale

__all__ = [
    "MoEDispatch",
    "MoEExpertLayer",
    "Qwen3Nvfp4Experts",
    "TensorScale",
    "dgrad2_mod",
    "fc1_quant_mod",
    "fc2_weighted_mod",
    "gemm",
    "moe_dispatch",
    "moe_finalize",
    "moe_finalize_bwd",
    "nvfp4_quantize_colwise",
    "nvfp4_quantize_rowwise",
]
__version__ = "0.1.0"
