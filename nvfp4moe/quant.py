"""Rowwise and colwise NVFP4 quantization for variable expert lengths.

Colwise quantization supports Transformer Engine-compatible randomized
Hadamard transforms and deterministic stochastic rounding.
"""

from . import _vendor  # noqa: F401
from quack.nvfp4_quant import (  # noqa: E402
    nvfp4_quantize_colwise,
    nvfp4_quantize_rowwise,
    nvfp4_rht_amax,
    rht_matrix,
)

__all__ = ["nvfp4_quantize_rowwise", "nvfp4_quantize_colwise",
           "nvfp4_rht_amax", "rht_matrix"]
