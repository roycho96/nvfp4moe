"""Pure-PyTorch NVFP4 quantization and MoE reference functions."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kernels.nvfp4_ref import *  # noqa: F401,F403,E402
from kernels import nvfp4_ref as _ref  # noqa: E402

quantize_nvfp4_lastdim = _ref.quantize_nvfp4_lastdim
quantize_nvfp4_lastdim_te = _ref.quantize_nvfp4_lastdim_te
rht_matrix_ref = _ref.rht_matrix_ref
rht_transform_ref = _ref.rht_transform_ref
dequantize_nvfp4_lastdim = _ref.dequantize_nvfp4_lastdim
pack_sf_blocked = _ref.pack_sf_blocked
unpack_sf_blocked = _ref.unpack_sf_blocked
varlen_sf_tile_offsets = _ref.varlen_sf_tile_offsets
varlen_sf_num_tiles = _ref.varlen_sf_num_tiles
ceil_div = _ref.ceil_div
