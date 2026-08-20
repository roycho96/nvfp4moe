"""Import boundaries for native kernels."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_exports_remain_available():
    import nvfp4moe
    from nvfp4moe.inference import InferenceMoE
    from nvfp4moe.kernels import nvfp4_decode_prepare
    from nvfp4moe.kernels.decode import nvfp4_decode_prepare as decode_prepare
    from nvfp4moe.kernels.dispatch import MoEDispatch
    from nvfp4moe.layer import MoEExpertLayer

    assert nvfp4moe.InferenceMoE is InferenceMoE
    assert nvfp4moe.MoEExpertLayer is MoEExpertLayer
    assert nvfp4moe.MoEDispatch is MoEDispatch
    assert nvfp4_decode_prepare is decode_prepare
    assert nvfp4moe.__all__ == ["InferenceMoE", "MoEDispatch", "MoEExpertLayer"]


def test_standalone_scheduler_policy():
    from nvfp4moe.kernels.gemm import _resolve_dynamic_schedule

    assert not _resolve_dynamic_schedule(None, 2048)
    assert _resolve_dynamic_schedule(None, 4096)
    assert _resolve_dynamic_schedule(True, 2048)
    assert not _resolve_dynamic_schedule(False, 4096)


def test_inference_tile_policy():
    from nvfp4moe.inference import _tile_m

    assert _tile_m(8 * 128, 128, 2048, 768) == (128, 128)
    assert _tile_m(256 * 128, 128, 2048, 768) == (256, 128)
    assert _tile_m(256 * 32, 32, 7168, 2048) == (256, 256)


def test_native_layer_import_is_self_contained():
    code = """
import nvfp4moe.kernels.gemm
from nvfp4moe.kernels.epilogue import GatedEpilogue
from nvfp4moe.kernels.gemm import GroupedNvfp4Gemm
from nvfp4moe.layer import MoEExpertLayer

assert MoEExpertLayer.__name__ == "MoEExpertLayer"
assert GroupedNvfp4Gemm.__name__ == "GroupedNvfp4Gemm"
assert GatedEpilogue("swiglu").activation == "swiglu"
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
