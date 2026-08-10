"""Import boundaries for native kernels."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_exports_remain_available():
    import nvfp4moe
    from nvfp4moe.layer import MoEExpertLayer
    from nvfp4moe.recipe import TensorScale

    assert nvfp4moe.MoEExpertLayer is MoEExpertLayer
    assert nvfp4moe.TensorScale is TensorScale


def test_native_layer_import_is_self_contained():
    code = """
import nvfp4moe.kernels.grouped_gemm_runtime
from nvfp4moe.layer import MoEExpertLayer

assert MoEExpertLayer.__name__ == "MoEExpertLayer"
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
