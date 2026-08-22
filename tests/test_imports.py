"""Import boundaries for LightMoE kernels."""

import inspect
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_exports_remain_available():
    import lightmoe
    from lightmoe.inference import InferenceMoE
    from lightmoe.kernels import nvfp4_decode_prepare
    from lightmoe.kernels.quantize.decode import nvfp4_decode_prepare as decode_prepare
    from lightmoe.kernels.routing.dispatch import MoEDispatch
    from lightmoe.training import MoEExpertLayer

    assert lightmoe.InferenceMoE is InferenceMoE
    assert lightmoe.MoEExpertLayer is MoEExpertLayer
    assert lightmoe.MoEDispatch is MoEDispatch
    assert nvfp4_decode_prepare is decode_prepare
    assert lightmoe.__all__ == ["InferenceMoE", "MoEDispatch", "MoEExpertLayer"]


def test_standalone_scheduler_policy():
    from lightmoe.kernels.grouped.runtime import _resolve_dynamic_schedule

    assert not _resolve_dynamic_schedule(None, 2048)
    assert _resolve_dynamic_schedule(None, 4096)
    assert _resolve_dynamic_schedule(True, 2048)
    assert not _resolve_dynamic_schedule(False, 4096)


def test_inference_tile_policy():
    from lightmoe.inference import _tile_m

    assert _tile_m(8 * 128, 128, 2048, 768) == (128, 128)
    assert _tile_m(256 * 128, 128, 2048, 768) == (256, 128)
    assert _tile_m(256 * 32, 32, 7168, 2048) == (256, 256)


def test_public_constructor_names_are_descriptive():
    from lightmoe import InferenceMoE, MoEDispatch, MoEExpertLayer

    assert tuple(inspect.signature(InferenceMoE).parameters)[:5] == (
        "hidden_size",
        "intermediate_size",
        "num_experts",
        "top_k",
        "max_tokens",
    )
    assert tuple(inspect.signature(MoEDispatch).parameters)[:3] == (
        "num_tokens",
        "num_experts",
        "top_k",
    )
    assert tuple(inspect.signature(MoEExpertLayer).parameters)[:4] == (
        "hidden_size",
        "intermediate_size",
        "num_experts",
        "top_k",
    )
    assert tuple(inspect.signature(InferenceMoE.load_packed_weights).parameters)[1:7] == (
        "gate_up",
        "gate_up_sf",
        "gate_up_scale",
        "down",
        "down_sf",
        "down_scale",
    )


def test_layer_import_is_self_contained():
    code = """
import lightmoe.kernels.grouped.runtime
from lightmoe.kernels.gated import GatedEpilogue
from lightmoe.kernels.grouped.runtime import GroupedNvfp4Gemm
from lightmoe.training import MoEExpertLayer

assert MoEExpertLayer.__name__ == "MoEExpertLayer"
assert GroupedNvfp4Gemm.__name__ == "GroupedNvfp4Gemm"
assert GatedEpilogue("swiglu").activation == "swiglu"
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
