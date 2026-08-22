import pytest
import torch


def test_public_plans_are_runtime_aliases():
    from lightmoe.gemm import DenseGemm, GroupedGemm
    from lightmoe.kernels.dense.runtime import DenseNvfp4Gemm
    from lightmoe.kernels.grouped.runtime import GroupedNvfp4Gemm

    assert DenseGemm is DenseNvfp4Gemm
    assert GroupedGemm is GroupedNvfp4Gemm
    assert DenseGemm.run is DenseGemm.__call__
    assert GroupedGemm.run is GroupedGemm.__call__


def test_public_gemm_exports_are_small():
    from lightmoe import gemm

    assert gemm.__all__ == ["DenseGemm", "GroupedGemm", "quantize", "quantize_grouped"]


def test_quantize_rejects_cpu_input():
    from lightmoe.gemm import quantize

    with pytest.raises(ValueError, match="contiguous CUDA matrix"):
        quantize(torch.empty(8, 64, dtype=torch.bfloat16))
