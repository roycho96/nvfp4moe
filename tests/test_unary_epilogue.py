import pytest

from lightmoe.kernels.gated import resolve_gemm_epilogue
from lightmoe.kernels.unary import (
    UnaryBackwardEpilogue,
    UnaryEpilogue,
    unary_backward_values,
    unary_postact_fragment,
    validate_unary_activation,
)


def test_relu2_policy_resolves_compile_mode():
    assert resolve_gemm_epilogue(UnaryEpilogue(save_preact=True), None, None) == (
        "relu2",
        None,
    )
    assert resolve_gemm_epilogue(UnaryBackwardEpilogue(), None, None) == (None, "relu2")


def test_relu2_policy_rejects_unknown_activation():
    assert validate_unary_activation("relu2") == "relu2"
    with pytest.raises(ValueError, match="relu2"):
        UnaryEpilogue("relu")


def test_relu2_helpers_are_jitted():
    assert callable(unary_postact_fragment)
    assert callable(unary_backward_values)
