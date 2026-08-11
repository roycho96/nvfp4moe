"""Tests for the optional TorchTitan integration."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nvfp4moe.torchtitan import TorchTitanExpertsConfig, TorchTitanExpertsConverter

ROOT = Path(__file__).resolve().parents[1]


class FakeGroupedExperts(nn.Module):
    def __init__(self, experts=3, hidden=4, intermediate=6):
        super().__init__()
        self.w1_EFD = nn.Parameter(torch.randn(experts, intermediate, hidden, dtype=torch.bfloat16))
        self.w2_EDF = nn.Parameter(torch.randn(experts, hidden, intermediate, dtype=torch.bfloat16))
        self.w3_EFD = nn.Parameter(torch.randn(experts, intermediate, hidden, dtype=torch.bfloat16))
        self.use_grouped_mm = True

    def forward(self, _inp, _counts):
        raise AssertionError("original expert forward was called")


class ReferenceCore:
    def __init__(self):
        self.refreshes = []
        self.calibrations = []
        self.calls = []

    def refresh_weights(self, w1, w3, w2):
        self.refreshes.append((w1, w3, w2))

    def calibrate(self, inp, cu, off_pad=None):
        self.calibrations.append((inp, cu.clone(), off_pad.clone()))

    def __call__(self, inp, w1, w3, w2, cu, off_pad=None):
        self.calls.append((w1, w3, w2, cu.clone(), off_pad.clone()))
        outputs = []
        for expert in range(w1.shape[0]):
            lo, hi = int(cu[expert]), int(cu[expert + 1])
            x = inp[lo:hi]
            hidden = F.silu(x @ w1[expert].T) * (x @ w3[expert].T)
            outputs.append(hidden @ w2[expert].T)
        return torch.cat(outputs) if outputs else inp.new_empty(inp.shape)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = FakeGroupedExperts()
        self.dense = nn.Linear(4, 4)


def make_converter(core=None, **config_kwargs):
    core = ReferenceCore() if core is None else core
    config = TorchTitanExpertsConfig(core=core, **config_kwargs)
    converter = TorchTitanExpertsConverter(config, grouped_experts_type=FakeGroupedExperts)
    return converter, core


def test_module_import_does_not_require_torchtitan():
    code = """
import sys
assert "torchtitan" not in sys.modules
import nvfp4moe.torchtitan
assert "torchtitan" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_converter_preserves_parameters_and_runs_functional_core():
    torch.manual_seed(3)
    model = Model()
    parameters = {name: parameter for name, parameter in model.named_parameters()}
    state_keys = tuple(model.state_dict())
    converter, core = make_converter()

    converter.convert(model)

    assert isinstance(model.experts, FakeGroupedExperts)
    assert tuple(model.state_dict()) == state_keys
    assert all(dict(model.named_parameters())[name] is value for name, value in parameters.items())

    inp = torch.randn(3, 4, dtype=torch.bfloat16, requires_grad=True)
    counts = torch.tensor([2, 0, 1], dtype=torch.int64)
    out = model.experts(inp, counts)
    out.square().sum().backward()

    assert out.shape == inp.shape
    assert inp.grad is not None
    assert all(parameter.grad is not None for parameter in model.experts.parameters())
    assert len(core.refreshes) == 1
    assert len(core.calibrations) == 1
    assert len(core.calls) == 1
    assert torch.equal(core.calls[0][3], torch.tensor([0, 2, 2, 3], dtype=torch.int32))
    assert torch.equal(core.calls[0][4], torch.tensor([128, 128, 256], dtype=torch.int32))
    assert core.calls[0][0] is model.experts.w1_EFD
    assert core.calls[0][1] is model.experts.w3_EFD
    assert core.calls[0][2] is model.experts.w2_EDF

    model.experts(inp.detach(), counts)
    assert len(core.refreshes) == 1
    assert len(core.calibrations) == 1

    converter.post_optimizer_hook(model)
    assert len(core.refreshes) == 2


def test_factory_is_lazy_and_receives_the_original_module():
    model = Model()
    calls = []
    core = ReferenceCore()

    def factory(module):
        calls.append(module)
        return core

    config = TorchTitanExpertsConfig(core_factory=factory, fqns=["experts"])
    converter = TorchTitanExpertsConverter(config, grouped_experts_type=FakeGroupedExperts)
    converter.convert(model)
    assert calls == []

    model.experts(
        torch.randn(2, 4, dtype=torch.bfloat16),
        torch.tensor([1, 1, 0], dtype=torch.int32),
    )
    assert calls == [model.experts]


def test_config_build_matches_torchtitan_converter_signature():
    core = ReferenceCore()
    config = TorchTitanExpertsConfig(core=core, strict=False)
    converter = config.build(parallel_dims=object(), model_compile_enabled=True)
    assert isinstance(converter, TorchTitanExpertsConverter)


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        (torch.ones(3, 1, dtype=torch.int32), "rank 1"),
        (torch.ones(3, dtype=torch.float32), "int32 or int64"),
        (torch.ones(2, dtype=torch.int32), "local expert counts"),
    ],
)
def test_input_validation(counts, message):
    model = Model()
    converter, _ = make_converter()
    converter.convert(model)
    with pytest.raises((TypeError, ValueError), match=message):
        model.experts(torch.randn(3, 4, dtype=torch.bfloat16), counts)


def test_target_validation_and_idempotence():
    model = Model()
    converter, _ = make_converter(fqns=["experts"])
    converter.convert(model)
    converter.convert(model)

    wrong, _ = make_converter(fqns=["dense"])
    with pytest.raises(TypeError, match="not TorchTitan GroupedExperts"):
        wrong.convert(Model())

    missing, _ = make_converter(fqns=["layers.*.experts"])
    with pytest.raises(ValueError, match="no TorchTitan GroupedExperts"):
        missing.convert(Model())
