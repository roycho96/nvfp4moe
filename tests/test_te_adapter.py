import pytest
import torch
from torch import nn

from nvfp4moe.te import TEExpertAdapter, TransformerEngineExpertAdapter, te_splits_to_cu


class RecordingCore:
    def __init__(self):
        self.call = None
        self.refresh_calls = []
        self.calibrate_calls = []

    def refresh_weights(self, w1, w3, w2):
        self.refresh_calls.append((w1, w3, w2))

    def calibrate(self, inp, cu, off_pad=None):
        self.calibrate_calls.append((inp, cu, off_pad))

    def __call__(self, inp, w1, w3, w2, cu, off_pad=None):
        self.call = {
            "w1": w1,
            "w3": w3,
            "w2": w2,
            "cu": cu,
            "off_pad": off_pad,
        }
        return inp + w1[0, 0] + w3[0, 0] + w2[0, :, 0]


def make_weights():
    w1 = nn.Parameter(torch.ones(3, 4, 4, dtype=torch.bfloat16))
    w3 = nn.Parameter(torch.ones(3, 4, 4, dtype=torch.bfloat16))
    w2 = nn.Parameter(torch.ones(3, 4, 4, dtype=torch.bfloat16))
    return w1, w3, w2


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_te_splits_to_cu_is_int32_and_stays_on_device(dtype):
    splits = torch.tensor([3, 0, 5], dtype=dtype)

    cu = te_splits_to_cu(splits)

    assert cu.dtype == torch.int32
    assert cu.device == splits.device
    torch.testing.assert_close(cu, torch.tensor([0, 3, 3, 8], dtype=torch.int32))


def test_adapter_forwards_te_arguments_and_preserves_master_parameters():
    core = RecordingCore()
    w1, w3, w2 = make_weights()
    adapter = TransformerEngineExpertAdapter(core, w1, w3, w2)
    inp = torch.zeros(8, 4, dtype=torch.bfloat16, requires_grad=True)

    out = adapter(inp, torch.tensor([3, 0, 5], dtype=torch.int64), True)

    assert adapter.w1 is w1
    assert adapter.w3 is w3
    assert adapter.w2 is w2
    assert adapter.num_experts == 3
    assert core.call["w1"] is w1
    assert core.call["w3"] is w3
    assert core.call["w2"] is w2
    torch.testing.assert_close(core.call["cu"], torch.tensor([0, 3, 3, 8], dtype=torch.int32))
    torch.testing.assert_close(
        core.call["off_pad"], torch.tensor([128, 128, 256], dtype=torch.int32)
    )
    assert core.refresh_calls == [(w1, w3, w2)]
    assert len(core.calibrate_calls) == 1
    assert set(adapter.state_dict()) == {"w1", "w3", "w2"}

    out.float().sum().backward()
    assert inp.grad is not None
    assert w1.grad is not None
    assert w3.grad is not None
    assert w2.grad is not None

    adapter(inp.detach(), torch.tensor([3, 0, 5], dtype=torch.int32), False)
    assert len(core.refresh_calls) == 1
    assert len(core.calibrate_calls) == 1

    adapter(inp.detach(), torch.tensor([3, 0, 5], dtype=torch.int32), True)
    assert len(core.refresh_calls) == 2
    assert len(core.calibrate_calls) == 1


def test_loading_state_dict_invalidates_runtime_caches():
    core = RecordingCore()
    w1, w3, w2 = make_weights()
    adapter = TEExpertAdapter(core, w1, w3, w2)
    inp = torch.zeros(8, 4, dtype=torch.bfloat16)
    splits = torch.tensor([3, 0, 5], dtype=torch.int32)
    adapter(inp, splits)

    adapter.load_state_dict(adapter.state_dict())
    adapter(inp, splits)

    assert len(core.refresh_calls) == 2
    assert len(core.calibrate_calls) == 2


def test_adapter_does_not_require_transformer_engine(monkeypatch):
    import builtins
    import importlib

    real_import = builtins.__import__

    def reject_te(name, *args, **kwargs):
        if name == "transformer_engine" or name.startswith("transformer_engine."):
            raise AssertionError("Transformer Engine was imported")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_te)
    module = importlib.reload(importlib.import_module("nvfp4moe.te"))
    assert module.TransformerEngineExpertAdapter.__name__ == "TransformerEngineExpertAdapter"


@pytest.mark.parametrize(
    ("splits", "error"),
    [
        (torch.ones(3, 1, dtype=torch.int64), ValueError),
        (torch.ones(3, dtype=torch.float32), TypeError),
    ],
)
def test_te_splits_to_cu_rejects_invalid_metadata(splits, error):
    with pytest.raises(error):
        te_splits_to_cu(splits)


def test_adapter_validates_local_expert_count():
    w1, w3, w2 = make_weights()
    adapter = TransformerEngineExpertAdapter(RecordingCore(), w1, w3, w2)

    with pytest.raises(ValueError, match="3 local expert counts"):
        adapter(
            torch.zeros(4, 4, dtype=torch.bfloat16),
            torch.tensor([2, 2], dtype=torch.int64),
        )
