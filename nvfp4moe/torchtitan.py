"""Optional TorchTitan grouped-expert integration.

The adapter keeps TorchTitan's router, dispatcher, parameters, and DTensor
sharding in place.  Only the expert computation is replaced.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any, Protocol
from weakref import WeakSet

import torch
from torch import nn


class ExpertCore(Protocol):
    """Functional routed-expert core used by the adapter."""

    def __call__(
        self,
        inp: torch.Tensor,
        w1: torch.Tensor,
        w3: torch.Tensor,
        w2: torch.Tensor,
        cu: torch.Tensor,
        off_pad: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def refresh_weights(
        self,
        w1: torch.Tensor,
        w3: torch.Tensor,
        w2: torch.Tensor,
    ) -> None: ...

    def calibrate(
        self,
        inp: torch.Tensor,
        cu: torch.Tensor,
        off_pad: torch.Tensor | None = None,
    ) -> None: ...


ExpertCoreFactory = Callable[[nn.Module], ExpertCore]

_CORE = "_nvfp4moe_core"
_CORE_FACTORY = "_nvfp4moe_core_factory"
_CONVERTER = "_nvfp4moe_converter"
_WEIGHTS_READY = "_nvfp4moe_weights_ready"
_CALIBRATED = "_nvfp4moe_calibrated"
_ADAPTER_CLASSES: dict[type[nn.Module], type[nn.Module]] = {}


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    """Return a DTensor's local shard without importing DTensor eagerly."""
    to_local = getattr(value, "to_local", None)
    return to_local() if callable(to_local) else value


def _validate_weights(
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[int, int, int]:
    if any(weight.dtype != torch.bfloat16 for weight in (w1, w3, w2)):
        raise ValueError("TorchTitan master expert weights must use bfloat16")
    if w1.device != w3.device or w1.device != w2.device:
        raise ValueError("TorchTitan expert weights must be on the same device")
    if any(weight.ndim != 3 for weight in (w1, w3, w2)):
        raise ValueError("TorchTitan expert weights must be rank-3 tensors")
    experts, intermediate, hidden = w1.shape
    if w3.shape != w1.shape or w2.shape != (experts, hidden, intermediate):
        raise ValueError(
            "expected w1/w3 [E, I, D] and w2 [E, D, I], got "
            f"{tuple(w1.shape)}, {tuple(w3.shape)}, and {tuple(w2.shape)}"
        )
    return experts, intermediate, hidden


def _validate_core(core: Any) -> ExpertCore:
    if not callable(core):
        raise TypeError("expert core factory must return a callable")
    for method in ("refresh_weights", "calibrate"):
        if not callable(getattr(core, method, None)):
            raise TypeError(f"expert core must define {method}()")
    return core


def _core_for(module: nn.Module) -> ExpertCore:
    core = module.__dict__.get(_CORE)
    if core is None:
        factory = module.__dict__[_CORE_FACTORY]
        core = _validate_core(factory(module))
        # Bypass nn.Module registration: the core owns no model parameters.
        module.__dict__[_CORE] = core
    return core


def _weights_for(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if all(hasattr(module, name) for name in ("w1_EFD", "w3_EFD", "w2_EDF")):
        names = ("w1_EFD", "w3_EFD", "w2_EDF")
    elif all(hasattr(module, name) for name in ("w1", "w3", "w2")):
        names = ("w1", "w3", "w2")
    else:
        raise TypeError("GroupedExperts must expose w1_EFD/w3_EFD/w2_EDF parameters")
    return (
        _local_tensor(getattr(module, names[0])),
        _local_tensor(getattr(module, names[1])),
        _local_tensor(getattr(module, names[2])),
    )


def default_core_factory(module: nn.Module) -> ExpertCore:
    """Build the native routed core from a local TorchTitan expert shard."""
    w1, w3, w2 = _weights_for(module)
    experts, intermediate, hidden = _validate_weights(w1, w3, w2)
    from .layer import NVFP4ExpertCore

    with torch.cuda.device(w1.device):
        return NVFP4ExpertCore(hidden, intermediate, experts)


def _offsets(counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    counts_i32 = counts.to(dtype=torch.int32)
    cu = torch.cat((counts_i32.new_zeros(1), counts_i32.cumsum(0)))
    padded = torch.div(counts_i32 + 127, 128, rounding_mode="floor") * 128
    return cu, padded.cumsum(0)


def _refresh(module: nn.Module, core: ExpertCore | None = None) -> None:
    if core is None:
        core = _core_for(module)
    w1, w3, w2 = _weights_for(module)
    _validate_weights(w1, w3, w2)
    if any(weight.device.type == "meta" for weight in (w1, w3, w2)):
        module.__dict__[_WEIGHTS_READY] = False
        return
    with torch.no_grad():
        core.refresh_weights(w1, w3, w2)
    module.__dict__[_WEIGHTS_READY] = True


def _expert_forward(
    self: nn.Module,
    inp: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    if inp.ndim != 2:
        raise ValueError(f"expected expert input [M, D], got {tuple(inp.shape)}")
    if inp.dtype != torch.bfloat16:
        raise ValueError("expert input must use bfloat16")
    if num_tokens_per_expert.ndim != 1:
        raise ValueError("num_tokens_per_expert must be rank 1")
    if num_tokens_per_expert.dtype not in (torch.int32, torch.int64):
        raise TypeError("num_tokens_per_expert must use int32 or int64")
    if num_tokens_per_expert.device != inp.device:
        raise ValueError("expert input and token counts must be on the same device")

    w1, w3, w2 = _weights_for(self)
    experts, _, hidden = _validate_weights(w1, w3, w2)
    if w1.device != inp.device:
        raise ValueError("expert input and local weight shards must be on the same device")
    if num_tokens_per_expert.shape[0] != experts:
        raise ValueError(
            f"expected {experts} local expert counts, got {num_tokens_per_expert.shape[0]}"
        )
    if inp.shape[1] != hidden:
        raise ValueError(f"expected input hidden size {hidden}, got {inp.shape[1]}")

    core = _core_for(self)
    if not self.__dict__[_WEIGHTS_READY]:
        _refresh(self, core)
    cu, off_pad = _offsets(num_tokens_per_expert)
    if not self.__dict__[_CALIBRATED]:
        with torch.no_grad():
            core.calibrate(inp.detach(), cu, off_pad=off_pad)
        self.__dict__[_CALIBRATED] = True
    out = core(inp, w1, w3, w2, cu, off_pad=off_pad)
    if not isinstance(out, torch.Tensor):
        raise TypeError("expert core must return a tensor")
    if out.ndim != 2 or out.shape != inp.shape:
        raise ValueError(f"expert core returned {tuple(out.shape)}; expected {tuple(inp.shape)}")
    if out.dtype != inp.dtype or out.device != inp.device:
        raise ValueError("expert core output must match the input dtype and device")
    return out


def _adapter_class(grouped_experts_type: type[nn.Module]) -> type[nn.Module]:
    adapter = _ADAPTER_CLASSES.get(grouped_experts_type)
    if adapter is None:
        adapter = type(
            f"Nvfp4{grouped_experts_type.__name__}",
            (grouped_experts_type,),
            {
                "__module__": __name__,
                "__doc__": "TorchTitan grouped experts backed by an NVFP4 expert core.",
                "forward": _expert_forward,
            },
        )
        _ADAPTER_CLASSES[grouped_experts_type] = adapter
    return adapter


def _resolve_grouped_experts_type() -> type[nn.Module]:
    try:
        from torchtitan.models.common.moe import GroupedExperts
    except ImportError as exc:
        raise ImportError(
            "TorchTitan is required for conversion; importing nvfp4moe.torchtitan "
            "does not require it"
        ) from exc
    return GroupedExperts


@dataclass(kw_only=True, slots=True)
class TorchTitanExpertsConfig:
    """Configuration accepted by TorchTitan's model-converter container."""

    fqns: list[str] = field(default_factory=list)
    core: ExpertCore | None = field(default=None, repr=False)
    core_factory: ExpertCoreFactory | None = field(default=None, repr=False)
    strict: bool = True

    def build(
        self,
        *,
        parallel_dims: Any = None,
        model_compile_enabled: bool = False,
    ) -> TorchTitanExpertsConverter:
        del parallel_dims, model_compile_enabled
        return TorchTitanExpertsConverter(self)


class TorchTitanExpertsConverter:
    """Replace TorchTitan ``GroupedExperts.forward`` without replacing parameters."""

    Config = TorchTitanExpertsConfig

    def __init__(
        self,
        config: TorchTitanExpertsConfig,
        *,
        grouped_experts_type: type[nn.Module] | None = None,
    ) -> None:
        if config.core is not None and config.core_factory is not None:
            raise ValueError("set at most one of core or core_factory")
        if config.core is not None:
            _validate_core(config.core)
            self._factory: ExpertCoreFactory = lambda _module: config.core  # type: ignore[return-value]
        elif config.core_factory is not None:
            if not callable(config.core_factory):
                raise TypeError("core_factory must be callable")
            self._factory = config.core_factory
        else:
            self._factory = default_core_factory
        self.config = config
        self._grouped_experts_type = grouped_experts_type
        self._converted: WeakSet[nn.Module] = WeakSet()

    def _selected(self, fqn: str) -> bool:
        return not self.config.fqns or any(
            fnmatchcase(fqn, pattern) for pattern in self.config.fqns
        )

    def convert(self, model: nn.Module) -> None:
        grouped_experts_type = self._grouped_experts_type or _resolve_grouped_experts_type()
        selected = [(fqn, module) for fqn, module in model.named_modules() if self._selected(fqn)]
        wrong = [fqn for fqn, module in selected if not isinstance(module, grouped_experts_type)]
        if self.config.fqns and wrong:
            names = ", ".join(name or "<root>" for name in wrong)
            raise TypeError(f"selected modules are not TorchTitan GroupedExperts: {names}")
        targets = [
            (fqn, module) for fqn, module in selected if isinstance(module, grouped_experts_type)
        ]
        if self.config.strict and not targets:
            patterns = ", ".join(self.config.fqns) or "<all GroupedExperts>"
            raise ValueError(f"no TorchTitan GroupedExperts matched {patterns}")
        if self.config.core is not None and len(targets) > 1:
            raise ValueError("a shared core instance can only convert one GroupedExperts module")

        adapter = _adapter_class(grouped_experts_type)
        for _fqn, module in targets:
            owner = module.__dict__.get(_CONVERTER)
            if owner is self:
                self._converted.add(module)
                continue
            if owner is not None:
                raise RuntimeError("GroupedExperts module was already converted")
            _validate_weights(*_weights_for(module))
            module.__dict__[_CORE_FACTORY] = self._factory
            module.__dict__[_CORE] = None
            module.__dict__[_CONVERTER] = self
            module.__dict__[_WEIGHTS_READY] = False
            module.__dict__[_CALIBRATED] = False
            module.__class__ = adapter
            self._converted.add(module)

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]) -> None:
        """Refresh resident NVFP4 weights after an optimizer step."""
        roots = [model] if isinstance(model, nn.Module) else model
        live = {module for root in roots for module in root.modules()}
        for module in tuple(self._converted):
            if module not in live:
                continue
            core = module.__dict__.get(_CORE)
            if core is None:
                module.__dict__[_WEIGHTS_READY] = False
            else:
                _refresh(module, core)


__all__ = [
    "ExpertCore",
    "ExpertCoreFactory",
    "TorchTitanExpertsConfig",
    "TorchTitanExpertsConverter",
    "default_core_factory",
]
