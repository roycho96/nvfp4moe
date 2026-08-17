"""Training-capable NVFP4 MoE expert kernels for NVIDIA B200."""

from importlib import import_module

__all__ = [
    "MoEDispatch",
    "MoEExpertLayer",
]
__version__ = "0.1.0"

_EXPORTS = {
    "MoEDispatch": (".kernels.dispatch", "MoEDispatch"),
    "MoEExpertLayer": (".layer", "MoEExpertLayer"),
}


def __getattr__(name):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted((*globals(), *_EXPORTS))
