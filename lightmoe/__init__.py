"""CuTe DSL NVFP4 GEMM and MoE kernels for NVIDIA Blackwell SM100."""

from importlib import import_module

__all__ = [
    "InferenceMoE",
    "MoEDispatch",
    "MoEExpertLayer",
]
__version__ = "0.1.0"

_EXPORTS = {
    "InferenceMoE": (".inference", "InferenceMoE"),
    "MoEDispatch": (".routing", "MoEDispatch"),
    "MoEExpertLayer": (".training", "MoEExpertLayer"),
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
