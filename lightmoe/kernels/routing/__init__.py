"""MoE dispatch and combine kernels."""

from importlib import import_module

__all__ = [
    "MoEDispatch",
    "moe_dispatch",
    "moe_finalize",
    "moe_finalize_bwd",
]

_EXPORTS = {
    "MoEDispatch": (".dispatch", "MoEDispatch"),
    "moe_dispatch": (".dispatch", "moe_dispatch"),
    "moe_finalize": (".combine", "moe_finalize"),
    "moe_finalize_bwd": (".combine", "moe_finalize_bwd"),
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
