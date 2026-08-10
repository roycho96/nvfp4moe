"""CuTe DSL kernels used by the expert layer."""

from importlib import import_module

__all__ = [
    "GroupedWgrad",
    "MoEDispatch",
    "moe_dispatch",
    "moe_finalize",
    "moe_finalize_bwd",
    "nvfp4_quantize_colwise",
    "nvfp4_quantize_rowwise",
    "nvfp4_rht_amax",
    "rht_matrix",
]

_EXPORTS = {
    "GroupedWgrad": (".wgrad", "GroupedWgrad"),
    "MoEDispatch": (".dispatch", "MoEDispatch"),
    "moe_dispatch": (".dispatch", "moe_dispatch"),
    "moe_finalize": (".finalize", "moe_finalize"),
    "moe_finalize_bwd": (".finalize", "moe_finalize_bwd"),
    "nvfp4_quantize_colwise": (".quantize", "nvfp4_quantize_colwise"),
    "nvfp4_quantize_rowwise": (".quantize", "nvfp4_quantize_rowwise"),
    "nvfp4_rht_amax": (".quantize", "nvfp4_rht_amax"),
    "rht_matrix": (".quantize", "rht_matrix"),
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
