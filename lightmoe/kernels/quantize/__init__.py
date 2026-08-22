"""NVFP4 quantization kernels."""

from importlib import import_module

__all__ = [
    "nvfp4_decode_prepare",
    "nvfp4_quantize_colwise",
    "nvfp4_quantize_decode",
    "nvfp4_quantize_row_colwise",
    "nvfp4_quantize_rowwise",
    "nvfp4_rht_amax",
    "rht_matrix",
]

_EXPORTS = {
    "nvfp4_decode_prepare": (".decode", "nvfp4_decode_prepare"),
    "nvfp4_quantize_colwise": (".runtime", "nvfp4_quantize_colwise"),
    "nvfp4_quantize_decode": (".decode", "nvfp4_quantize_decode"),
    "nvfp4_quantize_row_colwise": (".runtime", "nvfp4_quantize_row_colwise"),
    "nvfp4_quantize_rowwise": (".runtime", "nvfp4_quantize_rowwise"),
    "nvfp4_rht_amax": (".runtime", "nvfp4_rht_amax"),
    "rht_matrix": (".runtime", "rht_matrix"),
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
