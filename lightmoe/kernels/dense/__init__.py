"""Dense NVFP4 GEMM kernels."""

from importlib import import_module

__all__ = ["DenseNvfp4Gemm", "Sm100BlockScaledPersistentDenseGemmKernel"]

_EXPORTS = {
    "DenseNvfp4Gemm": (".runtime", "DenseNvfp4Gemm"),
    "Sm100BlockScaledPersistentDenseGemmKernel": (
        ".kernel",
        "Sm100BlockScaledPersistentDenseGemmKernel",
    ),
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
