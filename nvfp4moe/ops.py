"""PyTorch custom operators for standalone NVFP4 grouped GEMM use."""

from __future__ import annotations

import threading

import torch

_RUNTIME_CACHE = {}
_RUNTIME_LOCK = threading.Lock()


def _validate_geometry(
    a: torch.Tensor,
    b: torch.Tensor,
    cu: torch.Tensor,
    output_dtype: torch.dtype,
    activation: str | None,
) -> tuple[int, int, int, int]:
    if a.ndim != 2 or b.ndim != 3:
        raise ValueError("A must be rank 2 and B must be rank 3")
    experts, n, packed_k = b.shape
    if a.shape[1] != packed_k:
        raise ValueError("A and B must have the same packed K dimension")
    if cu.ndim != 1 or cu.shape[0] != experts + 1:
        raise ValueError("cu must contain one offset per expert plus the final offset")
    if output_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("the public grouped GEMM op supports BF16 or FP32 output")
    if activation not in (None, "swiglu", "geglu", "reglu"):
        raise ValueError("activation must be swiglu, geglu, reglu, or None")
    if activation is not None and n % 2:
        raise ValueError("a gated epilogue requires an even output dimension")
    output_n = n // 2 if activation is not None else n
    return experts, n, packed_k * 2, output_n


def _runtime(
    device: torch.device,
    experts: int,
    n: int,
    k: int,
    tile_m: int,
    tile_n: int,
    output_dtype: torch.dtype,
    activation: str | None,
):
    key = (device.index, experts, n, k, tile_m, tile_n, output_dtype, activation)
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is not None:
        return runtime
    with _RUNTIME_LOCK:
        runtime = _RUNTIME_CACHE.get(key)
        if runtime is None:
            from .kernels.gemm import GroupedNvfp4Gemm

            with torch.cuda.device(device):
                runtime = GroupedNvfp4Gemm(
                    experts,
                    n,
                    k,
                    tile_m,
                    tile_n,
                    output_dtype=output_dtype,
                    activation=activation,
                )
            _RUNTIME_CACHE[key] = runtime
    return runtime


@torch.library.custom_op(
    "nvfp4moe::grouped_gemm",
    mutates_args=(),
    device_types="cuda",
)
def _grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    cu: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
    activation: str | None,
    tile_m: int,
    tile_n: int,
) -> torch.Tensor:
    experts, n, k, output_n = _validate_geometry(a, b, cu, output_dtype, activation)
    out = torch.empty((a.shape[0], output_n), dtype=output_dtype, device=a.device)
    _runtime(
        a.device,
        experts,
        n,
        k,
        tile_m,
        tile_n,
        output_dtype,
        activation,
    )(a, b, out, cu, sfa, sfb, alpha)
    return out


@_grouped_gemm.register_fake
def _grouped_gemm_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    cu: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
    activation: str | None,
    tile_m: int,
    tile_n: int,
) -> torch.Tensor:
    del sfa, sfb, alpha, tile_m, tile_n
    _, _, _, output_n = _validate_geometry(a, b, cu, output_dtype, activation)
    return torch.empty((a.shape[0], output_n), dtype=output_dtype, device=a.device)


def grouped_nvfp4_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    cu: torch.Tensor,
    alpha: torch.Tensor,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
    activation: str | None = None,
    tile_m: int = 128,
    tile_n: int = 256,
) -> torch.Tensor:
    """Run a prepacked grouped NVFP4 GEMM through an opaque PyTorch op.

    The op is visible to ``torch.compile`` and supports a dynamic total row
    count. Its packed operands are intentionally non-differentiable; training
    code should use :class:`nvfp4moe.NVFP4ExpertCore`.
    """
    return _grouped_gemm(
        a,
        b,
        sfa,
        sfb,
        cu,
        alpha,
        output_dtype,
        activation,
        tile_m,
        tile_n,
    )


__all__ = ["grouped_nvfp4_gemm"]
