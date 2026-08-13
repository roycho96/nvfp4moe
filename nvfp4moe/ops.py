"""PyTorch custom operators for standalone NVFP4 grouped GEMM use."""

from __future__ import annotations

import threading

import torch

_RUNTIME_CACHE = {}
_DENSE_RUNTIME_CACHE = {}
_RUNTIME_LOCK = threading.Lock()
_SINGLE_CU_CACHE = {}


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


def _single_cu(device: torch.device, rows: int) -> torch.Tensor:
    key = (device.index, rows)
    cu = _SINGLE_CU_CACHE.get(key)
    if cu is not None:
        return cu
    with _RUNTIME_LOCK:
        cu = _SINGLE_CU_CACHE.get(key)
        if cu is None:
            cu = torch.tensor([0, rows], dtype=torch.int32, device=device)
            _SINGLE_CU_CACHE[key] = cu
    return cu


def _dense_runtime(
    device: torch.device,
    n: int,
    k: int,
    tile_m: int,
    tile_n: int,
    output_dtype: torch.dtype,
):
    key = (device.index, n, k, tile_m, tile_n, output_dtype)
    runtime = _DENSE_RUNTIME_CACHE.get(key)
    if runtime is not None:
        return runtime
    with _RUNTIME_LOCK:
        runtime = _DENSE_RUNTIME_CACHE.get(key)
        if runtime is None:
            from .kernels.dense_gemm import DenseNvfp4Gemm

            with torch.cuda.device(device):
                runtime = DenseNvfp4Gemm(n, k, tile_m, tile_n, output_dtype)
            _DENSE_RUNTIME_CACHE[key] = runtime
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


def _validate_dense_geometry(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    output_dtype: torch.dtype,
) -> tuple[int, int, int]:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be rank-2 packed matrices")
    n, packed_k = b.shape
    if a.shape[1] != packed_k:
        raise ValueError("A and B must have the same packed K dimension")
    if sfa.ndim != 5 or sfb.ndim != 5:
        raise ValueError("SFA and SFB must use rank-5 dense blocked layouts")
    if output_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("the public dense GEMM supports BF16 or FP32 output")
    return n, packed_k * 2, a.shape[0]


@torch.library.custom_op(
    "nvfp4moe::gemm",
    mutates_args=(),
    device_types="cuda",
)
def _gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
    tile_m: int,
    tile_n: int,
) -> torch.Tensor:
    n, k, rows = _validate_dense_geometry(a, b, sfa, sfb, output_dtype)
    out = torch.empty((rows, n), dtype=output_dtype, device=a.device)
    _dense_runtime(a.device, n, k, tile_m, tile_n, output_dtype)(
        a,
        b,
        out,
        sfa,
        sfb,
        alpha,
    )
    return out


@_gemm.register_fake
def _gemm_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
    tile_m: int,
    tile_n: int,
) -> torch.Tensor:
    del alpha, tile_m, tile_n
    n, _, rows = _validate_dense_geometry(a, b, sfa, sfb, output_dtype)
    return torch.empty((rows, n), dtype=output_dtype, device=a.device)


@torch.library.custom_op(
    "nvfp4moe::gemm_out",
    mutates_args=("out",),
    device_types="cuda",
)
def _gemm_out(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    alpha: torch.Tensor,
    out: torch.Tensor,
    tile_m: int,
    tile_n: int,
) -> None:
    n, k, rows = _validate_dense_geometry(a, b, sfa, sfb, out.dtype)
    if tuple(out.shape) != (rows, n):
        raise ValueError(f"out must have shape ({rows}, {n})")
    _dense_runtime(a.device, n, k, tile_m, tile_n, out.dtype)(a, b, out, sfa, sfb, alpha)


@_gemm_out.register_fake
def _gemm_out_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    alpha: torch.Tensor,
    out: torch.Tensor,
    tile_m: int,
    tile_n: int,
) -> None:
    del alpha, tile_m, tile_n
    n, _, rows = _validate_dense_geometry(a, b, sfa, sfb, out.dtype)
    if tuple(out.shape) != (rows, n):
        raise ValueError(f"out must have shape ({rows}, {n})")


def nvfp4_quantize(
    x: torch.Tensor,
    per_tensor_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a contiguous 2D matrix for :func:`nvfp4_gemm`.

    Returns E2M1 data, blocked E4M3 scale factors, and the FP32 per-tensor
    scale. The logical matrix shape is preserved by the first two outputs.
    """
    if not x.is_cuda or x.ndim != 2 or not x.is_contiguous():
        raise ValueError("x must be a contiguous CUDA matrix")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("x must use BF16 or FP32")
    rows, features = x.shape
    if features % 64:
        raise ValueError("the logical K dimension must be aligned to 64")
    from .kernels.quantize import nvfp4_quantize_rowwise
    from .recipe import _DEN

    if per_tensor_scale is None:
        pts = (x.abs().amax().float() / _DEN).clamp_min(1e-30).reshape(1)
    else:
        pts = per_tensor_scale.detach().to(device=x.device, dtype=torch.float32).reshape(1)
    pair = torch.cat((pts, pts.reciprocal()))
    q = torch.empty(rows, features // 2, dtype=torch.uint8, device=x.device)
    sf_rows = -(-rows // 128)
    sf_storage = torch.empty(
        sf_rows + 1,
        features // 64,
        32,
        4,
        4,
        dtype=torch.float8_e4m3fn,
        device=x.device,
    )
    cu = _single_cu(x.device, rows)
    nvfp4_quantize_rowwise(x, cu, pair, q, sf_storage)
    return q.view(torch.float4_e2m1fn_x2), sf_storage[:sf_rows], pts


def nvfp4_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    alpha: torch.Tensor,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
    tile_m: int = 256,
    tile_n: int = 256,
) -> torch.Tensor:
    """Compute ``A @ B.T`` from two prepacked 2D NVFP4 matrices."""
    from torch._subclasses.fake_tensor import is_fake

    if torch.compiler.is_compiling() or is_fake(a):
        return _gemm(a, b, sfa, sfb, alpha, output_dtype, tile_m, tile_n)
    n, k, rows = _validate_dense_geometry(a, b, sfa, sfb, output_dtype)
    out = torch.empty((rows, n), dtype=output_dtype, device=a.device)
    _dense_runtime(a.device, n, k, tile_m, tile_n, output_dtype)(
        a,
        b,
        out,
        sfa,
        sfb,
        alpha,
    )
    return out


def nvfp4_gemm_out(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    alpha: torch.Tensor,
    out: torch.Tensor,
    *,
    tile_m: int = 256,
    tile_n: int = 256,
) -> torch.Tensor:
    """Compute ``A @ B.T`` into a preallocated output tensor."""
    from torch._subclasses.fake_tensor import is_fake

    if torch.compiler.is_compiling() or is_fake(a):
        _gemm_out(a, b, sfa, sfb, alpha, out, tile_m, tile_n)
        return out
    n, k, rows = _validate_dense_geometry(a, b, sfa, sfb, out.dtype)
    if tuple(out.shape) != (rows, n):
        raise ValueError(f"out must have shape ({rows}, {n})")
    _dense_runtime(a.device, n, k, tile_m, tile_n, out.dtype)(a, b, out, sfa, sfb, alpha)
    return out


__all__ = ["grouped_nvfp4_gemm", "nvfp4_gemm", "nvfp4_gemm_out", "nvfp4_quantize"]
