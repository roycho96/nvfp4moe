"""Public NVFP4 GEMM plans and packing helpers."""

import torch

from ._quantization import _DEN
from .kernels.dense.runtime import DenseNvfp4Gemm
from .kernels.grouped.runtime import GroupedNvfp4Gemm
from .kernels.quantize.runtime import nvfp4_quantize_rowwise

DenseGemm = DenseNvfp4Gemm
GroupedGemm = GroupedNvfp4Gemm

_CU_CACHE: dict[tuple[str, int | None, int], torch.Tensor] = {}


def _single_group(device: torch.device, rows: int) -> torch.Tensor:
    key = (device.type, device.index, rows)
    cu = _CU_CACHE.get(key)
    if cu is None:
        cu = torch.tensor((0, rows), dtype=torch.int32, device=device)
        _CU_CACHE[key] = cu
    return cu


def _tensor_scale(x: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    if scale is None:
        return (x.abs().amax().float() / _DEN).clamp_min(1e-30).reshape(1)
    return scale.detach().to(device=x.device, dtype=torch.float32).reshape(1)


def _check_matrix(x: torch.Tensor, name: str) -> None:
    if not x.is_cuda or x.ndim != 2 or not x.is_contiguous():
        raise ValueError(f"{name} must be a contiguous CUDA matrix")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"{name} must use BF16 or FP32")
    if x.shape[1] % 64:
        raise ValueError(f"{name} K must be aligned to 64")


def _quantize_matrix(
    x: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, features = x.shape
    q = torch.empty(rows, features // 2, dtype=torch.uint8, device=x.device)
    sf_rows = -(-rows // 128)
    sf = torch.empty(
        sf_rows + 1,
        features // 64,
        32,
        4,
        4,
        dtype=torch.float8_e4m3fn,
        device=x.device,
    )
    pair = torch.cat((scale, scale.reciprocal()))
    nvfp4_quantize_rowwise(x, _single_group(x.device, rows), pair, q, sf)
    return q.view(torch.float4_e2m1fn_x2), sf[:sf_rows]


def quantize(
    x: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack one BF16 or FP32 matrix into the LightMoE NVFP4 layout."""
    _check_matrix(x, "x")
    scale = _tensor_scale(x, scale)
    q, sf = _quantize_matrix(x, scale)
    return q, sf, scale


def quantize_grouped(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indptr: torch.Tensor,
    a_scale: torch.Tensor | None = None,
    b_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack expert-major activations and weights for grouped GEMM."""
    _check_matrix(a, "a")
    if not b.is_cuda or b.ndim != 3 or not b.is_contiguous():
        raise ValueError("b must be contiguous CUDA expert weights [E, N, K]")
    if b.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("b must use BF16 or FP32")
    if b.device != a.device or b.shape[2] != a.shape[1]:
        raise ValueError("a and b must share device and K")
    experts, _, k = b.shape
    if m_indptr.dtype != torch.int32 or tuple(m_indptr.shape) != (experts + 1,):
        raise ValueError("m_indptr must be contiguous int32 with E + 1 offsets")
    if not m_indptr.is_cuda or not m_indptr.is_contiguous() or m_indptr.device != a.device:
        raise ValueError("m_indptr must be contiguous and share the input device")

    a_scale = _tensor_scale(a, a_scale)
    b_scale = _tensor_scale(b, b_scale)
    pair_a = torch.cat((a_scale, a_scale.reciprocal()))
    qa_u8 = torch.empty(a.shape[0], k // 2, dtype=torch.uint8, device=a.device)
    sfa = torch.zeros(
        1,
        -(-a.shape[0] // 128) + experts,
        k // 64,
        32,
        4,
        4,
        dtype=torch.float8_e4m3fn,
        device=a.device,
    )
    nvfp4_quantize_rowwise(a, m_indptr, pair_a, qa_u8, sfa)

    qb_parts = []
    sfb_parts = []
    for expert in range(experts):
        qb, sfb = _quantize_matrix(b[expert], b_scale)
        qb_parts.append(qb.view(torch.uint8))
        sfb_parts.append(sfb)
    qa = qa_u8.view(torch.float4_e2m1fn_x2)
    qb = torch.stack(qb_parts).view(torch.float4_e2m1fn_x2)
    sfb = torch.stack(sfb_parts)
    return qa, qb, sfa, sfb, (a_scale * b_scale).reshape(1)


__all__ = ["DenseGemm", "GroupedGemm", "quantize", "quantize_grouped"]
