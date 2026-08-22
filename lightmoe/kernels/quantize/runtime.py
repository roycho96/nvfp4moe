"""Host runtime for the NVFP4 quantization kernels."""

import torch
from torch import Tensor

from .._common import torch2cute_dtype_map
from .kernel import NVFP4QuantKernel

# TE get_wgrad_sign_vector (hard-coded random signs; bitmask 0xD7E8)
_RHT_SIGN_VECTOR = (1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1)
_rht_m_cache = {}


def rht_matrix(device="cuda") -> Tensor:
    """(16, 16) bf16 S @ H16 * 0.25 - byte-identical to TE get_rht_matrix
    (with_random_sign_mask=True): Sylvester H16, hard-coded sign vector,
    scale 1/sqrt(16); every entry is +-0.25, exact in bf16."""
    dev = torch.device(device)
    t = _rht_m_cache.get(dev)
    if t is None:
        s = torch.tensor(_RHT_SIGN_VECTOR, dtype=torch.float32, device=dev)
        h = torch.ones(1, 1, device=dev)
        while h.shape[0] < 16:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        t = ((s * torch.eye(16, device=dev)) @ (h * 0.25)).to(torch.bfloat16).contiguous()
        _rht_m_cache[dev] = t
    return t


def nvfp4_rht_amax(
    z: Tensor,
    cu: Tensor,
    partials: Tensor,
    gather_idx: Tensor | None = None,
    padded_offsets: Tensor | None = None,
):
    """TE with_post_rht_amax pre-pass: ONE read of z (rows, F) bf16 emits
    per-CTA partial (raw amax, post-RHT columnwise amax) pairs into partials
    ((>= num_seg_tiles * F/128), 2) f32, which the CALLER must zero-fill
    beforehand and reduce with torch.amax(partials, 0) -> (2,) [row, col].
    max is exact, so the two-level reduction is bitwise deterministic. The
    col amax is the amax of |bf16 RHT transform| over the zero-padded expert
    segments - exactly the values the rht=True colwise quantizer will see."""
    F = z.shape[1]
    assert F % 128 == 0 and z.stride(-1) == 1
    assert z.dtype == torch.bfloat16, "RHT is bf16-only (TE constraint)"
    E = cu.numel() - 1
    _check_padded_offsets(padded_offsets, cu, E)
    M = int(gather_idx.numel()) if gather_idx is not None else z.shape[0]
    n_tiles = -(-M // 128) + E
    assert partials.dtype == torch.float32 and partials.shape[0] >= n_tiles * (F // 128)
    NVFP4QuantKernel.compile(
        torch2cute_dtype_map[z.dtype],
        "amax",
        gather_idx is not None,
        "rn",
        "blocked",
        False,
        False,
        False,
        padded_offsets is not None,
        E,
    )(
        z,
        gather_idx,
        cu,
        padded_offsets,
        partials,
        None,
        rht_matrix(z.device),
        None,
        None,
        None,
        None,
        None,
        n_tiles,
        F,
        0,
    )


def _prep(pts) -> Tensor:
    """Return device-resident FP32 ``[pts, 1 / pts]`` without a host sync."""
    if torch.is_tensor(pts):
        if pts.numel() == 2:
            return pts  # prepared [pts, 1/pts] pair (recipe path)
        p32 = pts.detach().to(device="cuda", dtype=torch.float32).reshape(1)
    else:
        assert pts > 0, "per_tensor_scale must be positive (no pts=1 default)"
        p32 = torch.tensor([float(pts)], dtype=torch.float32, device="cuda")
    return torch.cat([p32, 1.0 / p32])


def _check_padded_offsets(padded_offsets: Tensor | None, cu: Tensor, experts: int):
    if padded_offsets is None:
        return
    if (
        padded_offsets.dtype != torch.int32
        or padded_offsets.numel() != experts
        or not padded_offsets.is_cuda
        or not padded_offsets.is_contiguous()
        or padded_offsets.device != cu.device
    ):
        raise ValueError("padded_offsets must be contiguous CUDA int32 with one value per expert")


def nvfp4_quantize_rowwise(
    z: Tensor,
    cu: Tensor,
    pts,
    q_out: Tensor,
    sf_out: Tensor,
    gather_idx: Tensor | None = None,
    rounding: str = "rn",
    seed: int = 0,
    sf_layout: str = "blocked",
    padded_offsets: Tensor | None = None,
    te_math: bool = False,
):
    """Quantize expert-ordered rows, optionally gathering source rows from ``z``."""
    F = z.shape[1]
    assert F % 32 == 0 and z.stride(-1) == 1  # 16B store chunks
    assert sf_layout == "blocked" or F % 256 == 0  # linear: full-tile 8B rows
    E = cu.numel() - 1
    _check_padded_offsets(padded_offsets, cu, E)
    M = int(gather_idx.numel()) if gather_idx is not None else z.shape[0]
    pts2 = _prep(pts)
    n_tiles = -(-M // 128) + E
    NVFP4QuantKernel.compile(
        torch2cute_dtype_map[z.dtype],
        "row",
        gather_idx is not None,
        rounding,
        sf_layout,
        False,
        False,
        te_math,
        padded_offsets is not None,
        E,
    )(
        z,
        gather_idx,
        cu,
        padded_offsets,
        q_out.view(torch.uint8),
        sf_out.view(torch.uint8).view(-1),
        None,
        pts2,
        None,
        None,
        None,
        None,
        n_tiles,
        F,
        int(seed) & 0x7FFFFFFF,
    )


def nvfp4_quantize_colwise(
    z: Tensor,
    cu: Tensor,
    pts,
    q_out: Tensor,
    sf_out: Tensor,
    gather_idx: Tensor | None = None,
    rounding: str = "rn",
    seed: int = 0,
    rht: bool = False,
    amax_out: Tensor | None = None,
    padded_offsets: Tensor | None = None,
):
    """Colwise (token-axis) quantize for wgrad operands. z (rows, F); with
    gather_idx (M,), rows are read as z[gather_idx[...]] (the X variant - the
    one unavoidable high-precision gather read). q_out (F, mp_tot/2) uint8;
    sf_out e4m3 (flat view): per-expert independently-blocked SF segments
    concatenated - expert base F*off[e]/16 bytes, inside (F/128, K_e/64,
    32, 4, 4) atoms, off = cumsum(ceil(len/128))*128 (= the cudnn-frontend
    grouped-wgrad SF contract; each chunk is also a contiguous blocked SF
    tensor for per-expert GEMMs). pts REQUIRED. rht=True applies the TE
    16-token-group hadamard transform before block scaling (module
    docstring); pts must then come from the POST-RHT amax
    (nvfp4_rht_amax) - TE with_post_rht_amax semantics.

    amax_out ((>= n_tiles * F/128,) f32): emit one partial per CTA - the CTA
    max of the group amaxes this call quantized with (post-RHT when rht=True;
    bitwise the value the nvfp4_rht_amax pre-pass computes on the same
    tensor). Every launched CTA stores (empty tiles store 0), so the caller
    reduces amax_out[:n_tiles * F/128] with torch.amax, NO zero-fill needed.
    This is the delayed-col-amax source: reduce feeds the NEXT step's pts."""
    F = z.shape[1]
    assert F % 128 == 0 and z.stride(-1) == 1
    assert not rht or z.dtype == torch.bfloat16, "RHT is bf16-only"
    E = cu.numel() - 1
    _check_padded_offsets(padded_offsets, cu, E)
    M = int(gather_idx.numel()) if gather_idx is not None else z.shape[0]
    pts2 = _prep(pts)
    n_tiles = -(-M // 128) + E
    if amax_out is not None:
        assert amax_out.dtype == torch.float32 and amax_out.numel() >= n_tiles * (F // 128)
    NVFP4QuantKernel.compile(
        torch2cute_dtype_map[z.dtype],
        "col",
        gather_idx is not None,
        rounding,
        "blocked",
        rht,
        amax_out is not None,
        False,
        padded_offsets is not None,
        E,
    )(
        z,
        gather_idx,
        cu,
        padded_offsets,
        q_out.view(torch.uint8),
        sf_out.view(torch.uint8).view(-1),
        rht_matrix(z.device) if rht else None,
        pts2,
        amax_out,
        None,
        None,
        None,
        n_tiles,
        F,
        int(seed) & 0x7FFFFFFF,
    )


def nvfp4_quantize_row_colwise(
    z: Tensor,
    cu: Tensor,
    row_pts,
    col_pts,
    row_q: Tensor,
    row_sf: Tensor,
    col_q: Tensor,
    col_sf: Tensor,
    gather_idx: Tensor | None = None,
    rounding: str = "rn",
    seed: int = 0,
    amax_out: Tensor | None = None,
    padded_offsets: Tensor | None = None,
):
    """Quantize rowwise and post-RHT columnwise from one staged BF16 tile."""
    F = z.shape[1]
    assert F % 128 == 0 and z.stride(-1) == 1
    assert z.dtype == torch.bfloat16
    E = cu.numel() - 1
    _check_padded_offsets(padded_offsets, cu, E)
    M = int(gather_idx.numel()) if gather_idx is not None else z.shape[0]
    row_pts2 = _prep(row_pts)
    col_pts2 = _prep(col_pts)
    n_tiles = -(-M // 128) + E
    if amax_out is not None:
        assert amax_out.dtype == torch.float32 and amax_out.numel() >= n_tiles * (F // 128)
    NVFP4QuantKernel.compile(
        torch2cute_dtype_map[z.dtype],
        "dual",
        gather_idx is not None,
        rounding,
        "blocked",
        True,
        amax_out is not None,
        False,
        padded_offsets is not None,
        E,
    )(
        z,
        gather_idx,
        cu,
        padded_offsets,
        col_q.view(torch.uint8),
        col_sf.view(torch.uint8).view(-1),
        rht_matrix(z.device),
        col_pts2,
        amax_out,
        row_q.view(torch.uint8),
        row_sf.view(torch.uint8).view(-1),
        row_pts2,
        n_tiles,
        F,
        int(seed) & 0x7FFFFFFF,
    )
