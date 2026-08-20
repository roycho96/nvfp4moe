"""Weight packing shared by training and inference plans."""

import torch

from .kernels.quantize import nvfp4_quantize_rowwise


@torch.no_grad()
def quantize_expert_stack(matrices, scale):
    packed, factors = [], []
    scale = scale.detach().to(dtype=torch.float32).reshape(1)
    scale_pair = torch.cat((scale, scale.reciprocal()))
    for matrix in matrices:
        rows, features = matrix.shape
        q = torch.empty(rows, features // 2, dtype=torch.uint8, device=matrix.device)
        sf_rows = -(-rows // 128)
        sf = torch.empty(
            sf_rows + 1,
            features // 64,
            32,
            4,
            4,
            dtype=torch.float8_e4m3fn,
            device=matrix.device,
        )
        offsets = torch.tensor((0, rows), dtype=torch.int32, device=matrix.device)
        nvfp4_quantize_rowwise(matrix.contiguous(), offsets, scale_pair, q, sf)
        packed.append(q)
        factors.append(sf[:sf_rows])
    return torch.stack(packed).view(torch.float4_e2m1fn_x2), torch.stack(factors)


__all__ = ["quantize_expert_stack"]
