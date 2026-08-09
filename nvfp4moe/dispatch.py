"""Fused expert-major ordering for preselected router outputs."""

import torch

from . import _vendor  # noqa: F401
from quack.moe_dispatch import B_MAX, moe_dispatch  # noqa: E402


class MoEDispatch:
    """Preallocated fused dispatch: (topi, topv) -> (gi, cu, ps, slots).

    Buffers are allocated once for ``(T, E, k)``. The fused path is device-only
    and uses stable counting-sort order.
    """

    def __init__(self, T: int, E: int, k: int, device="cuda"):
        M = T * k
        self.T, self.E, self.k = T, E, k
        self.gi = torch.empty(M, dtype=torch.int32, device=device)
        self.cu = torch.empty(E + 1, dtype=torch.int32, device=device)
        self.ps = torch.empty(M, dtype=torch.float32, device=device)
        self.slots = torch.empty(M, dtype=torch.int32, device=device)
        self.part = torch.empty(B_MAX * E, dtype=torch.int32, device=device)
        # padded grouped-wgrad offsets, emitted by the same kernel; pass to
        # the layer (off_pad=) to skip its torch fallback chain in bwd
        self.off_pad = torch.empty(E, dtype=torch.int32, device=device)

    def __call__(self, topi: torch.Tensor, topv: torch.Tensor):
        assert topi.shape == (self.T, self.k)
        moe_dispatch(topi, topv.float() if topv.dtype != torch.float32 else topv,
                     self.E, self.gi, self.cu, self.ps, self.slots, self.part,
                     self.off_pad)
        return self.gi, self.cu, self.ps, self.slots

    def differentiable_probs(self, topv: torch.Tensor) -> torch.Tensor:
        """Return expert-major probabilities with an autograd path to ``topv``.

        The preallocated ``ps`` output is outside autograd. Use this method
        when router gradients are required.
        """
        assert topv.shape == (self.T, self.k)
        return torch.zeros_like(self.ps).index_put(
            (self.slots.long(),), topv.reshape(-1).float()
        )


__all__ = ["MoEDispatch", "moe_dispatch"]
