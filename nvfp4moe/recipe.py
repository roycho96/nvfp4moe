"""Device-resident per-tensor scale state for NVFP4 quantization."""

import torch

F4_MAX = 6.0
E4M3_MAX = 448.0
_DEN = E4M3_MAX * F4_MAX  # 2688
# Match Transformer Engine's fp32 compile-time inverse.
_INV6 = float(torch.tensor(1.0, dtype=torch.float32)
              / torch.tensor(6.0, dtype=torch.float32))


class TensorScale:
    """Track amax and expose quantization/dequantization scale pairs.

    ``te_rht=True`` uses Transformer Engine's RHT scale convention.
    """

    def __init__(self, device="cuda", te_rht=False):
        self._amax = torch.zeros(1, dtype=torch.float32, device=device)
        self._pair = torch.empty(2, dtype=torch.float32, device=device)
        self._ready = False
        self.te_rht = te_rht

    @torch.no_grad()
    def update(self, t: torch.Tensor):
        """Update amax without a host synchronization."""
        mn, mx = torch.aminmax(t.detach())
        self._amax.copy_(torch.maximum(mn.neg_(), mx).reshape(1).float())
        self._refresh()

    @torch.no_grad()
    def set_amax(self, amax: torch.Tensor):
        """Use an amax computed by another device kernel."""
        self._amax.copy_(amax.reshape(1))
        self._refresh()

    @torch.no_grad()
    def _refresh(self):
        if self.te_rht:
            # Transformer Engine's fp32 scale calculation.
            f32max = torch.finfo(torch.float32).max
            ges = torch.where(
                self._amax > 0,
                torch.clamp(_DEN / self._amax, max=f32max),
                torch.ones_like(self._amax),
            )
            self._pair[0:1].copy_(ges * _INV6)
            self._pair[1:2].copy_(1.0 / ges)
        else:
            p = (self._amax / _DEN).clamp_(min=torch.finfo(torch.float32).tiny)
            self._pair[0:1].copy_(p)
            self._pair[1:2].copy_(1.0 / p)
        self._ready = True

    @property
    def pair(self) -> torch.Tensor:
        """Return ``[pts, 1/pts]`` or the equivalent RHT scale pair."""
        assert self._ready, (
            "TensorScale used before calibration: call update(t) once "
            "(previous-step amax / calibration pass). There is no pts default."
        )
        return self._pair

    @property
    def pts(self) -> torch.Tensor:
        """Return the global dequantization scale."""
        return self.pair[1:2] if self.te_rht else self.pair[0:1]

    @property
    def inv_pts(self) -> torch.Tensor:
        assert not self.te_rht, "inv_pts is a divide-chain (non-TE) notion"
        return self.pair[1:2]
