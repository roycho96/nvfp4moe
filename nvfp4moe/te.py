"""Transformer Engine-compatible entry point for routed expert cores.

This module only adapts the ``inp, m_splits`` calling convention. Transformer
Engine remains responsible for the rest of the model and any token transport.
"""

from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn


def te_splits_to_cu(m_splits: Tensor) -> Tensor:
    """Convert per-expert row counts to device-side int32 cumulative offsets."""
    if not isinstance(m_splits, Tensor):
        raise TypeError("m_splits must be a tensor")
    if m_splits.ndim != 1:
        raise ValueError("m_splits must be a rank-1 tensor")
    if m_splits.dtype not in (torch.int32, torch.int64):
        raise TypeError("m_splits must have int32 or int64 dtype")

    return torch.cat(
        (
            m_splits.new_zeros(1, dtype=torch.int32),
            torch.cumsum(m_splits, dim=0, dtype=torch.int32),
        )
    )


class TransformerEngineExpertAdapter(nn.Module):
    """Expose a functional expert core through TE's grouped-expert interface.

    ``w1`` and ``w3`` use ``[E, I, D]`` and ``w2`` uses ``[E, D, I]``.
    Existing :class:`~torch.nn.Parameter` objects are kept by identity, so an
    external optimizer and the adapter state dict share the same BF16 master
    weights.

    The injected core is called as::

        core(inp, w1, w3, w2, cu, off_pad=off_pad)

    The core also provides ``refresh_weights(w1, w3, w2)`` and
    ``calibrate(inp, cu, off_pad=None)``. The adapter initializes both lazily;
    subsequent ``is_first_microbatch=True`` calls refresh the weight cache.
    This adapter does not import or modify Transformer Engine.
    """

    def __init__(
        self,
        core: Callable[..., Tensor],
        w1: Tensor,
        w3: Tensor,
        w2: Tensor,
    ) -> None:
        super().__init__()
        if not callable(core):
            raise TypeError("core must be callable")
        if not callable(getattr(core, "refresh_weights", None)):
            raise TypeError("core must provide refresh_weights(w1, w3, w2)")
        if not callable(getattr(core, "calibrate", None)):
            raise TypeError("core must provide calibrate(inp, cu, off_pad=None)")
        self._validate_weights(w1, w3, w2)

        self.core = core
        self.w1 = w1 if isinstance(w1, nn.Parameter) else nn.Parameter(w1)
        self.w3 = w3 if isinstance(w3, nn.Parameter) else nn.Parameter(w3)
        self.w2 = w2 if isinstance(w2, nn.Parameter) else nn.Parameter(w2)
        self._weights_ready = False
        self._calibrated = False
        self.register_load_state_dict_post_hook(self._reset_runtime_state)

    @staticmethod
    def _validate_weights(w1: Tensor, w3: Tensor, w2: Tensor) -> None:
        if not all(isinstance(weight, Tensor) for weight in (w1, w3, w2)):
            raise TypeError("w1, w3, and w2 must be tensors")
        if any(weight.dtype != torch.bfloat16 for weight in (w1, w3, w2)):
            raise ValueError("master expert weights must use bfloat16")
        if any(weight.ndim != 3 for weight in (w1, w3, w2)):
            raise ValueError("master expert weights must be rank-3 tensors")
        if w1.device != w3.device or w1.device != w2.device:
            raise ValueError("w1, w3, and w2 must be on the same device")

        experts, intermediate, hidden = w1.shape
        expected_w2 = (experts, hidden, intermediate)
        if tuple(w3.shape) != tuple(w1.shape) or tuple(w2.shape) != expected_w2:
            raise ValueError("w1, w3, and w2 expert shapes do not match")

    @property
    def num_experts(self) -> int:
        """Number of experts in the local weight shard."""
        return self.w1.shape[0]

    def _reset_runtime_state(self, module: nn.Module, incompatible_keys: object) -> None:
        del module, incompatible_keys
        self._weights_ready = False
        self._calibrated = False

    @torch.no_grad()
    def refresh_weights(self) -> None:
        """Refresh the core's NVFP4 cache from the shared BF16 parameters."""
        self.core.refresh_weights(self.w1, self.w3, self.w2)
        self._weights_ready = True

    @torch.no_grad()
    def calibrate(self, inp: Tensor, cu: Tensor, off_pad: Tensor) -> None:
        """Seed the core's activation state from expert-ordered input rows."""
        self.core.calibrate(inp, cu, off_pad=off_pad)
        self._calibrated = True

    def forward(
        self,
        inp: Tensor,
        m_splits: Tensor | Sequence[int],
        is_first_microbatch: bool | None = None,
    ) -> Tensor:
        """Run the expert core for TE-ordered rows and per-expert row counts."""
        if inp.ndim != 2 or inp.shape[1] != self.w1.shape[2]:
            raise ValueError(f"inp must have shape [M, {self.w1.shape[2]}]")
        if inp.dtype != torch.bfloat16:
            raise ValueError("inp must use bfloat16")
        if inp.device != self.w1.device:
            raise ValueError("inp and master weights must share a device")
        if not isinstance(m_splits, Tensor):
            m_splits = torch.tensor(m_splits, dtype=torch.int32, device=inp.device)
        elif m_splits.device != inp.device:
            m_splits = m_splits.to(device=inp.device, non_blocking=True)
        if m_splits.ndim == 1 and m_splits.shape[0] != self.num_experts:
            raise ValueError(f"m_splits must contain {self.num_experts} local expert counts")

        cu = te_splits_to_cu(m_splits)
        counts = m_splits.to(torch.int32)
        off_pad = torch.cumsum(((counts + 127) // 128) * 128, dim=0, dtype=torch.int32)
        if is_first_microbatch is True or not self._weights_ready:
            self.refresh_weights()
        if not self._calibrated:
            self.calibrate(inp, cu, off_pad)
        return self.core(inp, self.w1, self.w3, self.w2, cu, off_pad=off_pad)


TEExpertAdapter = TransformerEngineExpertAdapter


__all__ = ["TEExpertAdapter", "TransformerEngineExpertAdapter", "te_splits_to_cu"]
