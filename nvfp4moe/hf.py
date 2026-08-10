"""Hugging Face model adapters."""

import torch
from torch import nn

from .kernels.dispatch import MoEDispatch
from .layer import MoEExpertLayer


class Qwen3Nvfp4Experts(nn.Module):
    """Drop-in replacement for Transformers' grouped Qwen3 MoE experts."""

    def __init__(
        self,
        gate_up_proj,
        down_proj,
        tokens,
        topk,
        *,
        sr_seed=1234,
    ):
        super().__init__()
        if gate_up_proj.ndim != 3 or down_proj.ndim != 3:
            raise ValueError("Qwen3 expert weights must be rank-3 tensors")

        experts, twice_intermediate, hidden = gate_up_proj.shape
        intermediate = twice_intermediate // 2
        expected_down = (experts, hidden, intermediate)
        if twice_intermediate % 2 or tuple(down_proj.shape) != expected_down:
            raise ValueError("gate/up and down expert weight shapes do not match")
        if gate_up_proj.device.type != "cuda" or down_proj.device != gate_up_proj.device:
            raise ValueError("Qwen3 expert weights must be on the same CUDA device")
        if gate_up_proj.dtype != torch.bfloat16 or down_proj.dtype != torch.bfloat16:
            raise ValueError("Qwen3 expert weights must be loaded in bfloat16")

        self.tokens = int(tokens)
        self.topk = int(topk)
        self.sr_seed = int(sr_seed)
        self._step = 0
        self._calibrated = False

        device = gate_up_proj.device
        with torch.cuda.device(device):
            self.layer = MoEExpertLayer(
                hidden,
                intermediate,
                experts,
                self.topk,
                rht=True,
                delayed_col_amax=True,
                activation="swiglu",
            ).to(device)
            self.dispatch = MoEDispatch(self.tokens, experts, self.topk, device=device)

        gate, up = gate_up_proj.chunk(2, dim=1)
        with torch.no_grad():
            self.layer.w1[:, 0::2].copy_(gate)
            self.layer.w1[:, 1::2].copy_(up)
            self.layer.w2.copy_(down_proj)
        self.refresh_weights()

    @classmethod
    def from_hf(cls, experts, tokens, topk, **kwargs):
        """Copy weights from a Transformers Qwen3 grouped-expert module."""
        try:
            gate_up_proj = experts.gate_up_proj
            down_proj = experts.down_proj
        except AttributeError as exc:
            raise TypeError("expected grouped Qwen3 experts from Transformers 5.5+") from exc
        return cls(gate_up_proj, down_proj, tokens, topk, **kwargs)

    @torch.no_grad()
    def refresh_weights(self):
        """Rebuild resident NVFP4 weights after an optimizer step."""
        self.layer.refresh_weights()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if hidden_states.ndim != 2 or hidden_states.shape[0] != self.tokens:
            raise ValueError(
                f"expected hidden_states [{self.tokens}, hidden], got {tuple(hidden_states.shape)}"
            )

        hidden_states = hidden_states.contiguous()
        top_k_index = top_k_index.to(torch.int32).contiguous()
        top_k_weights = top_k_weights.contiguous()
        gather_idx, cu, probs, slots = self.dispatch(top_k_index, top_k_weights.detach())

        if not self._calibrated:
            self.layer.calibrate(hidden_states.detach(), gather_idx, cu, probs)
            self._calibrated = True

        self.layer.sr_seed = self.sr_seed + self._step
        self._step += 1
        probs = self.dispatch.differentiable_probs(top_k_weights)
        return self.layer(
            hidden_states,
            gather_idx,
            cu,
            probs,
            slots,
            off_pad=self.dispatch.off_pad,
        )


__all__ = ["Qwen3Nvfp4Experts"]
