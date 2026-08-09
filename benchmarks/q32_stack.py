"""Qwen3 layer-0 context used around each expert implementation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _rms(x, weight, eps=1e-6):
    # Match Transformers' Qwen3MoeRMSNorm boundary exactly: reduction in
    # float32, cast the normalized value back, then apply the BF16 weight.
    input_dtype = x.dtype
    xf = x.float()
    normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return weight * normalized.to(input_dtype)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rope(seq_len, head_dim, theta, device):
    inv = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    freq = torch.outer(torch.arange(seq_len, dtype=torch.float32, device=device), inv)
    emb = torch.cat((freq, freq), dim=-1)
    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


class QwenNonExpert(nn.Module):
    def __init__(self, spec, layer_state, attention_backend: str):
        super().__init__()
        if attention_backend != "sdpa":
            raise ValueError(attention_backend)
        self.spec = spec
        self.attention_backend = attention_backend

        def param(name):
            try:
                return nn.Parameter(layer_state[name].cuda().to(torch.bfloat16))
            except KeyError as exc:
                raise RuntimeError(f"trace lacks Qwen layer-0 tensor {name!r}") from exc

        self.in_norm = param("input_layernorm.weight")
        self.post_norm = param("post_attention_layernorm.weight")
        self.wq = param("self_attn.q_proj.weight")
        self.wk = param("self_attn.k_proj.weight")
        self.wv = param("self_attn.v_proj.weight")
        self.wo = param("self_attn.o_proj.weight")
        self.qnorm = param("self_attn.q_norm.weight")
        self.knorm = param("self_attn.k_norm.weight")
        self.router = param("mlp.gate.weight")

    def _attention(self, q, k, v):
        # q/k/v enter as [B,S,H,D].
        return F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
            enable_gqa=True,
        ).transpose(1, 2)

    def forward(self, block_input):
        spec = self.spec
        b, s, _ = block_input.shape
        xn = _rms(block_input, self.in_norm)
        q = F.linear(xn, self.wq).view(b, s, spec.num_heads, spec.head_dim)
        k = F.linear(xn, self.wk).view(b, s, spec.num_kv_heads, spec.head_dim)
        v = F.linear(xn, self.wv).view(b, s, spec.num_kv_heads, spec.head_dim)
        q = _rms(q, self.qnorm)
        k = _rms(k, self.knorm)
        cos, sin = _rope(s, spec.head_dim, spec.rope_theta, block_input.device)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        attn = self._attention(q, k, v).reshape(b, s, -1)
        residual = block_input + F.linear(attn, self.wo)
        expert_input = _rms(residual, self.post_norm).reshape(b * s, spec.hidden)
        logits = F.linear(expert_input, self.router)
        probs = torch.softmax(logits.float(), dim=-1)
        topv, topi = torch.topk(probs, spec.topk, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True)
        return residual, expert_input, topi.to(torch.int32), topv


class QwenMoEBlock(nn.Module):
    def __init__(self, nonexpert, expert):
        super().__init__()
        self.nonexpert = nonexpert
        self.expert = expert

    def forward(self, x, step=0):
        residual, expert_input, topi, topv = self.nonexpert(x)
        y = self.expert(expert_input, topi, topv, step)
        return residual + y.view_as(residual)


def compile_nonexpert(module: nn.Module, mode: str):
    if mode == "eager":
        return module
    if mode not in ("default", "reduce-overhead", "max-autotune-no-cudagraphs"):
        raise ValueError(mode)
    return torch.compile(module, mode=mode, fullgraph=False, dynamic=False)


__all__ = ["QwenMoEBlock", "QwenNonExpert", "compile_nonexpert"]
