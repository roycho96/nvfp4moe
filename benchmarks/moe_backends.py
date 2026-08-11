"""Expert implementations used by the MoE benchmark.

The DeepGEMM training path follows Megatron-LM's public grouped-GEMM contract:
M-grouped kernels for forward and dgrad, and K-grouped TN kernels for wgrad.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _pad_weights(gate_up: torch.Tensor, down: torch.Tensor, padded_i: int):
    """Pad checkpoint [gate;up] weights without changing model semantics."""
    e, two_i, d = gate_up.shape
    i = two_i // 2
    if i == padded_i:
        return gate_up.contiguous(), down.contiguous()
    gate, up = gate_up[:, :i], gate_up[:, i:]
    z = torch.zeros(e, padded_i - i, d, dtype=gate_up.dtype, device=gate_up.device)
    gate_up_pad = torch.cat((torch.cat((gate, z), 1), torch.cat((up, z), 1)), 1)
    down_pad = F.pad(down, (0, padded_i - i))
    return gate_up_pad.contiguous(), down_pad.contiguous()


def _interleave_gate_up(gate_up: torch.Tensor) -> torch.Tensor:
    """[gate block; up block] -> [g0,u0,g1,u1,...] used by nvfp4moe."""
    gate, up = gate_up.chunk(2, dim=1)
    return torch.stack((gate, up), dim=2).flatten(1, 2).contiguous()


def _deinterleave_gate_up(gate_up: torch.Tensor) -> torch.Tensor:
    """[g0,u0,g1,u1,...] -> [gate block; up block] checkpoint layout."""
    return torch.cat((gate_up[:, 0::2], gate_up[:, 1::2]), dim=1).contiguous()


def _activation(gate: torch.Tensor, up: torch.Tensor, kind: str):
    if kind == "swiglu":
        return F.silu(gate) * up
    if kind == "geglu":
        return F.gelu(gate, approximate="tanh") * up
    raise ValueError(kind)


def _dactivation(gate: torch.Tensor, up: torch.Tensor, dout: torch.Tensor, kind: str):
    """BF16 output gradients matching an unfused BF16 activation boundary."""
    g, u, dy = gate.float(), up.float(), dout.float()
    if kind == "swiglu":
        sig = torch.sigmoid(g)
        act = g * sig
        dact = sig * (1.0 + g * (1.0 - sig))
    elif kind == "geglu":
        # tanh GELU, the exact derivative of torch.gelu(approximate="tanh")
        c = 0.7978845608028654
        q = c * (g + 0.044715 * g * g * g)
        tq = torch.tanh(q)
        act = 0.5 * g * (1.0 + tq)
        dact = 0.5 * (1.0 + tq) + 0.5 * g * (1.0 - tq * tq) * c * (1.0 + 3.0 * 0.044715 * g * g)
    else:
        raise ValueError(kind)
    return (dy * u * dact).to(torch.bfloat16), (dy * act).to(torch.bfloat16)


def _te_autocast(te, recipe):
    autocast = getattr(te, "autocast", None)
    if callable(autocast):
        return autocast(enabled=True, recipe=recipe)
    return te.fp8_autocast(enabled=True, fp8_recipe=recipe)


@dataclass
class BackendInfo:
    name: str
    precision: str
    training: bool
    includes_dispatch: bool = True


class Nvfp4MoeExpert(nn.Module):
    info = BackendInfo("nvfp4moe", "NVFP4 x NVFP4", True)

    def __init__(self, spec, trace):
        super().__init__()
        from nvfp4moe import MoEDispatch, MoEExpertLayer

        i_pad = spec.padded_intermediate
        gate_up, down = _pad_weights(trace["gate_up_weight"], trace["down_weight"], i_pad)
        self.layer = MoEExpertLayer(
            spec.hidden,
            i_pad,
            spec.experts,
            spec.topk,
            rht=True,
            delayed_col_amax=True,
            activation=spec.activation,
        ).cuda()
        with torch.no_grad():
            self.layer.w1.copy_(_interleave_gate_up(gate_up))
            self.layer.w2.copy_(down)
        self.layer.refresh_weights()
        self.dispatch = MoEDispatch(trace["expert_input"].shape[0], spec.experts, spec.topk)
        self._calibrated = False

    def forward(self, x, topi, topv, step=0):
        gi, cu, ps, slots = self.dispatch(topi, topv)
        if not self._calibrated:
            self.layer.calibrate(x.detach(), gi, cu, ps)
            self._calibrated = True
        self.layer.sr_seed = 123_000 + step
        ps_diff = self.dispatch.differentiable_probs(topv)
        return self.layer(x, gi, cu, ps_diff, slots, off_pad=self.dispatch.off_pad)

    def training_gradients(self):
        return {
            "gate_up": _deinterleave_gate_up(self.layer.w1.grad),
            "down": self.layer.w2.grad.contiguous(),
        }


class TEExpert(nn.Module):
    def __init__(self, spec, trace, nvfp4: bool):
        super().__init__()
        import transformer_engine.pytorch as te
        from transformer_engine.common import recipe as te_recipe

        self.info = BackendInfo(
            "te_nvfp4_2x64" if nvfp4 else "te_bf16",
            "TE NVFP4" if nvfp4 else "BF16",
            True,
        )
        gate_up, down = _pad_weights(
            trace["gate_up_weight"], trace["down_weight"], spec.padded_intermediate
        )
        # Interleaving lets all backends use the same simple [0::2]/[1::2]
        # activation convention while retaining the exact checkpoint values.
        gate_up = _interleave_gate_up(gate_up)
        self.te = te
        self.nvfp4 = nvfp4
        self.align = 64 if nvfp4 else 16
        self.recipe = te_recipe.NVFP4BlockScaling() if nvfp4 else None
        # TE 2.17's NVFP4 grouped RHT kernel accepts at most 64 tensors per
        # launch. Preserve the real E=128 model by using two grouped launches,
        # not by shrinking the checkpoint geometry.
        self.group_width = 64 if nvfp4 else spec.experts
        group_sizes = [
            min(self.group_width, spec.experts - start)
            for start in range(0, spec.experts, self.group_width)
        ]
        self.gl1 = nn.ModuleList(
            [
                te.GroupedLinear(
                    group_size,
                    spec.hidden,
                    2 * spec.padded_intermediate,
                    bias=False,
                    params_dtype=torch.bfloat16,
                ).cuda()
                for group_size in group_sizes
            ]
        )
        self.gl2 = nn.ModuleList(
            [
                te.GroupedLinear(
                    group_size,
                    spec.padded_intermediate,
                    spec.hidden,
                    bias=False,
                    params_dtype=torch.bfloat16,
                ).cuda()
                for group_size in group_sizes
            ]
        )
        with torch.no_grad():
            for expert in range(spec.experts):
                group, local = divmod(expert, self.group_width)
                getattr(self.gl1[group], f"weight{local}").copy_(gate_up[expert])
                getattr(self.gl2[group], f"weight{local}").copy_(down[expert])
        self.experts = spec.experts
        self.activation = spec.activation
        self.first = True
        self._fmb = True

    def _gl(self, module, x, splits):
        if self._fmb:
            try:
                return module(x, splits, is_first_microbatch=self.first)
            except TypeError:
                self._fmb = False
        return module(x, splits)

    def _grouped(self, modules, x, splits):
        outputs = []
        row_start = 0
        for group, module in enumerate(modules):
            expert_start = group * self.group_width
            local_splits = splits[expert_start : expert_start + self.group_width]
            row_end = row_start + sum(local_splits)
            outputs.append(self._gl(module, x[row_start:row_end], local_splits))
            row_start = row_end
        if row_start != x.shape[0]:
            raise RuntimeError("TE grouped shard rows do not cover permuted input")
        return torch.cat(outputs, dim=0)

    def forward(self, x, topi, topv, step=0):
        from transformer_engine.pytorch import permutation as te_perm

        t = x.shape[0]
        mask = torch.zeros(t, self.experts, dtype=torch.int32, device=x.device)
        mask.scatter_(1, topi.long(), 1)
        probs = torch.zeros(t, self.experts, dtype=torch.float32, device=x.device)
        probs.scatter_(1, topi.long(), topv.float())
        counts = mask.sum(0)
        xp, pprob, row_map, pad_offsets, target_counts = te_perm.moe_permute_and_pad_with_probs(
            x, probs, mask, counts, self.align
        )
        splits = target_counts.tolist()
        autocast = _te_autocast(self.te, self.recipe) if self.nvfp4 else contextlib.nullcontext()
        with autocast:
            h = self._grouped(self.gl1, xp, splits)
            hh = _activation(h[:, 0::2], h[:, 1::2], self.activation)
            ym = self._grouped(self.gl2, hh, splits)
        self.first = False
        ym = ym * pprob[:, None]
        return te_perm.moe_unpermute(
            ym,
            row_map,
            map_type="mask",
            restore_shape=x.shape,
            pad_offsets=pad_offsets,
        ).to(torch.bfloat16)

    def training_gradients(self):
        gate_up = []
        down = []
        for expert in range(self.experts):
            group, local = divmod(expert, self.group_width)
            gate_up.append(getattr(self.gl1[group], f"weight{local}").grad)
            down.append(getattr(self.gl2[group], f"weight{local}").grad)
        return {
            "gate_up": _deinterleave_gate_up(torch.stack(gate_up)),
            "down": torch.stack(down).contiguous(),
        }


def _dg_m_gemm(a, b, m_indices):
    import deep_gemm

    out = torch.empty(a.shape[0], b.shape[1], dtype=torch.bfloat16, device=a.device)
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
        a.contiguous(), b.contiguous(), out, m_indices.contiguous()
    )
    return out


def _dg_wgrad(dy, x, padded_counts, ks_tensor, experts):
    import deep_gemm

    out = torch.zeros(experts, dy.shape[1], x.shape[1], dtype=torch.float32, device=dy.device)
    zero = torch.zeros_like(out)
    deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
        dy.contiguous(),
        x.contiguous(),
        out,
        padded_counts,
        ks_tensor,
        zero,
    )
    return out.to(torch.bfloat16)


class _DeepGEMMTrainFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, m_indices, ks_tensor, padded_counts, activation):
        h = _dg_m_gemm(x, w1, m_indices)
        gate, up = h.chunk(2, dim=-1)
        hh = _activation(gate, up, activation).to(torch.bfloat16)
        y = _dg_m_gemm(hh, w2, m_indices)
        ctx.save_for_backward(x, w1, w2, h, hh, m_indices, ks_tensor)
        ctx.padded_counts = padded_counts
        ctx.activation = activation
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w1, w2, h, hh, m_indices, ks_tensor = ctx.saved_tensors
        dy = dy.contiguous().to(torch.bfloat16)
        experts = w1.shape[0]
        dw2 = _dg_wgrad(dy, hh, ctx.padded_counts, ks_tensor, experts)
        dhh = _dg_m_gemm(dy, w2.transpose(1, 2).contiguous(), m_indices)
        gate, up = h.chunk(2, dim=-1)
        dg, du = _dactivation(gate, up, dhh, ctx.activation)
        dh = torch.cat((dg, du), dim=-1).contiguous()
        dw1 = _dg_wgrad(dh, x, ctx.padded_counts, ks_tensor, experts)
        dx = _dg_m_gemm(dh, w1.transpose(1, 2).contiguous(), m_indices)
        return dx, dw1, dw2, None, None, None, None


class TEDeepGEMMTrainExpert(nn.Module):
    """TE permutation/finalize + DeepGEMM fwd, dgrad and wgrad."""

    info = BackendInfo("te_deepgemm_bf16", "BF16", True)

    def __init__(self, spec, trace):
        super().__init__()
        import deep_gemm

        gate_up, down = _pad_weights(
            trace["gate_up_weight"], trace["down_weight"], spec.padded_intermediate
        )
        self.w1 = nn.Parameter(gate_up.contiguous())
        self.w2 = nn.Parameter(down.contiguous())
        self.experts = spec.experts
        self.activation = spec.activation
        self.align = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
        deep_gemm.set_mk_alignment_for_contiguous_layout(self.align)
        self._layout = None

    def _layout_for(self, target_counts):
        if self._layout is None:
            padded_counts = tuple(int(v) for v in target_counts.tolist())
            m_indices = torch.repeat_interleave(
                torch.arange(self.experts, device=target_counts.device, dtype=torch.int32),
                target_counts.long(),
            ).contiguous()
            ks_tensor = torch.tensor(padded_counts, dtype=torch.int32, device=target_counts.device)
            self._layout = (m_indices, ks_tensor, padded_counts)
        return self._layout

    def forward(self, x, topi, topv, step=0):
        from transformer_engine.pytorch import permutation as te_perm

        t = x.shape[0]
        mask = torch.zeros(t, self.experts, dtype=torch.int32, device=x.device)
        mask.scatter_(1, topi.long(), 1)
        probs = torch.zeros(t, self.experts, dtype=torch.float32, device=x.device)
        probs.scatter_(1, topi.long(), topv.float())
        counts = mask.sum(0)
        xp, pprob, row_map, pad_offsets, target_counts = te_perm.moe_permute_and_pad_with_probs(
            x, probs, mask, counts, self.align
        )
        m_indices, ks_tensor, padded_counts = self._layout_for(target_counts)
        if xp.shape[0] != m_indices.shape[0]:
            raise RuntimeError("routing layout changed after DeepGEMM layout warmup")
        ym = _DeepGEMMTrainFn.apply(
            xp, self.w1, self.w2, m_indices, ks_tensor, padded_counts, self.activation
        )
        ym = ym * pprob[:, None]
        return te_perm.moe_unpermute(
            ym,
            row_map,
            map_type="mask",
            restore_shape=x.shape,
            pad_offsets=pad_offsets,
        ).to(torch.bfloat16)

    def training_gradients(self):
        return {
            "gate_up": self.w1.grad.contiguous(),
            "down": self.w2.grad.contiguous(),
        }


def _dg_grouped_fp4_weights(weight):
    """Prequantize [E,N,K] weights for DeepGEMM's grouped FP8xFP4 NT path."""
    from deep_gemm.utils import per_token_cast_to_fp4

    packed, scales = [], []
    for expert_weight in weight:
        q, sf = per_token_cast_to_fp4(expert_weight.contiguous(), use_ue8m0=True, gran_k=32)
        packed.append(q)
        scales.append(sf)
    return torch.stack(packed), torch.stack(scales)


def _dg_m_fp8_fp4(a, b, m_indices):
    """DeepGEMM grouped FP8 activation x FP4 weight, including A cast."""
    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp8

    qa = per_token_cast_to_fp8(a.contiguous(), use_ue8m0=True, gran_k=128)
    out = torch.empty(a.shape[0], b[0].shape[1], dtype=torch.bfloat16, device=a.device)
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
        qa,
        b,
        out,
        m_indices.contiguous(),
        disable_ue8m0_cast=False,
        recipe_a=(1, 128),
        recipe_b=(1, 32),
    )
    return out


class TEDeepGEMMFP8FP4Expert(nn.Module):
    """TE dispatch plus grouped DeepGEMM FP8xFP4 forward."""

    info = BackendInfo("te_deepgemm_fp8_fp4", "FP8 x FP4", False)

    def __init__(self, spec, trace):
        super().__init__()
        import deep_gemm

        gate_up, down = _pad_weights(
            trace["gate_up_weight"], trace["down_weight"], spec.padded_intermediate
        )
        self.w1 = _dg_grouped_fp4_weights(gate_up)
        self.w2 = _dg_grouped_fp4_weights(down)
        self.experts = spec.experts
        self.activation = spec.activation
        self.align = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
        deep_gemm.set_mk_alignment_for_contiguous_layout(self.align)
        self._m_indices = None

    def _indices_for(self, target_counts):
        if self._m_indices is None:
            self._m_indices = torch.repeat_interleave(
                torch.arange(self.experts, device=target_counts.device, dtype=torch.int32),
                target_counts.long(),
            ).contiguous()
        return self._m_indices

    def forward(self, x, topi, topv, step=0):
        from transformer_engine.pytorch import permutation as te_perm

        t = x.shape[0]
        mask = torch.zeros(t, self.experts, dtype=torch.int32, device=x.device)
        mask.scatter_(1, topi.long(), 1)
        probs = torch.zeros(t, self.experts, dtype=torch.float32, device=x.device)
        probs.scatter_(1, topi.long(), topv.float())
        counts = mask.sum(0)
        xp, pprob, row_map, pad_offsets, target_counts = te_perm.moe_permute_and_pad_with_probs(
            x, probs, mask, counts, self.align
        )
        m_indices = self._indices_for(target_counts)
        h = _dg_m_fp8_fp4(xp, self.w1, m_indices)
        gate, up = h.chunk(2, dim=-1)
        hh = _activation(gate, up, self.activation).to(torch.bfloat16)
        ym = _dg_m_fp8_fp4(hh, self.w2, m_indices) * pprob[:, None]
        return te_perm.moe_unpermute(
            ym,
            row_map,
            map_type="mask",
            restore_shape=x.shape,
            pad_offsets=pad_offsets,
        ).to(torch.bfloat16)


__all__ = [
    "BackendInfo",
    "Nvfp4MoeExpert",
    "TEDeepGEMMFP8FP4Expert",
    "TEDeepGEMMTrainExpert",
    "TEExpert",
]
