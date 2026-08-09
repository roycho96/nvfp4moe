"""NVFP4 MoE expert layer with forward, backward, and router gradients."""

import torch
from torch import nn

from . import _vendor  # noqa: F401
from quack.blockscaled.quantize import pack_scale_2d_to_blocked_contig, to_nvfp4  # noqa: E402
from .finalize import moe_finalize, moe_finalize_bwd
from .gemm import (dgrad2_amax_mod, dgrad2_mod, fc1_quant_mod,
                   fc2_weighted_mod, gemm)
from .cwgrad import GroupedWgrad
from .quant import nvfp4_quantize_colwise, nvfp4_quantize_rowwise, nvfp4_rht_amax
from .recipe import TensorScale, _DEN


def _quant_expert_stack(mats, pts):
    qs, ss = [], []
    for m in mats:
        q, sf, _ = to_nvfp4(m.contiguous(), 16, pts)
        qs.append(q)
        ss.append(pack_scale_2d_to_blocked_contig(sf)[0])
    return torch.stack(qs).view(torch.float4_e2m1fn_x2), torch.stack(ss)


_GKW = dict(tile_M=128, tile_N=256, cluster_M=1, cluster_N=1, max_swizzle_size=1)
# Tuned fine-grained configs for FC2 and input-gradient GEMMs.
_GKW_2CTA = dict(tile_M=256, tile_N=256, cluster_M=2, cluster_N=1,
                 max_swizzle_size=1)
_GKW_D2 = dict(tile_M=128, tile_N=192, cluster_M=1, cluster_N=1,
               max_swizzle_size=1)


def _gkws(L, M):
    """Select tile configs for FC2 and the two input-gradient GEMMs."""
    if L.gemm_cfg == "legacy" or L.pre_permute or M < 1024 * L.E:
        return _GKW, _GKW, _GKW
    return _GKW_2CTA, _GKW_D2, _GKW_2CTA


class _MoEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, gather_idx, cu, probs_sorted, slots, off_pad,
                layer):
        L = layer
        T, d = x.shape
        M = gather_idx.numel()
        I = L.I
        # Quantized inputs are saved for activation recomputation. The default
        # path gathers inside FC1; pre_permute materializes gathered rows first.
        L.s_x.update(x)
        if L.pre_permute:
            xg = L._buf("xg", (M, d), torch.bfloat16)
            moe_finalize_bwd(x, gather_idx, L._ones(M), xg)
            qx_u8 = torch.empty(M, d // 2, dtype=torch.uint8, device=x.device)
            sfx = torch.empty(1, L.rm_max, d // 64, 32, 4, 4,
                              dtype=torch.float8_e4m3fn, device=x.device)
            nvfp4_quantize_rowwise(xg, cu, L.s_x.pair, qx_u8, sfx)
        else:
            qx_u8 = torch.empty(T, d // 2, dtype=torch.uint8, device=x.device)
            sfx = torch.empty(T, d // 16, dtype=torch.float8_e4m3fn,
                              device=x.device)
            nvfp4_quantize_rowwise(x, L._cu_x(T), L.s_x.pair, qx_u8, sfx,
                                   sf_layout="linear")
        qx = qx_u8.view(torch.float4_e2m1fn_x2)
        # FC1 fuses the gather, gated activation, and NVFP4 post-activation.
        ascale = (L.s_x.pts * L.p_w1).reshape(1)
        q_h = L._buf("q_h", (M, I // 2), torch.float4_e2m1fn_x2)
        sf_h = L._buf("sf_h", (L.rm_max, I // 64, 32, 4, 4),
                      torch.float8_e4m3fn, batch1=True)
        sf_h.zero_()
        gkw = (dict(SFA=sfx) if L.pre_permute else
               dict(A_idx=gather_idx, use_tma_gather=True,
                    SFA=sfx.view(torch.uint8).view(torch.int32),
                    post_init_attrs=L._fc1_post_init_attrs))
        L.fc1.gemm(
            qx, L.qb1, None,
            epi_args=dict(postact=q_h, postact_sf=sf_h, ascale=ascale,
                          sfd_norm_const=L.s_h.inv_pts),
            cu_seqlens_m=cu, SFB=L.sfb1,
            bs_format_a="nvfp4", bs_format_b="nvfp4", **gkw, **_GKW,
        )
        # FC2 folds routing probabilities and scales into its epilogue.
        # Save a private output buffer when router gradients are required.
        need_dp = probs_sorted.requires_grad
        if need_dp:
            yw = torch.empty(M, d, device=x.device, dtype=torch.bfloat16)
        else:
            yw = L._buf("yw", (M, d), torch.bfloat16)
        fc2_args = dict(yw=yw, probs=probs_sorted,
                        pscale=(L.s_h.pts * L.p_w2).reshape(1))
        gkw_fc2, _, _ = _gkws(L, M)
        L.fc2.gemm(
            q_h, L.qb2, None, epi_args=fc2_args,
            cu_seqlens_m=cu, SFA=sf_h, SFB=L.sfb2,
            bs_format_a="nvfp4", bs_format_b="nvfp4", **gkw_fc2,
        )
        y = torch.empty(T, d, device=x.device, dtype=torch.bfloat16)
        # Fixed-order combine keeps repeated runs deterministic.
        moe_finalize(yw, slots, y, L.topk, tile_t=4, n_frag=2)
        ctx.save_for_backward(x, qx_u8, sfx, gather_idx, cu, probs_sorted,
                              slots, off_pad,
                              *((yw,) if need_dp else ()))
        ctx.need_dprobs = need_dp
        ctx.layer = L
        return y

    @staticmethod
    def backward(ctx, dY):
        saved = ctx.saved_tensors
        x, qx_u8, sfx, gi, cu, ps, slots, off_pad = saved[:8]
        yw_saved = saved[8] if ctx.need_dprobs else None
        L = ctx.layer
        T, d = x.shape
        M = gi.numel()
        I = L.I
        dY = dY.contiguous().to(torch.bfloat16)
        qx = qx_u8.view(torch.float4_e2m1fn_x2)
        # Delayed mode reuses the previous post-RHT column amax. Row scales
        # remain same-step, and the first backward seeds delayed state.
        dca = L.delayed_col_amax
        warm = dca and not L._dcol_ready
        # Gather routed output gradients and optionally collect router dots.
        dY_M = L._buf("dY_M", (M, d), torch.bfloat16)
        dp_kw = {}
        n_dp = -(-d // 2048)
        if ctx.need_dprobs:
            dp_part = L._buf("dprobs_part", (n_dp * M,), torch.float32)
            dp_kw = dict(yw=yw_saved, dot_out=dp_part)
        if dca:
            n_fb = -(-M // 8) * n_dp
            fb_part = L._buf("dca_fb_part", (n_fb,), torch.float32)
            moe_finalize_bwd(dY, gi, ps.float().contiguous(), dY_M,
                             amax_out=fb_part, **dp_kw)
            torch.amax(fb_part, 0, out=L._dca_red)
            L.s_dy.set_amax(L._dca_red)
            if warm:
                L._amax2(dY_M, cu, None, L.s_dy_c)
        else:
            moe_finalize_bwd(dY, gi, ps.float().contiguous(), dY_M, **dp_kw)
            if L.rht:
                # Collect row and post-RHT column amax values in one pass.
                L._amax2(dY_M, cu, L.s_dy, L.s_dy_c)
            else:
                L.s_dy.update(dY_M)
        q_dy = L._buf("q_dy", (M, d // 2), torch.uint8)
        sf_dy = L._buf("sf_dy", (L.rm_max, d // 64, 32, 4, 4),
                       torch.float8_e4m3fn, batch1=True)
        rd = "sr" if L.sr_seed is not None else "rn"
        sv = L.sr_seed if L.sr_seed is not None else 0
        nvfp4_quantize_rowwise(dY_M, cu, L.s_dy.pair, q_dy, sf_dy,
                               rounding=rd, seed=sv)
        # Recompute the BF16 pre-activation using the forward input path.
        preact = L._buf("preact", (M, 2 * I), torch.bfloat16)
        gkw = (dict(SFA=sfx) if L.pre_permute else
               dict(A_idx=gi, use_tma_gather=True,
                    SFA=sfx.view(torch.uint8).view(torch.int32)))
        gemm(qx, L.qb1, preact, None, None,
             cu_seqlens_m=cu, SFB=L.sfb1,
             alpha=(L.s_x.pts * L.p_w1).reshape(1),
             bs_format_a="nvfp4", bs_format_b="nvfp4", **gkw, **_GKW)
        # dgrad2 fuses the gated-activation derivative and saved activation.
        dH = L._buf("dH", (M, 2 * I), torch.bfloat16)
        hh = L._buf("hh", (M, I), torch.bfloat16)
        a2 = L.s_dy.pts * L.p_w2d  # (1,) device
        dm_args = dict(mAuxOut=hh, gscale=a2.reshape(1))
        _, gkw_dg2, gkw_dg1 = _gkws(L, M)
        if dca:
            # Reuse row-amax partial storage in delayed mode.
            dmx = L._buf("dmx_part", (M, -(-I // gkw_dg2["tile_N"])),
                         torch.float32)
            dm_args["damax"] = dmx
        L.dmod.gemm(q_dy.view(torch.float4_e2m1fn_x2), L.qW2d, dH, C=preact,
                    epi_args=dm_args,
                    cu_seqlens_m=cu, SFA=sf_dy, SFB=L.sW2d,
                    bs_format_a="nvfp4", bs_format_b="nvfp4", **gkw_dg2)
        # dgrad1
        if L.rht:
            if dca:
                # Preserve the BF16 store rounding used by the default path.
                torch.amax(dmx.view(-1), 0, out=L._dh_red)
                L._dh_red16.copy_(L._dh_red)
                L._dh_redf.copy_(L._dh_red16)
                L.s_dh.set_amax(L._dh_redf)
                if warm:
                    L._amax2(dH, cu, None, L.s_dh_c)
            else:
                L._amax2(dH, cu, L.s_dh, L.s_dh_c)
        else:
            L.s_dh.update(dH)
        q_dh = L._buf("q_dh", (M, I), torch.uint8)  # 2I/2 bytes
        sf_dh = L._buf("sf_dh", (L.rm_max, 2 * I // 64, 32, 4, 4),
                       torch.float8_e4m3fn, batch1=True)
        nvfp4_quantize_rowwise(dH, cu, L.s_dh.pair, q_dh, sf_dh,
                               rounding=rd, seed=sv + 1)
        dX_M = L._buf("dX_M", (M, d), torch.bfloat16)
        gemm(q_dh.view(torch.float4_e2m1fn_x2), L.qW1T, dX_M, None, None,
             cu_seqlens_m=cu, SFA=sf_dh, SFB=L.sW1T,
             alpha=(L.s_dh.pts * L.p_w1t).reshape(1),
             bs_format_a="nvfp4", bs_format_b="nvfp4", **gkw_dg1)
        dX = torch.empty(T, d, device=x.device, dtype=torch.bfloat16)
        moe_finalize(dX_M, slots, dX, L.topk, tile_t=4, n_frag=2)
        # Columnwise quantization feeds one grouped wgrad per weight.
        # Buffers use the padded static upper bound for variable routing.
        mp_max = L._mp_max(M)
        if off_pad is None:  # torch fallback (dispatch can emit this for free)
            seg = cu[1:] - cu[:-1]
            off_pad = (((seg + 127) // 128) * 128).cumsum(0).to(torch.int32)
        s_hh, s_cx = L.s_hh_c, L.s_x_c
        if L.rht:
            if not dca or warm:
                L._amax2(hh, cu, None, s_hh)
                L._amax2(x, cu, None, s_cx, gather_idx=gi)
        else:
            s_hh.update(hh)
            s_cx.update(x)
        cw = {}
        pend = []  # Delayed scales are refreshed after wgrad consumes them.
        n_ct = -(-M // 128) + L.E
        for name, z, F_, sc, gidx, crd, csd in (
            # Apply stochastic rounding only to gradient casts.
            ("dy", dY_M, d, L.s_dy_c if L.rht else L.s_dy, None, rd, sv + 2),
            ("hh", hh, I, s_hh, None, "rn", 0),
            ("dh", dH, 2 * I, L.s_dh_c if L.rht else L.s_dh, None, rd, sv + 3),
            ("x", x, d, s_cx, gi, "rn", 0),
        ):
            qb = L._buf(f"cw_{name}_q", (F_, mp_max // 2), torch.uint8)
            sb = L._buf(f"cw_{name}_sf", (F_ * mp_max // 16,),
                        torch.float8_e4m3fn)
            aout = None
            if dca:
                aout = L._buf(f"dca_{name}_part", (n_ct * (F_ // 128),),
                              torch.float32)
                pend.append((sc, aout))
            nvfp4_quantize_colwise(z, cu, sc.pair, qb, sb, gather_idx=gidx,
                                   rounding=crd, seed=csd, rht=L.rht,
                                   amax_out=aout)
            cw[name] = (qb, sb, sc)
        # Accumulation overwrites on the first microbatch, then reduce-adds
        # into layer-owned buffers. Autograd does not add those gradients again.
        acc = L.wgrad_accumulate
        if acc:
            dW1, dW2 = L.acc_dw1, L.acc_dw2
            wg1_ = L.wg1 if L._acc_fresh else L.wg1_acc
            wg2_ = L.wg2 if L._acc_fresh else L.wg2_acc
        else:
            # Keep fresh gradients because autograd may retain prior aliases.
            dW1 = torch.empty(L.E, 2 * I, d, device=x.device,
                              dtype=torch.bfloat16)
            dW2 = torch.empty(L.E, d, I, device=x.device,
                              dtype=torch.bfloat16)
            wg1_, wg2_ = L.wg1, L.wg2
        for (an, bn, out, wg, gs) in (("dy", "hh", dW2, wg2_, L._gs2),
                                      ("dh", "x", dW1, wg1_, L._gs1)):
            qa, sa, pa = cw[an]
            qb_, sb_, pb = cw[bn]
            gs.copy_((pa.pts * pb.pts).expand(L.E))
            wg(qa, qb_, sa, sb_, off_pad, out, gs, L._gs_one)
        if acc:
            L._acc_fresh = False
        # Refresh delayed scales only after current scale products are consumed.
        for sc, part in pend:
            torch.amax(part, 0, out=L._dca_red)
            sc.set_amax(L._dca_red)
        if dca:
            L._dcol_ready = True
        # Recover router gradients from routed outputs. Exact-zero probabilities
        # return zero because the fused forward does not retain unweighted y2.
        dprobs = None
        if ctx.need_dprobs:
            dot = (dp_part.view(n_dp, M).sum(0) if n_dp > 1
                   else dp_part[:M])
            dprobs = torch.where(
                ps != 0, dot / ps,
                torch.zeros((), device=dot.device, dtype=torch.float32))
        if acc:
            return dX, None, None, None, None, dprobs, None, None, None
        return dX, dW1, dW2, None, None, dprobs, None, None, None


class MoEExpertLayer(nn.Module):
    """Training-capable NVFP4 expert layer with external routing selection."""

    def __init__(self, d, I, E, topk, rht=True, pre_permute=False,
                 delayed_col_amax=False, wgrad_accumulate=False,
                 gemm_cfg="auto", fc1_stage_overrides=None,
                 activation="swiglu"):
        super().__init__()
        assert d % 256 == 0 and I % 128 == 0, "tile/SF alignment (see README)"
        assert not delayed_col_amax or rht, \
            "delayed_col_amax targets the rht pre-passes (rht=True only)"
        assert gemm_cfg in ("auto", "legacy")
        assert activation in ("swiglu", "geglu"), \
            "activation must be 'swiglu' (Qwen) or 'geglu' (Gemma 4)"
        assert fc1_stage_overrides is None or (
            len(fc1_stage_overrides) == 3
            and all(x is None or isinstance(x, int) for x in fc1_stage_overrides)
        )
        # "auto" selects tuned tiles; "legacy" pins the baseline config.
        self.gemm_cfg = gemm_cfg
        self.activation = activation
        self._fc1_post_init_attrs = (() if fc1_stage_overrides is None else
                                     (("stage_overrides", fc1_stage_overrides),))
        self.d, self.I, self.E, self.topk = d, I, E, topk
        # Delayed column amax removes pre-passes at the cost of one-step scale lag.
        self.delayed_col_amax = delayed_col_amax
        self._dcol_ready = False
        self._dca_red = torch.empty((), dtype=torch.float32, device="cuda")
        # RHT follows the columnwise placement used for wgrad operands.
        self.rht = rht
        # Optionally materialize gathered rows before FC1 and recomputation.
        self.pre_permute = pre_permute
        self.w1 = nn.Parameter(
            torch.randn(E, 2 * I, d, dtype=torch.bfloat16) * d**-0.5)
        self.w2 = nn.Parameter(
            torch.randn(E, d, I, dtype=torch.bfloat16) * I**-0.5)
        self.fc1 = fc1_quant_mod(activation)
        self.fc2 = fc2_weighted_mod()
        # Delayed mode collects dgrad2 row-amax partials in the epilogue.
        self.dmod = (dgrad2_amax_mod(activation) if delayed_col_amax
                     else dgrad2_mod(activation))
        # Match the BF16 store rounding before updating the row scale.
        self._dh_red = torch.empty((), dtype=torch.float32, device="cuda")
        self._dh_red16 = torch.empty((), dtype=torch.bfloat16, device="cuda")
        self._dh_redf = torch.empty((), dtype=torch.float32, device="cuda")
        self.s_x, self.s_h = TensorScale(), TensorScale()
        self.s_dy, self.s_dh = TensorScale(), TensorScale()
        # Wgrad operands keep separate post-RHT column scales.
        self.s_dy_c, self.s_dh_c = (TensorScale(te_rht=rht),
                                    TensorScale(te_rht=rht))
        self.s_hh_c, self.s_x_c = (TensorScale(te_rht=rht),
                                   TensorScale(te_rht=rht))
        # Grouped wgrad computes dW2 = dy @ hh^T and dW1 = dh @ x^T.
        self.wg2 = GroupedWgrad(d, I, E)
        self.wg1 = GroupedWgrad(2 * I, d, E)
        # Optional microbatch accumulation happens inside grouped wgrad stores.
        # commit_wgrad() exposes the layer-owned buffers to the optimizer.
        self.wgrad_accumulate = wgrad_accumulate
        self._acc_fresh = True
        if wgrad_accumulate:
            self.wg2_acc = GroupedWgrad(d, I, E, accumulate=True)
            self.wg1_acc = GroupedWgrad(2 * I, d, E, accumulate=True)
            self.acc_dw1 = torch.zeros(E, 2 * I, d, dtype=torch.bfloat16,
                                       device="cuda")
            self.acc_dw2 = torch.zeros(E, d, I, dtype=torch.bfloat16,
                                       device="cuda")
        self._gs1 = torch.empty(E, dtype=torch.float32, device="cuda")
        self._gs2 = torch.empty(E, dtype=torch.float32, device="cuda")
        self._gs_one = torch.ones(E, dtype=torch.float32, device="cuda")
        self._bufs = {}
        self.rm_max = 0
        # An integer seed enables reproducible Philox stochastic rounding.
        self.sr_seed = None

    def _mp_max(self, M):
        """Static upper bound of the 128-padded token total (M is T*topk)."""
        return M + 128 * self.E

    @torch.no_grad()
    def _amax2(self, z, cu, row_scale, col_scale, gather_idx=None):
        """Collect raw row and post-RHT column amax values in one pass."""
        M = int(gather_idx.numel()) if gather_idx is not None else z.shape[0]
        n = (-(-M // 128) + self.E) * (z.shape[1] // 128)
        part = self._buf("amax_part", (n, 2), torch.float32)
        part.zero_()
        nvfp4_rht_amax(z, cu, part, gather_idx=gather_idx)
        red = self._bufs.get(("amax_red", torch.float32))
        if red is None:
            red = torch.empty(2, dtype=torch.float32, device="cuda")
            self._bufs[("amax_red", torch.float32)] = red
        torch.amax(part, 0, out=red)
        if row_scale is not None:
            row_scale.set_amax(red[0:1])
        col_scale.set_amax(red[1:2])

    def _ones(self, M):
        """Return a cached FP32 identity scale for pre-permute gathers."""
        b = self._bufs.get(("ones_M", torch.float32))
        if b is None or b.numel() < M:
            b = torch.ones(M, dtype=torch.float32, device="cuda")
            self._bufs[("ones_M", torch.float32)] = b
        return b[:M]

    def _cu_x(self, T):
        """Cached [0, T] cu for the single-segment x-quantize."""
        c = self._bufs.get(("cu_x", T))
        if c is None:
            c = torch.tensor([0, T], dtype=torch.int32, device="cuda")
            self._bufs[("cu_x", T)] = c
        return c

    def _buf(self, name, shape, dtype, batch1=False):
        shape = (1, *shape) if batch1 else shape
        key = (name, dtype)
        b = self._bufs.get(key)
        need = 1
        for sdim in shape:
            need *= int(sdim)
        if b is None or b.numel() < need:
            if dtype == torch.float4_e2m1fn_x2:  # no zeros/fill kernels for fp4
                b = torch.zeros(*shape, dtype=torch.uint8, device="cuda").view(dtype)
            else:
                b = torch.zeros(*shape, dtype=dtype, device="cuda")
            self._bufs[key] = b
            return b
        return b.view(-1)[:need].view(*shape)

    def commit_wgrad(self):
        """Expose accumulated wgrads and start a fresh accumulation cycle.

        Call after the last microbatch backward and before optimizer.step().
        The gradients alias persistent layer buffers without a copy.
        """
        assert self.wgrad_accumulate, "layer built with wgrad_accumulate=False"
        self.w1.grad = self.acc_dw1
        self.w2.grad = self.acc_dw2
        self._acc_fresh = True

    @torch.no_grad()
    def refresh_weights(self):
        """(Re)quantize the master weights, both orientations, per-tensor pts."""
        dev = self.w1.device
        for nm, mats in (("w1", [self.w1[e] for e in range(self.E)]),
                         ("w2", [self.w2[e] for e in range(self.E)]),
                         ("w2d", [self.w2[e].t() for e in range(self.E)]),
                         ("w1t", [self.w1[e].t() for e in range(self.E)])):
            amax = torch.stack([m.abs().amax() for m in mats]).amax()
            pts = (amax.float() / _DEN).clamp(min=1e-30).reshape(1).to(dev)
            q, s = _quant_expert_stack(mats, pts)
            setattr(self, {"w1": "qb1", "w2": "qb2", "w2d": "qW2d",
                           "w1t": "qW1T"}[nm], q)
            setattr(self, {"w1": "sfb1", "w2": "sfb2", "w2d": "sW2d",
                           "w1t": "sW1T"}[nm], s)
            setattr(self, {"w1": "p_w1", "w2": "p_w2", "w2d": "p_w2d",
                           "w1t": "p_w1t"}[nm], pts)

    @torch.no_grad()
    def calibrate(self, x, gather_idx, cu, probs_sorted):
        """Seed the delayed activation scale with one BF16 FC1 pass."""
        self.s_x.update(x)
        qx_u8, sfx, _ = to_nvfp4(x, 16, self.s_x.pts)
        M = gather_idx.numel()
        self.rm_max = -(-M // 128) + self.E
        preact = torch.empty(M, 2 * self.I, device=x.device, dtype=torch.bfloat16)
        gemm(qx_u8.view(torch.float4_e2m1fn_x2), self.qb1, preact, None, None,
             cu_seqlens_m=cu, A_idx=gather_idx, use_tma_gather=True,
             SFA=sfx.view(torch.uint8).view(torch.int32), SFB=self.sfb1,
             alpha=(self.s_x.pts * self.p_w1).reshape(1),
             bs_format_a="nvfp4", bs_format_b="nvfp4", **_GKW)
        g, u = preact.float()[:, 0::2], preact.float()[:, 1::2]
        if self.activation == "swiglu":
            h = torch.nn.functional.silu(g) * u
        else:
            h = torch.nn.functional.gelu(g, approximate="tanh") * u
        self.s_h.update(h)

    def forward(self, x, gather_idx, cu, probs_sorted, slots, off_pad=None):
        """Run the routed expert layer; off_pad avoids a backward fallback."""
        M = gather_idx.numel()
        self.rm_max = max(self.rm_max, -(-M // 128) + self.E)
        return _MoEFn.apply(x, self.w1, self.w2, gather_idx, cu,
                            probs_sorted, slots, off_pad, self)
