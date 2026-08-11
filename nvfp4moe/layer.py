"""NVFP4 MoE expert layer with forward, backward, and router gradients."""

import torch
from torch import nn

from .kernels.epilogue import GatedBackwardEpilogue, GatedEpilogue
from .kernels.finalize import moe_finalize, moe_finalize_bwd
from .kernels.gemm import GroupedNvfp4Gemm
from .kernels.quantize import (
    nvfp4_quantize_colwise,
    nvfp4_quantize_rowwise,
    nvfp4_rht_amax,
)
from .kernels.wgrad import GroupedWgrad
from .recipe import _DEN, TensorScale


def _quant_expert_stack(mats, pts):
    qs, ss = [], []
    pts = pts.detach().to(dtype=torch.float32).reshape(1)
    pts_pair = torch.cat([pts, 1.0 / pts])
    for m in mats:
        rows, features = m.shape
        q = torch.empty(rows, features // 2, dtype=torch.uint8, device=m.device)
        rm = -(-rows // 128)
        sf_buf = torch.empty(
            rm + 1,
            features // 64,
            32,
            4,
            4,
            dtype=torch.float8_e4m3fn,
            device=m.device,
        )
        cu = torch.tensor([0, rows], dtype=torch.int32, device=m.device)
        nvfp4_quantize_rowwise(m.contiguous(), cu, pts_pair, q, sf_buf)
        qs.append(q)
        ss.append(sf_buf[:rm])
    return torch.stack(qs).view(torch.float4_e2m1fn_x2), torch.stack(ss)


_GKW = {"tile_M": 128, "tile_N": 256, "cluster_M": 1, "cluster_N": 1, "max_swizzle_size": 1}
# Tuned fine-grained configs for FC2 and input-gradient GEMMs.
_GKW_2CTA = {"tile_M": 256, "tile_N": 256, "cluster_M": 2, "cluster_N": 1, "max_swizzle_size": 1}
_GKW_D2_NATIVE = {
    "tile_M": 256,
    "tile_N": 128,
    "cluster_M": 2,
    "cluster_N": 1,
    "max_swizzle_size": 1,
}
_GKW_D2_SMALL = {
    "tile_M": 128,
    "tile_N": 128,
    "cluster_M": 1,
    "cluster_N": 1,
    "max_swizzle_size": 1,
}


def _gkws(L, M):
    """Select tile configs for FC2 and the two input-gradient GEMMs."""
    if L.d <= 512 and L.I <= 512:
        dgrad2 = _GKW_D2_SMALL
    elif L.d <= 2048 and L.I <= 768:
        dgrad2 = _GKW_D2_NATIVE
    else:
        dgrad2 = _GKW_2CTA
    return _GKW, dgrad2, _GKW


def _native_fc1_config(L, M):
    """Select the native FC1 tile from the average routed expert size."""
    if L.gemm_cfg == "legacy" or M < 256 * L.E:
        return _GKW
    return _GKW_2CTA


class _MoEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, gather_idx, cu, probs_sorted, slots, off_pad, layer):
        L = layer
        T, d = x.shape
        M = gather_idx.numel()
        I = L.I
        # Quantized inputs are saved for activation recomputation. Native FC1
        # consumes expert-ordered rows, so its quantizer gathers them directly.
        L.s_x.update(x)
        qx_u8 = torch.empty(M, d // 2, dtype=torch.uint8, device=x.device)
        sfx = torch.empty(
            1, L.rm_max, d // 64, 32, 4, 4, dtype=torch.float8_e4m3fn, device=x.device
        )
        nvfp4_quantize_rowwise(
            x,
            cu,
            L.s_x.pair,
            qx_u8,
            sfx,
            gather_idx=gather_idx,
            padded_offsets=off_pad,
        )
        qx = qx_u8.view(torch.float4_e2m1fn_x2)
        sf_rows = -(-M // 128) + L.E
        # FC1 fuses the gather, gated activation, and NVFP4 post-activation.
        ascale = (L.s_x.pts * L.p_w1).reshape(1)
        q_h = L._buf("q_h", (M, I // 2), torch.float4_e2m1fn_x2)
        sf_h = L._buf("sf_h", (L.rm_max, I // 64, 32, 4, 4), torch.float8_e4m3fn, batch1=True)
        sf_h.zero_()
        native_fc1 = L._native_gemm(
            "fc1",
            2 * I,
            d,
            _native_fc1_config(L, M),
            output_dtype=torch.float4_e2m1fn_x2,
            activation=L.activation,
        )
        native_fc1(
            qx,
            L.qb1,
            q_h,
            cu,
            sfx[:, :sf_rows],
            L.sfb1,
            ascale,
            output_sf=sf_h[:, :sf_rows],
            output_scale=L.s_h.pair,
        )
        # FC2 folds routing probabilities and scales into its epilogue.
        # Save a private output buffer when router gradients are required.
        need_dp = probs_sorted.requires_grad
        if need_dp:
            yw = torch.empty(M, d, device=x.device, dtype=torch.bfloat16)
        else:
            yw = L._buf("yw", (M, d), torch.bfloat16)
        fc2_scale = (L.s_h.pts * L.p_w2).reshape(1)
        gkw_fc2, _, _ = _gkws(L, M)
        native_fc2_cfg = _GKW if I <= 1024 else gkw_fc2
        L._native_gemm("fc2", d, I, native_fc2_cfg)(
            q_h,
            L.qb2,
            yw,
            cu,
            sf_h[:, :sf_rows],
            L.sfb2,
            fc2_scale,
        )
        y = torch.empty(T, d, device=x.device, dtype=torch.bfloat16)
        # Fixed-order combine keeps repeated runs deterministic.
        moe_finalize(
            yw,
            slots,
            y,
            L.topk,
            tile_t=4,
            n_frag=2,
            weights=probs_sorted,
        )
        ctx.save_for_backward(
            x, qx_u8, sfx, gather_idx, cu, probs_sorted, slots, off_pad, *((yw,) if need_dp else ())
        )
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
        sf_rows = -(-M // 128) + L.E
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
            dp_kw = {"yw": yw_saved, "dot_out": dp_part}
        if dca:
            n_fb = -(-M // 8) * n_dp
            fb_part = L._buf("dca_fb_part", (n_fb,), torch.float32)
            moe_finalize_bwd(dY, gi, ps.float().contiguous(), dY_M, amax_out=fb_part, **dp_kw)
            torch.amax(fb_part, 0, out=L._dca_red)
            L.s_dy.set_amax(L._dca_red)
            if warm:
                L._amax2(dY_M, cu, None, L.s_dy_c, padded_offsets=off_pad)
        else:
            moe_finalize_bwd(dY, gi, ps.float().contiguous(), dY_M, **dp_kw)
            if L.rht:
                # Collect row and post-RHT column amax values in one pass.
                L._amax2(dY_M, cu, L.s_dy, L.s_dy_c, padded_offsets=off_pad)
            else:
                L.s_dy.update(dY_M)
        q_dy = L._buf("q_dy", (M, d // 2), torch.uint8)
        sf_dy = L._buf("sf_dy", (L.rm_max, d // 64, 32, 4, 4), torch.float8_e4m3fn, batch1=True)
        rd = "sr" if L.sr_seed is not None else "rn"
        sv = L.sr_seed if L.sr_seed is not None else 0
        nvfp4_quantize_rowwise(
            dY_M,
            cu,
            L.s_dy.pair,
            q_dy,
            sf_dy,
            rounding=rd,
            seed=sv,
            padded_offsets=off_pad,
        )
        # Recompute the BF16 pre-activation using the forward input path.
        preact = L._buf("preact", (M, 2 * I), torch.bfloat16)
        preact_scale = (L.s_x.pts * L.p_w1).reshape(1)
        L._native_gemm("recompute", 2 * I, d, _GKW)(
            qx,
            L.qb1,
            preact,
            cu,
            sfx[:, :sf_rows],
            L.sfb1,
            preact_scale,
        )
        # dgrad2 fuses the gated-activation derivative and saved activation.
        dH = L._buf("dH", (M, 2 * I), torch.bfloat16)
        hh = L._buf("hh", (M, I), torch.bfloat16)
        a2 = L.s_dy.pts * L.p_w2d  # (1,) device
        _, gkw_dg2, gkw_dg1 = _gkws(L, M)
        L._native_gemm(
            "dgrad2",
            I,
            d,
            gkw_dg2,
            output_dtype=torch.int32,
            dactivation=L.activation,
        )(
            q_dy.view(torch.float4_e2m1fn_x2),
            L.qW2d,
            dH.view(torch.int32),
            cu,
            sf_dy[:, :sf_rows],
            L.sW2d,
            a2.reshape(1),
            preact=preact,
            aux=hh,
        )
        # dgrad1
        if L.rht:
            if dca:
                L._amax2(
                    dH,
                    cu,
                    L.s_dh,
                    L.s_dh_c if warm else None,
                    padded_offsets=off_pad,
                )
            else:
                L._amax2(dH, cu, L.s_dh, L.s_dh_c, padded_offsets=off_pad)
        else:
            L.s_dh.update(dH)
        q_dh = L._buf("q_dh", (M, I), torch.uint8)  # 2I/2 bytes
        sf_dh = L._buf("sf_dh", (L.rm_max, 2 * I // 64, 32, 4, 4), torch.float8_e4m3fn, batch1=True)
        nvfp4_quantize_rowwise(
            dH,
            cu,
            L.s_dh.pair,
            q_dh,
            sf_dh,
            rounding=rd,
            seed=sv + 1,
            padded_offsets=off_pad,
        )
        dX_M = L._buf("dX_M", (M, d), torch.bfloat16)
        q_dh_fp4 = q_dh.view(torch.float4_e2m1fn_x2)
        dgrad1_scale = (L.s_dh.pts * L.p_w1t).reshape(1)
        L._native_gemm("dgrad1", d, 2 * I, gkw_dg1)(
            q_dh_fp4,
            L.qW1T,
            dX_M,
            cu,
            sf_dh[:, :sf_rows],
            L.sW1T,
            dgrad1_scale,
        )
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
                L._amax2(hh, cu, None, s_hh, padded_offsets=off_pad)
                L._amax2(x, cu, None, s_cx, gather_idx=gi, padded_offsets=off_pad)
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
            sb = L._buf(f"cw_{name}_sf", (F_ * mp_max // 16,), torch.float8_e4m3fn)
            aout = None
            if dca:
                aout = L._buf(f"dca_{name}_part", (n_ct * (F_ // 128),), torch.float32)
                pend.append((sc, aout))
            nvfp4_quantize_colwise(
                z,
                cu,
                sc.pair,
                qb,
                sb,
                gather_idx=gidx,
                rounding=crd,
                seed=csd,
                rht=L.rht,
                amax_out=aout,
                padded_offsets=off_pad,
            )
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
            dW1 = torch.empty(L.E, 2 * I, d, device=x.device, dtype=torch.bfloat16)
            dW2 = torch.empty(L.E, d, I, device=x.device, dtype=torch.bfloat16)
            wg1_, wg2_ = L.wg1, L.wg2
        for an, bn, out, wg, gs in (
            ("dy", "hh", dW2, wg2_, L._gs2),
            ("dh", "x", dW1, wg1_, L._gs1),
        ):
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
            dot = dp_part.view(n_dp, M).sum(0) if n_dp > 1 else dp_part[:M]
            dprobs = dot
        if acc:
            return dX, None, None, None, None, dprobs, None, None, None
        return dX, dW1, dW2, None, None, dprobs, None, None, None


class MoEExpertLayer(nn.Module):
    """Training-capable NVFP4 expert layer with external routing selection."""

    def __init__(
        self,
        d,
        I,
        E,
        topk,
        rht=True,
        pre_permute=False,
        delayed_col_amax=False,
        wgrad_accumulate=False,
        gemm_cfg="auto",
        activation="swiglu",
    ):
        super().__init__()
        assert d % 256 == 0 and I % 128 == 0, "tile/SF alignment (see README)"
        assert not delayed_col_amax or rht, (
            "delayed_col_amax targets the rht pre-passes (rht=True only)"
        )
        assert gemm_cfg in ("auto", "legacy")
        assert activation in ("swiglu", "geglu"), (
            "activation must be 'swiglu' (Qwen) or 'geglu' (Gemma 4)"
        )
        # "auto" selects tuned tiles; "legacy" pins the baseline config.
        self.gemm_cfg = gemm_cfg
        self._native_gemms = {}
        self.activation = activation
        self.d, self.I, self.E, self.topk = d, I, E, topk
        # Delayed column amax removes pre-passes at the cost of one-step scale lag.
        self.delayed_col_amax = delayed_col_amax
        self._dcol_ready = False
        self._dca_red = torch.empty((), dtype=torch.float32, device="cuda")
        # RHT follows the columnwise placement used for wgrad operands.
        self.rht = rht
        # Optionally materialize gathered rows before FC1 and recomputation.
        self.pre_permute = pre_permute
        self.w1 = nn.Parameter(torch.randn(E, 2 * I, d, dtype=torch.bfloat16) * d**-0.5)
        self.w2 = nn.Parameter(torch.randn(E, d, I, dtype=torch.bfloat16) * I**-0.5)
        self.s_x, self.s_h = TensorScale(), TensorScale()
        self.s_dy, self.s_dh = TensorScale(), TensorScale()
        # Wgrad operands keep separate post-RHT column scales.
        self.s_dy_c, self.s_dh_c = (TensorScale(te_rht=rht), TensorScale(te_rht=rht))
        self.s_hh_c, self.s_x_c = (TensorScale(te_rht=rht), TensorScale(te_rht=rht))
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
            self.acc_dw1 = torch.zeros(E, 2 * I, d, dtype=torch.bfloat16, device="cuda")
            self.acc_dw2 = torch.zeros(E, d, I, dtype=torch.bfloat16, device="cuda")
        self._gs1 = torch.empty(E, dtype=torch.float32, device="cuda")
        self._gs2 = torch.empty(E, dtype=torch.float32, device="cuda")
        self._gs_one = torch.ones(E, dtype=torch.float32, device="cuda")
        self._bufs = {}
        self.rm_max = 0
        # An integer seed enables reproducible Philox stochastic rounding.
        self.sr_seed = None

    def _mp_max(self, M):
        """Static upper bound of the 128-padded token total (M is T*topk)."""
        return (-(-M // 128) + self.E) * 128

    @torch.no_grad()
    def _amax2(self, z, cu, row_scale, col_scale, gather_idx=None, padded_offsets=None):
        """Collect raw row and post-RHT column amax values in one pass."""
        M = int(gather_idx.numel()) if gather_idx is not None else z.shape[0]
        n = (-(-M // 128) + self.E) * (z.shape[1] // 128)
        part = self._buf("amax_part", (n, 2), torch.float32)
        part.zero_()
        nvfp4_rht_amax(
            z,
            cu,
            part,
            gather_idx=gather_idx,
            padded_offsets=padded_offsets,
        )
        red = self._bufs.get(("amax_red", torch.float32))
        if red is None:
            red = torch.empty(2, dtype=torch.float32, device="cuda")
            self._bufs[("amax_red", torch.float32)] = red
        torch.amax(part, 0, out=red)
        if row_scale is not None:
            row_scale.set_amax(red[0:1])
        if col_scale is not None:
            col_scale.set_amax(red[1:2])

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

    def _native_gemm(
        self,
        name,
        n,
        k,
        config,
        output_dtype=torch.bfloat16,
        activation=None,
        dactivation=None,
    ):
        key = (
            name,
            n,
            k,
            config["tile_M"],
            config["tile_N"],
            output_dtype,
            activation,
            dactivation,
        )
        runtime = self._native_gemms.get(key)
        if runtime is None:
            epilogue = None
            if activation is not None:
                epilogue = GatedEpilogue(activation)
            elif dactivation is not None:
                epilogue = GatedBackwardEpilogue(dactivation)
            runtime = GroupedNvfp4Gemm(
                self.E,
                n,
                k,
                config["tile_M"],
                config["tile_N"],
                output_dtype=output_dtype,
                epilogue=epilogue,
            )
            self._native_gemms[key] = runtime
        return runtime

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
        for nm, mats in (
            ("w1", [self.w1[e] for e in range(self.E)]),
            ("w2", [self.w2[e] for e in range(self.E)]),
            ("w2d", [self.w2[e].t() for e in range(self.E)]),
            ("w1t", [self.w1[e].t() for e in range(self.E)]),
        ):
            amax = torch.stack([m.abs().amax() for m in mats]).amax()
            pts = (amax.float() / _DEN).clamp(min=1e-30).reshape(1).to(dev)
            q, s = _quant_expert_stack(mats, pts)
            setattr(self, {"w1": "qb1", "w2": "qb2", "w2d": "qW2d", "w1t": "qW1T"}[nm], q)
            setattr(self, {"w1": "sfb1", "w2": "sfb2", "w2d": "sW2d", "w1t": "sW1T"}[nm], s)
            setattr(self, {"w1": "p_w1", "w2": "p_w2", "w2d": "p_w2d", "w1t": "p_w1t"}[nm], pts)

    @torch.no_grad()
    def calibrate(self, x, gather_idx, cu, probs_sorted):
        """Seed the delayed activation scale with one BF16 FC1 pass."""
        self.s_x.update(x)
        _, d = x.shape
        M = gather_idx.numel()
        self.rm_max = -(-M // 128) + self.E
        preact = torch.empty(M, 2 * self.I, device=x.device, dtype=torch.bfloat16)
        qx_u8 = torch.empty(M, d // 2, dtype=torch.uint8, device=x.device)
        sfx = torch.empty(
            1,
            self.rm_max,
            d // 64,
            32,
            4,
            4,
            dtype=torch.float8_e4m3fn,
            device=x.device,
        )
        seg = cu[1:] - cu[:-1]
        off_pad = (((seg + 127) // 128) * 128).cumsum(0).to(torch.int32)
        nvfp4_quantize_rowwise(
            x,
            cu,
            self.s_x.pair,
            qx_u8,
            sfx,
            gather_idx=gather_idx,
            padded_offsets=off_pad,
        )
        self._native_gemm("calibrate", 2 * self.I, d, _GKW)(
            qx_u8.view(torch.float4_e2m1fn_x2),
            self.qb1,
            preact,
            cu,
            sfx,
            self.sfb1,
            (self.s_x.pts * self.p_w1).reshape(1),
        )
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
        return _MoEFn.apply(x, self.w1, self.w2, gather_idx, cu, probs_sorted, slots, off_pad, self)
