"""NVFP4 MoE expert layer with forward, backward, and router gradients."""

import torch
from torch import nn

from .kernels.epilogue import GatedBackwardEpilogue, GatedEpilogue
from .kernels.finalize import moe_finalize, moe_finalize_bwd
from .kernels.gemm import GroupedNvfp4Gemm
from .kernels.quantize import (
    nvfp4_quantize_colwise,
    nvfp4_quantize_row_colwise,
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
    common = _GKW_2CTA if M >= 256 * L.E and L.d > 2048 and L.I > 1024 else _GKW
    return common, dgrad2, common


def _native_fc1_config(L, M):
    """Select the native FC1 tile from the average routed expert size."""
    if L.gemm_cfg == "legacy" or M < 256 * L.E:
        return _GKW
    return _GKW_2CTA


class _MoEFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        w_gate,
        w_up,
        w2,
        gather_idx,
        cu,
        probs_sorted,
        slots,
        off_pad,
        layer,
        routed,
    ):
        del w_gate, w_up, w2
        L = layer
        ctx.interleaved_w1 = L._owns_weights
        T, d = x.shape
        M = T if routed else gather_idx.numel()
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
            gather_idx=None if routed else gather_idx,
            padded_offsets=off_pad,
            te_math=True,
        )
        qx = qx_u8.view(torch.float4_e2m1fn_x2)
        sf_rows = -(-M // 128) + L.E
        # FC1 fuses the gather, gated activation, and NVFP4 post-activation.
        ascale = (L.s_x.pts * L.p_w1).reshape(1)
        q_h = L._buf("q_h", (M, I // 2), torch.float4_e2m1fn_x2)
        sf_h = L._buf("sf_h", (L.rm_max, I // 64, 32, 4, 4), torch.float8_e4m3fn, batch1=True)
        preact = torch.empty(M, 2 * I, dtype=torch.bfloat16, device=x.device)
        sf_h.zero_()
        native_fc1 = L._native_gemm(
            "fc1",
            2 * I,
            d,
            _native_fc1_config(L, M),
            output_dtype=torch.float4_e2m1fn_x2,
            activation=L.activation,
            save_preact=True,
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
            aux=preact,
        )
        # FC2 folds routing probabilities and scales into its epilogue.
        # Save a private output buffer when router gradients are required.
        need_dp = not routed and probs_sorted.requires_grad
        if routed or need_dp:
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
        if routed:
            y = yw
        else:
            y = torch.empty(T, d, device=x.device, dtype=torch.bfloat16)
            # Fixed-order combine keeps repeated runs deterministic.
            moe_finalize(
                yw,
                slots,
                y,
                L.topk,
                tile_t=2,
                n_frag=2,
                weights=probs_sorted,
            )
        ctx.save_for_backward(
            x,
            preact,
            gather_idx,
            cu,
            probs_sorted,
            slots,
            off_pad,
            *((yw,) if need_dp else ()),
        )
        ctx.need_dprobs = need_dp
        ctx.routed = routed
        ctx.layer = L
        return y

    @staticmethod
    def backward(ctx, dY):
        saved = ctx.saved_tensors
        x, preact, gi, cu, ps, slots, off_pad = saved[:7]
        yw_saved = saved[7] if ctx.need_dprobs else None
        L = ctx.layer
        routed = ctx.routed
        T, d = x.shape
        M = T if routed else gi.numel()
        I = L.I
        sf_rows = -(-M // 128) + L.E
        if off_pad is None:
            raise RuntimeError("off_pad must be supplied by the routing layer")
        mp_max = L._mp_max(M)
        n_ct = -(-M // 128) + L.E
        cw = {}
        pend = []
        dY = dY.contiguous().to(torch.bfloat16)
        # Delayed mode reuses the previous post-RHT column amax. Row scales
        # remain same-step, and the first backward seeds delayed state.
        dca = L.delayed_col_amax
        warm = dca and not L._dcol_ready
        # Gather routed output gradients and optionally collect router dots.
        n_dp = -(-d // 2048)
        if routed:
            dY_M = dY
            if dca:
                L.s_dy.update(dY_M)
                if warm:
                    L._amax2(dY_M, cu, None, L.s_dy_c, padded_offsets=off_pad)
            elif L.rht:
                L._amax2(dY_M, cu, L.s_dy, L.s_dy_c, padded_offsets=off_pad)
            else:
                L.s_dy.update(dY_M)
        else:
            dY_M = L._buf("dY_M", (M, d), torch.bfloat16)
            dp_kw = {}
            if ctx.need_dprobs:
                dp_part = L._buf("dprobs_part", (n_dp * M,), torch.float32)
                dp_kw = {"yw": yw_saved, "dot_out": dp_part}
            if dca:
                n_fb = -(-M // 8) * n_dp
                fb_part = L._buf("dca_fb_part", (n_fb,), torch.float32)
                moe_finalize_bwd(
                    dY,
                    gi,
                    ps.float().contiguous(),
                    dY_M,
                    amax_out=fb_part,
                    **dp_kw,
                )
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
        if L.rht and L.fused_row_col:
            qb = L._buf("cw_dy_q", (d, mp_max // 2), torch.uint8)
            sb = L._buf("cw_dy_sf", (d * mp_max // 16,), torch.float8_e4m3fn)
            aout = None
            if dca:
                aout = L._buf("dca_dy_part", (n_ct * (d // 128),), torch.float32)
                pend.append((L.s_dy_c, aout))
            nvfp4_quantize_row_colwise(
                dY_M,
                cu,
                L.s_dy.pair,
                L.s_dy_c.pair,
                q_dy,
                sf_dy,
                qb,
                sb,
                rounding=rd,
                seed=sv,
                amax_out=aout,
                padded_offsets=off_pad,
            )
            cw["dy"] = (qb, sb, L.s_dy_c)
        else:
            nvfp4_quantize_rowwise(
                dY_M,
                cu,
                L.s_dy.pair,
                q_dy,
                sf_dy,
                rounding=rd,
                seed=sv,
                padded_offsets=off_pad,
                te_math=True,
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
        if L.rht and L.fused_row_col:
            qb = L._buf("cw_dh_q", (2 * I, mp_max // 2), torch.uint8)
            sb = L._buf("cw_dh_sf", (2 * I * mp_max // 16,), torch.float8_e4m3fn)
            aout = None
            if dca:
                aout = L._buf("dca_dh_part", (n_ct * (2 * I // 128),), torch.float32)
                pend.append((L.s_dh_c, aout))
            nvfp4_quantize_row_colwise(
                dH,
                cu,
                L.s_dh.pair,
                L.s_dh_c.pair,
                q_dh,
                sf_dh,
                qb,
                sb,
                rounding=rd,
                seed=sv + 1,
                amax_out=aout,
                padded_offsets=off_pad,
            )
            cw["dh"] = (qb, sb, L.s_dh_c)
        else:
            nvfp4_quantize_rowwise(
                dH,
                cu,
                L.s_dh.pair,
                q_dh,
                sf_dh,
                rounding=rd,
                seed=sv + 1,
                padded_offsets=off_pad,
                te_math=True,
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
        if routed:
            dX = dX_M
        else:
            dX = torch.empty(T, d, device=x.device, dtype=torch.bfloat16)
            moe_finalize(dX_M, slots, dX, L.topk, tile_t=1, n_frag=2)
        # Columnwise quantization feeds one grouped wgrad per weight.
        # Buffers use the padded static upper bound for variable routing.
        s_hh, s_cx = L.s_hh_c, L.s_x_c
        if L.rht:
            if not dca or warm:
                L._amax2(hh, cu, None, s_hh, padded_offsets=off_pad)
                L._amax2(
                    x,
                    cu,
                    None,
                    s_cx,
                    gather_idx=None if routed else gi,
                    padded_offsets=off_pad,
                )
        else:
            s_hh.update(hh)
            s_cx.update(x)
        # Consume each quantized operand pair while its qdata and scales are hot.
        acc = L.wgrad_accumulate
        if acc:
            dW1, dW2 = L.acc_dw1, L.acc_dw2
            wg1_ = L.wg1 if L._acc_fresh else L.wg1_acc
            wg2_ = L.wg2 if L._acc_fresh else L.wg2_acc
        else:
            dW1 = torch.empty(L.E, 2 * I, d, device=x.device, dtype=torch.bfloat16)
            dW2 = torch.empty(L.E, d, I, device=x.device, dtype=torch.bfloat16)
            wg1_, wg2_ = L.wg1, L.wg2
        for name, z, F_, sc, gidx, crd, csd in (
            # Apply stochastic rounding only to gradient casts.
            ("dy", dY_M, d, L.s_dy_c if L.rht else L.s_dy, None, rd, sv + 2),
            ("hh", hh, I, s_hh, None, "rn", 0),
            ("dh", dH, 2 * I, L.s_dh_c if L.rht else L.s_dh, None, rd, sv + 3),
            ("x", x, d, s_cx, None if routed else gi, "rn", 0),
        ):
            if name in cw:
                continue
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
            if name == "hh":
                an, bn, out, wg, gs = "dy", "hh", dW2, wg2_, L._gs2
            elif name == "x":
                an, bn, out, wg, gs = "dh", "x", dW1, wg1_, L._gs1
            else:
                continue
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
            return dX, None, None, None, None, None, dprobs, None, None, None, None
        if ctx.interleaved_w1:
            return dX, dW1, None, dW2, None, None, dprobs, None, None, None, None
        return (
            dX,
            dW1[:, 0::2],
            dW1[:, 1::2],
            dW2,
            None,
            None,
            dprobs,
            None,
            None,
            None,
            None,
        )


class MoEExpertLayer(nn.Module):
    """Training-capable NVFP4 expert layer with external routing selection."""

    def __init__(
        self,
        d,
        I,
        E,
        topk,
        rht=True,
        delayed_col_amax=False,
        fused_row_col=True,
        wgrad_accumulate=False,
        gemm_cfg="auto",
        activation="swiglu",
        allocate_weights=True,
        use_dynamic_sched=True,
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
        self.fused_row_col = fused_row_col
        self.use_dynamic_sched = bool(use_dynamic_sched)
        self._dcol_ready = False
        self._dca_red = torch.empty((), dtype=torch.float32, device="cuda")
        # RHT follows the columnwise placement used for wgrad operands.
        self.rht = rht
        self._owns_weights = bool(allocate_weights)
        if self._owns_weights:
            self.w1 = nn.Parameter(torch.randn(E, 2 * I, d, dtype=torch.bfloat16) * d**-0.5)
            self.w2 = nn.Parameter(torch.randn(E, d, I, dtype=torch.bfloat16) * I**-0.5)
        else:
            self.register_parameter("w1", None)
            self.register_parameter("w2", None)
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
        if wgrad_accumulate and not self._owns_weights:
            raise ValueError("wgrad accumulation requires layer-owned weights")
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
        save_preact=False,
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
            save_preact,
            self.use_dynamic_sched,
        )
        runtime = self._native_gemms.get(key)
        if runtime is None:
            epilogue = None
            if activation is not None:
                epilogue = GatedEpilogue(activation, save_preact)
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
                use_dynamic_sched=self.use_dynamic_sched,
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
    def refresh_weights_from(self, w_gate, w_up, w2):
        """Quantize external gate, up, and down master weights."""
        expected_gate = (self.E, self.I, self.d)
        expected_w2 = (self.E, self.d, self.I)
        if tuple(w_gate.shape) != expected_gate or tuple(w_up.shape) != expected_gate:
            raise ValueError(f"gate and up weights must have shape {expected_gate}")
        if tuple(w2.shape) != expected_w2:
            raise ValueError(f"down weights must have shape {expected_w2}")
        if (
            w_gate.dtype != torch.bfloat16
            or w_up.dtype != torch.bfloat16
            or w2.dtype != torch.bfloat16
        ):
            raise ValueError("NVFP4 expert master weights must use bfloat16")
        if w_gate.device != w_up.device or w_gate.device != w2.device:
            raise ValueError("NVFP4 expert master weights must share a device")

        dev = w_gate.device
        gate_up = [
            torch.stack((w_gate[e], w_up[e]), dim=1).reshape(2 * self.I, self.d)
            for e in range(self.E)
        ]
        for nm, mats in (
            ("w1", gate_up),
            ("w2", [w2[e] for e in range(self.E)]),
            ("w2d", [w2[e].t() for e in range(self.E)]),
            ("w1t", [weight.t() for weight in gate_up]),
        ):
            amax = torch.stack([m.abs().amax() for m in mats]).amax()
            pts = (amax.float() / _DEN).clamp(min=1e-30).reshape(1).to(dev)
            q, s = _quant_expert_stack(mats, pts)
            setattr(self, {"w1": "qb1", "w2": "qb2", "w2d": "qW2d", "w1t": "qW1T"}[nm], q)
            setattr(self, {"w1": "sfb1", "w2": "sfb2", "w2d": "sW2d", "w1t": "sW1T"}[nm], s)
            setattr(self, {"w1": "p_w1", "w2": "p_w2", "w2d": "p_w2d", "w1t": "p_w1t"}[nm], pts)

    @torch.no_grad()
    def refresh_weights(self):
        """Rebuild resident NVFP4 weights from layer-owned BF16 parameters."""
        if not self._owns_weights:
            raise RuntimeError("external-weight cores must call refresh_weights_from")
        self.refresh_weights_from(self.w1[:, 0::2], self.w1[:, 1::2], self.w2)

    @torch.no_grad()
    def _calibrate(self, x, cu, gather_idx=None, off_pad=None):
        self.s_x.update(x)
        _, d = x.shape
        M = x.shape[0] if gather_idx is None else gather_idx.numel()
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
        if off_pad is None:
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
            te_math=True,
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

    @torch.no_grad()
    def calibrate(self, x, gather_idx, cu, probs_sorted=None, off_pad=None):
        """Seed activation scales from an unpermuted input."""
        del probs_sorted
        self._calibrate(x, cu, gather_idx, off_pad)

    @torch.no_grad()
    def calibrate_routed(self, x, cu, off_pad):
        """Seed activation scales from expert-major routed rows."""
        self._calibrate(x, cu, off_pad=off_pad)

    def forward_routed(self, x, w_gate, w_up, w2, cu, off_pad):
        """Run expert-major rows without dispatch or combine work."""
        if off_pad is None:
            raise ValueError("off_pad is required; reuse the routing layer's padded offsets")
        M = x.shape[0]
        self.rm_max = max(self.rm_max, -(-M // 128) + self.E)
        empty_i32 = torch.empty(0, dtype=torch.int32, device=x.device)
        empty_f32 = torch.empty(0, dtype=torch.float32, device=x.device)
        return _MoEFn.apply(
            x,
            w_gate,
            w_up,
            w2,
            empty_i32,
            cu,
            empty_f32,
            empty_i32,
            off_pad,
            self,
            True,
        )

    def forward(self, x, gather_idx, cu, probs_sorted, slots, off_pad):
        """Run the routed expert layer with dispatch-provided padded offsets."""
        if off_pad is None:
            raise ValueError("off_pad is required; pass MoEDispatch.off_pad")
        M = gather_idx.numel()
        self.rm_max = max(self.rm_max, -(-M // 128) + self.E)
        return _MoEFn.apply(
            x,
            self.w1,
            None,
            self.w2,
            gather_idx,
            cu,
            probs_sorted,
            slots,
            off_pad,
            self,
            False,
        )


class NVFP4ExpertCore(nn.Module):
    """Expert-major NVFP4 MLP that keeps master parameters in its caller.

    This is the integration boundary for training frameworks that already own
    routing, expert parallel communication, parameters, and checkpoints.
    """

    def __init__(
        self,
        d,
        I,
        E,
        *,
        rht=True,
        delayed_col_amax=False,
        gemm_cfg="auto",
        activation="swiglu",
    ):
        super().__init__()
        self.d, self.I, self.E = int(d), int(I), int(E)
        self.runtime = MoEExpertLayer(
            self.d,
            self.I,
            self.E,
            1,
            rht=rht,
            delayed_col_amax=delayed_col_amax,
            gemm_cfg=gemm_cfg,
            activation=activation,
            allocate_weights=False,
        )
        self._weights_ready = False
        self._calibrated = False
        self._weight_fingerprint = None

    def _validate_weights(self, w1, w3, w2):
        expected_w1 = (self.E, self.I, self.d)
        expected_w2 = (self.E, self.d, self.I)
        if tuple(w1.shape) != expected_w1 or tuple(w3.shape) != expected_w1:
            raise ValueError(f"w1 and w3 must have shape {expected_w1}")
        if tuple(w2.shape) != expected_w2:
            raise ValueError(f"w2 must have shape {expected_w2}")
        if any(weight.dtype != torch.bfloat16 for weight in (w1, w3, w2)):
            raise ValueError("master expert weights must use bfloat16")
        if any(weight.device != w1.device for weight in (w3, w2)):
            raise ValueError("master expert weights must share a device")

    @staticmethod
    def _fingerprint(w1, w3, w2):
        return tuple((weight.data_ptr(), weight._version) for weight in (w1, w3, w2))

    @torch.no_grad()
    def refresh_weights(self, w1, w3, w2):
        """Refresh resident NVFP4 weights after the optimizer updates masters."""
        self._validate_weights(w1, w3, w2)
        self.runtime.refresh_weights_from(w1, w3, w2)
        self._weights_ready = True
        self._weight_fingerprint = self._fingerprint(w1, w3, w2)

    @torch.no_grad()
    def calibrate(self, x, cu, off_pad):
        """Initialize activation scales from representative routed rows."""
        if off_pad is None:
            raise ValueError("off_pad is required for routed calibration")
        self.runtime.calibrate_routed(x.detach(), cu, off_pad)
        self._calibrated = True

    def invalidate_weights(self):
        """Require a weight refresh before the next forward."""
        self._weights_ready = False
        self._weight_fingerprint = None

    def reset_calibration(self):
        """Require activation calibration before the next forward."""
        self._calibrated = False

    def forward(self, x, w1, w3, w2, cu, off_pad):
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(f"routed input must have shape [M, {self.d}]")
        if not x.is_contiguous():
            raise ValueError("routed input must be contiguous")
        if x.dtype != torch.bfloat16:
            raise ValueError("routed input must use bfloat16")
        self._validate_weights(w1, w3, w2)
        if any(weight.device != x.device for weight in (w1, w3, w2)):
            raise ValueError("routed input and master expert weights must share a device")
        if cu.dtype != torch.int32 or tuple(cu.shape) != (self.E + 1,):
            raise ValueError(f"cu must be int32 with shape ({self.E + 1},)")
        if off_pad is None:
            raise ValueError("off_pad is required; reuse the routing layer's padded offsets")
        fingerprint = self._fingerprint(w1, w3, w2)
        if not self._weights_ready or fingerprint != self._weight_fingerprint:
            self.refresh_weights(w1, w3, w2)
        if not self._calibrated:
            self.calibrate(x, cu, off_pad)
        return self.runtime.forward_routed(x, w1, w3, w2, cu, off_pad)


__all__ = ["MoEExpertLayer", "NVFP4ExpertCore"]
