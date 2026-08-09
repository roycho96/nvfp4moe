"""Train a toy BF16-router, NVFP4-expert MoE on one B200.

The example covers calibration, differentiable routing weights, weight
refresh, stochastic rounding, and optional microbatch accumulation. It also
checks reproducibility across identical seeded runs.

    modal run examples/train_toy_moe.py
"""

import sys
from pathlib import Path

import modal

app = modal.App("nvfp4moe-train-demo")
NGC = "nvcr.io/nvidia/pytorch:26.07-py3"
LOCAL_ROOT = Path(__file__).resolve().parent.parent
vol = modal.Volume.from_name("quack-jit-cache", create_if_missing=True)
img = (
    modal.Image.from_registry(NGC, add_python=None)
    .pip_install("quack-kernels")
    .add_local_dir(str(LOCAL_ROOT / "third_party" / "quack" / "quack"), "/root/fork/quack")
    .add_local_dir(str(LOCAL_ROOT / "nvfp4moe"), "/root/proj/nvfp4moe")
)

T, D, I, E, K = 1024, 256, 128, 16, 4
STEPS = 300
ACC_STEPS, N_MB = 100, 3   # accumulation phase: 100 optimizer steps x 3 mb
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


@app.function(gpu="B200", image=img, timeout=3600, volumes={"/vol": vol},
              single_use_containers=True)
def run():
    import os

    LOG.clear()
    os.environ["PYTHONPATH"] = "/root/proj:/root/fork"
    os.environ["NVFP4MOE_QUACK_PATH"] = "/root/fork"
    os.environ["QUACK_CACHE_DIR"] = "/vol/quack_cache"
    for p in ("/root/proj", "/root/fork"):
        sys.path.insert(0, p)

    import torch
    import torch.nn.functional as F
    from nvfp4moe import MoEDispatch, MoEExpertLayer

    def make_teacher():
        """Build a frozen target whose output depends on expert selection."""
        g = torch.Generator(device="cuda").manual_seed(97)
        wr = torch.randn(D, E, device="cuda", generator=g) * D**-0.5
        tw = torch.randn(E, D, D, device="cuda", generator=g,
                         dtype=torch.float32) * D**-0.5

        def target_of(x):
            with torch.no_grad():
                tv, ti = torch.topk(torch.softmax(x.float() @ wr, -1), K, -1)
                tv = tv / tv.sum(-1, keepdim=True)
                tgt = torch.zeros(x.shape[0], D, device="cuda")
                for j in range(K):
                    for e in range(E):
                        m_ = ti[:, j] == e
                        if m_.any():
                            tgt[m_] += tv[m_, j, None] * (x[m_].float() @ tw[e])
                return tgt

        return target_of

    def make_batch(seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        return torch.randn(T, D, device="cuda", generator=g,
                           dtype=torch.float32).to(torch.bfloat16)

    def fwd_once(router, layer, disp, x, sr_seed):
        """Run one routed forward while preserving router autograd."""
        logits = router(x)
        tv, ti = torch.topk(torch.softmax(logits.float(), -1), K, -1)
        tv = tv / tv.sum(-1, keepdim=True)
        gi, cu, ps, slots = disp(ti.to(torch.int32), tv.detach())
        ps_diff = disp.differentiable_probs(tv)
        layer.sr_seed = sr_seed
        return layer(x, gi, cu, ps_diff, slots, off_pad=disp.off_pad)

    def build(accumulate=False):
        torch.manual_seed(7)
        router = torch.nn.Linear(D, E, bias=False, dtype=torch.bfloat16,
                                 device="cuda")                      # BF16
        layer = MoEExpertLayer(D, I, E, K, rht=True,
                               delayed_col_amax=True,
                               wgrad_accumulate=accumulate).cuda()
        layer.refresh_weights()
        disp = MoEDispatch(T, E, K)
        # Seed delayed activation scaling with the first batch.
        x0 = make_batch(1000)
        with torch.no_grad():
            tv, ti = torch.topk(torch.softmax(router(x0).float(), -1), K, -1)
            gi, cu, ps, slots = disp(ti.to(torch.int32),
                                     tv / tv.sum(-1, keepdim=True))
            layer.calibrate(x0, gi, cu, ps)
        return router, layer, disp

    def train_once(tag, n_mb=1, n_steps=STEPS):
        """Train with one or more microbatches per optimizer step."""
        router, layer, disp = build(accumulate=(n_mb > 1))
        target_of = make_teacher()
        opt = torch.optim.Adam(
            [router.weight, layer.w1, layer.w2], lr=2e-3)

        losses = []
        for it in range(n_steps):
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for mb in range(n_mb):
                x = make_batch(1000 + it * n_mb + mb)
                tgt = target_of(x)
                y = fwd_once(router, layer, disp, x,
                             sr_seed=5000 + it * n_mb + mb)
                loss = F.mse_loss(y.float(), tgt) / n_mb
                loss.backward()
                tot += loss.item()
            if n_mb > 1:
                layer.commit_wgrad()
            opt.step()
            layer.refresh_weights()          # masters moved -> requantize
            losses.append(tot)
            if it % 25 == 0 or it == n_steps - 1:
                log(f"  [{tag}] step {it:3d}  loss {losses[-1]:.5f}")
        return losses, router.weight.detach().clone(), \
            layer.w1.detach().clone(), layer.w2.detach().clone()

    def check_accumulation(n_mb=N_MB, n_cycles=2):
        """Compare fused wgrad accumulation with PyTorch accumulation."""
        results = {}
        for mode in ("acc", "ref"):
            router, layer, disp = build(accumulate=(mode == "acc"))
            outs, grads = [], []
            for cyc in range(n_cycles):
                router.weight.grad = None
                layer.w1.grad = None
                layer.w2.grad = None
                for mb in range(n_mb):
                    x = make_batch(2000 + cyc * n_mb + mb)
                    y = fwd_once(router, layer, disp, x,
                                 sr_seed=9000 + cyc * n_mb + mb)
                    gd = torch.Generator(device="cuda").manual_seed(
                        3000 + cyc * n_mb + mb)
                    dY = torch.randn(T, D, device="cuda", generator=gd,
                                     dtype=torch.float32).to(torch.bfloat16)
                    y.backward(dY)
                    outs.append(y.detach().clone())
                if mode == "acc":
                    layer.commit_wgrad()
                grads.append((layer.w1.grad.clone(), layer.w2.grad.clone(),
                              router.weight.grad.clone()))
            results[mode] = (outs, grads)
        oa, ga = results["acc"]
        orf, gr = results["ref"]
        y_eq = all(torch.equal(a, b) for a, b in zip(oa, orf))
        names = ("w1.grad", "w2.grad", "router.grad")
        ok = y_eq
        for cyc in range(n_cycles):
            for nm, a, b in zip(names, ga[cyc], gr[cyc]):
                e = torch.equal(a, b)
                ok = ok and e
                log(f"  [check] cycle {cyc} {nm}: "
                    f"{'bitwise EQUAL' if e else 'MISMATCH'}")
        log(f"  [check] y bitwise across all microbatches: "
            f"{'PASS' if y_eq else 'FAIL'}")
        return ok

    import statistics

    log(f"toy MoE: T={T} d={D} I={I} E={E} k={K}, {STEPS} steps, Adam, "
        f"RHT+SR, delayed_col_amax")
    l1, r1, w1a, w2a = train_once("run1")
    l2, r2, w1b, w2b = train_once("run2")

    f5, e5 = statistics.mean(l1[:5]), statistics.mean(l1[-5:])
    log(f"\nloss first5 {f5:.5f} -> last5 {e5:.5f}  ({100 * (1 - e5 / f5):.1f}% down)")
    det = (l1 == l2 and torch.equal(r1, r2) and torch.equal(w1a, w1b)
           and torch.equal(w2a, w2b))
    log(f"determinism (two full 300-step runs bitwise identical, SR on): "
        f"{'PASS' if det else 'FAIL'}")
    log("loss curve (every 10th): "
        + " ".join(f"{v:.4f}" for v in l1[::10]))
    # The target is outside the student's exact function family, so use a
    # relative loss reduction rather than an arbitrary absolute threshold.
    ok = det and e5 < 0.8 * f5

    log(f"\nmicrobatch accumulation: N_MB={N_MB}, kernel reduce-add wgrads")
    accumulation_ok = check_accumulation()
    log(f"accumulation check (kernel accumulate == autograd add_ chain, "
        f"2 cycles): {'PASS' if accumulation_ok else 'FAIL'}")
    a1 = train_once("acc1", n_mb=N_MB, n_steps=ACC_STEPS)
    a2 = train_once("acc2", n_mb=N_MB, n_steps=ACC_STEPS)
    af, ae = statistics.mean(a1[0][:5]), statistics.mean(a1[0][-5:])
    adet = (a1[0] == a2[0] and torch.equal(a1[1], a2[1])
            and torch.equal(a1[2], a2[2]) and torch.equal(a1[3], a2[3]))
    log(f"accum loss first5 {af:.5f} -> last5 {ae:.5f} "
        f"({100 * (1 - ae / af):.1f}% down over {ACC_STEPS} steps x {N_MB} mb)")
    log(f"accum determinism (two full runs bitwise, SR on): "
        f"{'PASS' if adet else 'FAIL'}")
    ok = ok and accumulation_ok and adet and ae < 0.8 * af

    vol.commit()
    return ok, "\n".join(LOG)


@app.local_entrypoint()
def main():
    ok, out = run.remote()
    p = Path(__file__).resolve().parent / "train_toy_moe_out.txt"
    p.write_text(out + "\n")
    print(f"[{'PASS' if ok else 'FAIL'} -> {p}]")
