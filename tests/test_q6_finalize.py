"""Q6 correctness: deterministic MoE finalize (gather-reduce) on sm100.

Gates:
  1. bitwise determinism over 20 runs (the kernel's differentiator);
  2. exact match against a fixed-order fp32 python reference (same j order,
     fp32 accumulate, single bf16 round) — ULP-0;
  3. -1 (capacity-dropped) slots contribute nothing.

    PYTHONPATH=third_party/quack python tests/test_q6_finalize.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "quack"))

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def ref_finalize(yw, slots, T, k):
    # Same contract as the kernel: fixed j order, fp32 accumulate, one bf16 round.
    d = yw.shape[1]
    acc = torch.zeros(T, d, device=yw.device, dtype=torch.float32)
    sl = slots.view(T, k)
    for j in range(k):
        s = sl[:, j].long()
        valid = s >= 0
        acc[valid] += yw[s[valid]].float()
    return acc.to(torch.bfloat16)


def main():
    if torch.cuda.get_device_capability(0)[0] != 10:
        print("SKIP: requires SM100")
        return 0
    from quack.moe_finalize import moe_finalize

    torch.manual_seed(0)
    T, k, d = 1024, 8, 2048
    M = T * k
    yw = (torch.randn(M, d, device="cuda") * 0.1).to(torch.bfloat16)
    order = torch.randperm(M, device="cuda")
    slots = torch.empty(M, dtype=torch.int32, device="cuda")
    slots[order] = torch.arange(M, dtype=torch.int32, device="cuda")
    # knock out ~10% of slots (capacity drop)
    drop = torch.rand(M, device="cuda") < 0.1
    slots[drop] = -1

    out = torch.empty(T, d, device="cuda", dtype=torch.bfloat16)
    moe_finalize(yw, slots, out, k)
    torch.cuda.synchronize()

    ref = ref_finalize(yw, slots, T, k)
    check("exact vs fixed-order fp32 reference", torch.equal(out, ref),
          f"mismatched {(out != ref).sum().item()}/{out.numel()}")

    base = out.clone()
    stable = all(
        (moe_finalize(yw, slots, out, k) or torch.equal(out, base)) for _ in range(20)
    )
    check("bitwise deterministic over 20 runs", stable)

    # all-dropped token rows must be exactly zero
    tok_dropped = (slots.view(T, k) < 0).all(1)
    if tok_dropped.any():
        check("all-dropped tokens are zero", (out[tok_dropped] == 0).all().item())

    print(f"\n{'ALL PASS' if not FAILURES else f'{len(FAILURES)} FAILURES: {FAILURES}'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
