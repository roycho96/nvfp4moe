"""Run the nvfp4moe test suite on a Modal B200.

    modal run benchmarks/modal_ci.py
"""

import subprocess
import sys
from pathlib import Path

import modal

app = modal.App("nvfp4moe-ci")
NGC = "nvcr.io/nvidia/pytorch:26.07-py3"
ROOT = Path(__file__).resolve().parent.parent
vol = modal.Volume.from_name("quack-jit-cache", create_if_missing=True)
img = (
    modal.Image.from_registry(NGC, add_python=None)
    .pip_install("quack-kernels", "pytest")
    .add_local_dir(str(ROOT / "third_party" / "quack" / "quack"), "/root/fork/quack")
    .add_local_dir(str(ROOT / "nvfp4moe"), "/root/proj/nvfp4moe")
    .add_local_dir(str(ROOT / "kernels"), "/root/proj/kernels")
    .add_local_dir(str(ROOT / "tests"), "/root/proj/tests")
)


@app.function(gpu="B200", image=img, timeout=3600, volumes={"/vol": vol})
def run():
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = "/root/proj:/root/fork"
    env["NVFP4MOE_QUACK_PATH"] = "/root/fork"
    env["QUACK_CACHE_DIR"] = "/vol/quack_cache"
    ok = True
    for t in ("tests/test_nvfp4moe_layer.py", "tests/test_rht_te_equiv.py"):
        p = subprocess.run([sys.executable, f"/root/proj/{t}"],
                           capture_output=True, text=True, timeout=2400, env=env)
        print(f"===== {t} (exit {p.returncode}) =====")
        print(p.stdout[-6000:])
        if p.returncode != 0:
            print(p.stderr[-4000:])
            ok = False
    vol.commit()
    print("CI", "PASS" if ok else "FAIL")
    return ok


@app.local_entrypoint()
def main():
    run.remote()
