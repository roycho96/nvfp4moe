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
vol = modal.Volume.from_name("nvfp4moe-jit-cache", create_if_missing=True)
img = (
    modal.Image.from_registry(NGC, add_python=None)
    .pip_install(
        "pytest",
        "apache-tvm-ffi>=0.1.12,<0.2",
        "torch-c-dlpack-ext",
        "nvidia-cutlass-dsl[cu13]==4.6.0",
    )
    .add_local_dir(str(ROOT / "nvfp4moe"), "/root/proj/nvfp4moe")
    .add_local_dir(str(ROOT / "tests"), "/root/proj/tests")
    .add_local_file(
        str(ROOT / "benchmarks" / "native_grouped_gemm_smoke.py"),
        "/root/proj/benchmarks/native_grouped_gemm_smoke.py",
    )
)


@app.function(gpu="B200", image=img, timeout=3600, volumes={"/vol": vol})
def run(
    smoke_only=False,
    native_fp32=False,
    dgrad_matrix=False,
    frontier_matrix=False,
):
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = "/root/proj"
    ok = True
    scripts = (["benchmarks/native_grouped_gemm_smoke.py"],)
    if frontier_matrix:
        scripts = (["benchmarks/native_grouped_gemm_smoke.py", "--frontier-matrix"],)
    elif dgrad_matrix:
        scripts = (["benchmarks/native_grouped_gemm_smoke.py", "--dgrad-matrix"],)
    elif native_fp32:
        scripts = (
            ["benchmarks/native_grouped_gemm_smoke.py", "--quick", "--output-dtype", "fp32"],
        )
    if not smoke_only:
        scripts += (
            ["benchmarks/native_grouped_gemm_smoke.py", "--scheduler-edge"],
            ["tests/test_nvfp4moe_layer.py"],
            ["tests/test_rht_te_equiv.py"],
        )
    for command in scripts:
        p = subprocess.run(
            [sys.executable, f"/root/proj/{command[0]}", *command[1:]],
            capture_output=True,
            text=True,
            timeout=2400,
            env=env,
            check=False,
        )
        print(f"===== {' '.join(command)} (exit {p.returncode}) =====")
        print(p.stdout[-20_000:])
        if p.returncode != 0:
            print(p.stderr[-4000:])
            ok = False
    vol.commit()
    print("CI", "PASS" if ok else "FAIL")
    return ok


@app.function(gpu="B200", image=img, timeout=3600, volumes={"/vol": vol})
def profile(case="dgrad2-qwen"):
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = "/root/proj"
    command = [
        "ncu",
        "--target-processes",
        "all",
        "--kernel-name",
        "regex:.*Sm100GroupedBlockScaledGemmKernel.*",
        "--launch-count",
        "1",
        "--section",
        "SpeedOfLight",
        "--section",
        "LaunchStats",
        "--section",
        "Occupancy",
        "--section",
        "SchedulerStats",
        "--section",
        "WarpStateStats",
        "--section",
        "InstructionStats",
        "--section",
        "MemoryWorkloadAnalysis",
        sys.executable,
        "/root/proj/benchmarks/native_grouped_gemm_smoke.py",
        "--profile-case",
        case,
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=3000,
        env=env,
        check=False,
    )
    print(result.stdout[-80_000:])
    if result.stderr:
        print(result.stderr[-8000:])
    if result.returncode != 0:
        raise RuntimeError(f"NCU exited with {result.returncode}")


@app.local_entrypoint()
def main(
    smoke_only: bool = False,
    native_fp32: bool = False,
    dgrad_matrix: bool = False,
    frontier_matrix: bool = False,
    profile_case: str = "",
):
    if profile_case:
        profile.remote(profile_case)
    else:
        run.remote(smoke_only, native_fp32, dgrad_matrix, frontier_matrix)
