"""Run the nvfp4moe test suite on a Modal B200.

modal run benchmarks/modal_ci.py
"""

import subprocess
import sys
from pathlib import Path

import modal

app = modal.App("nvfp4moe-ci")
NGC = "nvcr.io/nvidia/pytorch:26.07-py3"
DEEPGEMM_SHA = "559d79fb6994a58b8a15b4b93bf13ccc16edf247"
ROOT = Path(__file__).resolve().parent.parent
vol = modal.Volume.from_name("nvfp4moe-jit-cache", create_if_missing=True)
base_img = modal.Image.from_registry(NGC, add_python=None).pip_install(
    "pytest",
    "apache-tvm-ffi>=0.1.12,<0.2",
    "torch-c-dlpack-ext",
    "nvidia-cutlass-dsl[cu13]==4.6.0",
)
img = (
    base_img.add_local_dir(str(ROOT / "nvfp4moe"), "/root/proj/nvfp4moe")
    .add_local_dir(str(ROOT / "tests"), "/root/proj/tests")
    .add_local_dir(str(ROOT / "benchmarks"), "/root/proj/benchmarks")
)
benchmark_img = (
    base_img.apt_install("git", "ninja-build")
    .run_commands(
        "git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git /opt/DeepGEMM",
        f"cd /opt/DeepGEMM && git checkout {DEEPGEMM_SHA} "
        "&& git submodule update --init --recursive",
        "cd /opt/DeepGEMM && DG_JIT_CACHE_DIR=/vol/deepgemm ./install.sh",
    )
    .add_local_dir(str(ROOT / "nvfp4moe"), "/root/proj/nvfp4moe")
    .add_local_dir(str(ROOT / "tests"), "/root/proj/tests")
    .add_local_dir(str(ROOT / "benchmarks"), "/root/proj/benchmarks")
)


@app.function(gpu="B200", image=img, timeout=3600, volumes={"/vol": vol})
def run(
    smoke_only=False,
    native_fp32=False,
    dgrad_matrix=False,
    frontier_matrix=False,
    benchmark_smoke=False,
):
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = "/root/proj"
    ok = True
    scripts = (["benchmarks/native_grouped_gemm_smoke.py"],)
    if benchmark_smoke:
        scripts = (
            [
                "benchmarks/nvfp4_gemm.py",
                "--models",
                "qwen3_30b_a3b",
                "--tokens",
                "128",
                "--routing",
                "jagged",
                "--projections",
                "fc2",
                "--backends",
                "native,te_nvfp4",
                "--mode",
                "both",
                "--iterations",
                "3",
            ],
            [
                "benchmarks/nvfp4_gemm.py",
                "--models",
                "qwen3_30b_a3b",
                "--tokens",
                "128",
                "--routing",
                "jagged",
                "--projections",
                "fc2",
                "--backends",
                "native",
                "--mode",
                "both",
                "--direction",
                "dgrad",
                "--iterations",
                "3",
            ],
            [
                "benchmarks/nvfp4_gemm.py",
                "--models",
                "qwen3_30b_a3b",
                "--tokens",
                "128",
                "--routing",
                "jagged",
                "--projections",
                "fc2",
                "--backends",
                "native",
                "--mode",
                "both",
                "--direction",
                "wgrad",
                "--iterations",
                "3",
            ],
            [
                "benchmarks/nvfp4_moe.py",
                "--models",
                "qwen3_30b_a3b",
                "--tokens",
                "128",
                "--routing",
                "jagged",
                "--backends",
                "native,te_nvfp4,torch_bf16",
                "--scope",
                "both",
                "--pass",
                "fwd",
                "--warmup",
                "1",
                "--iterations",
                "3",
            ],
        )
        smoke_only = True
    elif frontier_matrix:
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
            ["-m", "pytest", "-q", "tests/test_ops.py"],
            ["tests/test_nvfp4moe_layer.py"],
            ["tests/test_rht_te_equiv.py"],
        )
    for command in scripts:
        argv = (
            [sys.executable, *command]
            if command[0] == "-m"
            else [sys.executable, f"/root/proj/{command[0]}", *command[1:]]
        )
        p = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=2400,
            env=env,
            cwd="/root/proj",
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


@app.function(gpu="B200", image=benchmark_img, timeout=14400, volumes={"/vol": vol})
def benchmark_matrix(kind="moe-forward"):
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = "/root/proj:/root/proj/benchmarks"
    env["DG_JIT_CACHE_DIR"] = "/vol/deepgemm"
    models = (
        "qwen3_30b_a3b,qwen3_235b_a22b,gemma4_26b_a4b,deepseek_v3_2,"
        "kimi_k2_7,minimax_m2,llama4_scout"
    )
    if kind == "gemm":
        command = [
            "benchmarks/nvfp4_gemm.py",
            "--models",
            models,
            "--tokens",
            "8192",
            "--routing",
            "jagged",
            "--backends",
            "native,te_nvfp4",
            "--mode",
            "dynamic",
            "--iterations",
            "10",
        ]
    elif kind == "moe-training":
        command = [
            "benchmarks/nvfp4_moe.py",
            "--models",
            "qwen3_30b_a3b,deepseek_v3_2,kimi_k2_7,minimax_m2",
            "--tokens",
            "8192",
            "--routing",
            "jagged",
            "--backends",
            "native,te_nvfp4,deepgemm_bf16",
            "--scope",
            "full-layer",
            "--pass",
            "fwd_bwd",
            "--warmup",
            "2",
            "--iterations",
            "5",
        ]
    elif kind == "moe-forward-heavy":
        command = [
            "benchmarks/nvfp4_moe.py",
            "--models",
            "gemma4_26b_a4b,deepseek_v3_2,kimi_k2_7",
            "--tokens",
            "8192",
            "--routing",
            "jagged",
            "--backends",
            "native,te_nvfp4,deepgemm_bf16,deepgemm_fp8_fp4,torch_bf16",
            "--scope",
            "full-layer",
            "--pass",
            "fwd",
            "--warmup",
            "3",
            "--iterations",
            "10",
        ]
    elif kind == "moe-forward":
        command = [
            "benchmarks/nvfp4_moe.py",
            "--models",
            models,
            "--tokens",
            "8192",
            "--routing",
            "jagged",
            "--backends",
            "native,te_nvfp4,deepgemm_bf16,deepgemm_fp8_fp4,torch_bf16",
            "--scope",
            "both",
            "--pass",
            "fwd",
            "--warmup",
            "3",
            "--iterations",
            "10",
        ]
    else:
        raise ValueError("kind must be gemm, moe-forward, moe-forward-heavy, or moe-training")
    result = subprocess.run(
        [sys.executable, f"/root/proj/{command[0]}", *command[1:]],
        capture_output=True,
        text=True,
        timeout=13800,
        env=env,
        check=False,
    )
    print(result.stdout[-160_000:])
    if result.stderr:
        print(result.stderr[-20_000:])
    if result.returncode != 0:
        raise RuntimeError(f"benchmark matrix exited with {result.returncode}")
    vol.commit()


@app.local_entrypoint()
def main(
    smoke_only: bool = False,
    native_fp32: bool = False,
    dgrad_matrix: bool = False,
    frontier_matrix: bool = False,
    benchmark_smoke: bool = False,
    matrix: str = "",
    profile_case: str = "",
):
    if matrix:
        benchmark_matrix.remote(matrix)
    elif profile_case:
        profile.remote(profile_case)
    else:
        run.remote(smoke_only, native_fp32, dgrad_matrix, frontier_matrix, benchmark_smoke)
