"""Run the LightMoE test suite on a Modal B200.

modal run benchmarks/modal_ci.py
"""

import json
import subprocess
import sys
from pathlib import Path

import modal

app = modal.App("lightmoe-ci")
NGC = "nvcr.io/nvidia/pytorch:26.07-py3"
DEEPGEMM_SHA = "559d79fb6994a58b8a15b4b93bf13ccc16edf247"
ROOT = Path(__file__).resolve().parent.parent
vol = modal.Volume.from_name("lightmoe-jit-cache", create_if_missing=True)
base_img = modal.Image.from_registry(NGC, add_python=None).pip_install(
    "pytest",
    "apache-tvm-ffi>=0.1.12,<0.2",
    "torch-c-dlpack-ext",
    "nvidia-cutlass-dsl[cu13]==4.6.0",
)
img = (
    base_img.add_local_dir(str(ROOT / "lightmoe"), "/root/proj/lightmoe")
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
    .add_local_dir(str(ROOT / "lightmoe"), "/root/proj/lightmoe")
    .add_local_dir(str(ROOT / "tests"), "/root/proj/tests")
    .add_local_dir(str(ROOT / "benchmarks"), "/root/proj/benchmarks")
)
dense_benchmark_img = (
    modal.Image.from_registry(NGC, add_python=None)
    .pip_install(
        "pytest",
        "apache-tvm-ffi>=0.1.12,<0.2",
        "torch-c-dlpack-ext",
        "nvidia-cutlass-dsl[cu13]==4.7.0",
    )
    .add_local_dir(str(ROOT / "lightmoe"), "/root/proj/lightmoe")
    .add_local_dir(str(ROOT / "benchmarks"), "/root/proj/benchmarks")
    .add_local_dir(str(ROOT / "tests"), "/root/proj/tests")
)
grouped_benchmark_img = (
    modal.Image.from_registry(NGC, add_python=None)
    .pip_install(
        "pytest",
        "apache-tvm-ffi>=0.1.12,<0.2",
        "torch-c-dlpack-ext",
        "nvidia-cutlass-dsl[cu13]==4.7.0",
        "flashinfer-python==0.6.17",
    )
    .add_local_dir(str(ROOT / "lightmoe"), "/root/proj/lightmoe")
    .add_local_dir(str(ROOT / "benchmarks"), "/root/proj/benchmarks")
    .add_local_dir(str(ROOT / "tests"), "/root/proj/tests")
)


def _distributed_args(preset):
    if preset == "smoke":
        cases = ("qwen3_30b_a3b:1:128:balanced:fwd",)
        backends, warmup, iterations = (
            "lightmoe,transformer_engine_nvfp4_fused,pytorch_bf16",
            "3",
            "10",
        )
    elif preset == "imbalanced-lightmoe":
        cases = ("qwen3_30b_a3b:128:1:imbalanced:fwd",)
        backends, warmup, iterations = "lightmoe", "1", "3"
    elif preset == "imbalanced-transformer-engine":
        cases = ("qwen3_30b_a3b:128:1:imbalanced:fwd",)
        backends, warmup, iterations = "transformer_engine_nvfp4_fused", "1", "3"
    elif preset == "imbalanced-pytorch":
        cases = ("qwen3_30b_a3b:128:1:imbalanced:fwd",)
        backends, warmup, iterations = "pytorch_bf16", "1", "3"
    elif preset == "qwen-compare":
        cases = ("qwen3_30b_a3b:128:1:imbalanced:fwd",)
        backends, warmup, iterations = (
            "lightmoe,transformer_engine_nvfp4_fused,pytorch_bf16",
            "5",
            "20",
        )
    elif preset == "training-smoke":
        cases = ("qwen3_30b_a3b:1:128:imbalanced:fwd_bwd",)
        backends, warmup, iterations = (
            "lightmoe,transformer_engine_nvfp4_fused,pytorch_bf16",
            "2",
            "5",
        )
    elif preset == "inference":
        cases = (
            "qwen3_30b_a3b:128:1:imbalanced:fwd",
            "qwen3_30b_a3b:1:8192:imbalanced:fwd",
            "deepseek_v3_2:1:2048:imbalanced:fwd",
            "kimi_k2_7:1:2048:single_expert_skew:fwd",
        )
        backends, warmup, iterations = (
            "lightmoe,transformer_engine_nvfp4_fused,pytorch_bf16",
            "3",
            "20",
        )
    elif preset == "inference-extended":
        cases = (
            "qwen3_235b_a22b:1:2048:imbalanced:fwd",
            "minimax_m2:1:2048:imbalanced:fwd",
            "llama4_scout:1:2048:alignment_stress:fwd",
        )
        backends, warmup, iterations = (
            "lightmoe,transformer_engine_nvfp4_fused,pytorch_bf16",
            "3",
            "20",
        )
    elif preset == "training":
        cases = (
            "qwen3_30b_a3b:1:8192:balanced:fwd_bwd",
            "qwen3_30b_a3b:1:8192:imbalanced:fwd_bwd",
            "deepseek_v3_2:1:2048:imbalanced:fwd_bwd",
        )
        backends, warmup, iterations = (
            "lightmoe,transformer_engine_nvfp4_fused,pytorch_bf16",
            "3",
            "10",
        )
    else:
        raise ValueError(
            "distributed preset must be smoke, imbalanced-lightmoe, "
            "imbalanced-transformer-engine, imbalanced-pytorch, qwen-compare, "
            "training-smoke, inference, inference-extended, or training"
        )
    args = [
        "/root/proj/benchmarks/distributed_ep.py",
        "--backends",
        backends,
        "--warmup",
        warmup,
        "--iterations",
        iterations,
        "--stabilize-ms",
        "1000",
    ]
    for case in cases:
        args.extend(("--case", case))
    return args


@app.function(gpu="B200:8", image=grouped_benchmark_img, timeout=21600, volumes={"/vol": vol})
def benchmark_distributed(preset="smoke"):
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = "/root/proj:/root/proj/benchmarks"
    env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    env["NCCL_NET"] = "Socket"
    env["NCCL_NET_PLUGIN"] = "none"
    env["PYTHONUNBUFFERED"] = "1"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=8",
        *_distributed_args(preset),
    ]
    result = subprocess.run(
        command,
        timeout=21000,
        env=env,
        cwd="/root/proj",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"distributed benchmark exited with {result.returncode}")


@app.function(gpu="B200", image=img, timeout=3600, volumes={"/vol": vol})
def run(
    smoke_only=False,
    lightmoe_fp32=False,
    dgrad_matrix=False,
    frontier_matrix=False,
    benchmark_smoke=False,
    inference_only=False,
):
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = "/root/proj"
    ok = True
    scripts = (["benchmarks/grouped_gemm_smoke.py"],)
    if inference_only:
        scripts = (["-m", "pytest", "-q", "tests/test_inference.py"],)
        smoke_only = True
    elif benchmark_smoke:
        scripts = (
            [
                "benchmarks/nvfp4_gemm.py",
                "--models",
                "qwen3_30b_a3b",
                "--tokens",
                "128",
                "--routing",
                "imbalanced",
                "--projections",
                "down",
                "--backends",
                "lightmoe,transformer_engine_nvfp4",
                "--mode",
                "both",
                "--iterations",
                "20",
            ],
            [
                "benchmarks/nvfp4_gemm.py",
                "--models",
                "qwen3_30b_a3b",
                "--tokens",
                "128",
                "--routing",
                "imbalanced",
                "--projections",
                "down",
                "--backends",
                "lightmoe",
                "--mode",
                "both",
                "--direction",
                "dgrad",
                "--iterations",
                "20",
            ],
            [
                "benchmarks/nvfp4_gemm.py",
                "--models",
                "qwen3_30b_a3b",
                "--tokens",
                "128",
                "--routing",
                "imbalanced",
                "--projections",
                "down",
                "--backends",
                "lightmoe",
                "--mode",
                "both",
                "--direction",
                "wgrad",
                "--iterations",
                "20",
            ],
            [
                "benchmarks/moe.py",
                "--models",
                "qwen3_30b_a3b",
                "--tokens",
                "128",
                "--routing",
                "imbalanced",
                "--backends",
                "lightmoe,transformer_engine_nvfp4,pytorch_bf16",
                "--scope",
                "full-layer",
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
        scripts = (["benchmarks/grouped_gemm_smoke.py", "--frontier-matrix"],)
    elif dgrad_matrix:
        scripts = (["benchmarks/grouped_gemm_smoke.py", "--dgrad-matrix"],)
    elif lightmoe_fp32:
        scripts = (["benchmarks/grouped_gemm_smoke.py", "--quick", "--output-dtype", "fp32"],)
    if not smoke_only:
        scripts += (
            ["benchmarks/grouped_gemm_smoke.py", "--scheduler-edge"],
            ["-m", "pytest", "-q"],
            ["tests/test_training_gpu.py"],
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


@app.function(gpu="B200", image=grouped_benchmark_img, timeout=7200, volumes={"/vol": vol})
def profile(case="dgrad2-qwen"):
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = "/root/proj"
    profile_sections = (
        "SpeedOfLight",
        "LaunchStats",
        "Occupancy",
        "SchedulerStats",
        "WarpStateStats",
        "InstructionStats",
        "SourceCounters",
        "MemoryWorkloadAnalysis",
    )
    if case in ("training-lightmoe", "training-transformer-engine"):
        backend = "lightmoe" if case == "training-lightmoe" else "transformer_engine_nvfp4_fused"
        kernel_name = (
            "regex:.*Sm100GroupedBlockScaledGemmKernel.*"
            if case == "training-lightmoe"
            else "regex:.*nvjet_sm100.*"
        )
        launch_count = "1"
        nvtx_range = "lightmoe_training_profile/"
        targets = [
            [
                sys.executable,
                "/root/proj/benchmarks/moe.py",
                "--suite",
                "quick",
                "--models",
                "deepseek_v3_2",
                "--tokens",
                "8192",
                "--backends",
                backend,
                "--scope",
                "full-layer",
                "--pass",
                "fwd_bwd",
                "--routing",
                "balanced",
                "--interleave-training",
                "--max-cases",
                "1",
                "--profile-nvtx-arm",
                backend,
            ]
        ]
    elif case.startswith("dense-"):
        kernel_name = "regex:.*Sm100BlockScaledPersistentDenseGemmKernel.*"
        launch_count = "1"
        nvtx_range = "lightmoe_profile/"
        targets = [
            [
                sys.executable,
                "/root/proj/benchmarks/grouped_gemm_smoke.py",
                "--profile-case",
                case,
            ]
        ]
    else:
        kernel_name = "regex:.*Sm100GroupedBlockScaledGemmKernel.*"
        launch_count = "1"
        nvtx_range = "lightmoe_profile/"
        targets = [
            [
                sys.executable,
                "/root/proj/benchmarks/grouped_gemm_smoke.py",
                "--profile-case",
                case,
            ]
        ]
    profile_targets = [(kernel_name, nvtx_range, 0, target) for target in targets]
    for target_kernel, target_nvtx, launch_skip, target in profile_targets:
        command = [
            "ncu",
            "--clock-control",
            "none",
            "--target-processes",
            "all",
            "--nvtx",
            "--nvtx-include",
            target_nvtx,
            "--kernel-name",
            target_kernel,
            "--launch-count",
            launch_count,
        ]
        command.extend(("--print-details", "all"))
        for section in profile_sections:
            command.extend(("--section", section))
        if launch_skip:
            command.extend(("--launch-skip", str(launch_skip)))
        command.extend(target)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=3000,
            env=env,
            check=False,
        )
        print(f"===== {' '.join(target)} =====")
        print(result.stdout[-80_000:])
        if result.stderr:
            print(result.stderr[-8000:])
        if result.returncode != 0:
            raise RuntimeError(f"NCU exited with {result.returncode}")


@app.function(
    gpu="B200",
    image=grouped_benchmark_img,
    timeout=14400,
    volumes={"/vol": vol},
    single_use_containers=True,
)
def benchmark_inference(preset="smoke"):
    import os

    stabilize = "1000"
    if preset == "smoke":
        cases = ("qwen3_30b_a3b:128:16:balanced",)
        warmup, iterations = "3", "20"
    elif preset == "kimi":
        cases = ("kimi_k2_7:2048:48:single_expert_skew",)
        warmup, iterations = "10", "30"
        stabilize = "15000"
    elif preset == "gemma":
        cases = ("gemma4_26b_a4b:2048:16:balanced",)
        warmup, iterations = "3", "30"
    elif preset == "decode-qwen":
        cases = ("qwen3_30b_a3b:8:16:single_expert_skew",)
        warmup, iterations = "10", "50"
        stabilize = "5000"
    elif preset == "decode":
        cases = (
            "qwen3_30b_a3b:1:16:balanced",
            "qwen3_30b_a3b:8:16:single_expert_skew",
            "deepseek_v3_2:1:32:empty",
            "deepseek_v3_2:32:32:balanced",
            "kimi_k2_7:1:48:single_expert_skew",
        )
        warmup, iterations = "3", "30"
    elif preset == "full":
        cases = (
            "qwen3_30b_a3b:128:16:balanced",
            "qwen3_30b_a3b:8192:16:balanced",
            "qwen3_235b_a22b:2048:16:single_expert_skew",
            "gemma4_26b_a4b:2048:16:balanced",
            "deepseek_v3_2:2048:32:balanced",
            "kimi_k2_7:2048:48:single_expert_skew",
            "minimax_m2:2048:32:empty",
            "llama4_scout:2048:2:balanced",
        )
        warmup, iterations = "3", "20"
    else:
        raise ValueError(
            "inference preset must be smoke, kimi, gemma, decode-qwen, decode, or full"
        )
    command = [
        sys.executable,
        "/root/proj/benchmarks/inference_moe.py",
        "--warmup",
        warmup,
        "--iterations",
        iterations,
        "--stabilize-ms",
        stabilize,
    ]
    for case in cases:
        command.extend(("--case", case))
    env = dict(os.environ)
    env["PYTHONPATH"] = "/root/proj:/root/proj/benchmarks"
    env["FLASHINFER_WORKSPACE_BASE"] = "/vol/flashinfer"
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=13800,
        env=env,
        cwd="/root/proj",
        check=False,
    )
    print(result.stdout[-200_000:])
    if result.stderr:
        print(result.stderr[-20_000:])
    if result.returncode != 0:
        raise RuntimeError(f"inference benchmark exited with {result.returncode}")
    vol.commit()


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
            "imbalanced",
            "--backends",
            "lightmoe,transformer_engine_nvfp4",
            "--mode",
            "dynamic",
            "--iterations",
            "10",
        ]
    elif kind == "moe-training":
        command = [
            "benchmarks/moe.py",
            "--models",
            "qwen3_30b_a3b,deepseek_v3_2,kimi_k2_7,minimax_m2",
            "--tokens",
            "8192",
            "--routing",
            "imbalanced",
            "--backends",
            "lightmoe,transformer_engine_nvfp4_fused,deepgemm_bf16,pytorch_bf16",
            "--scope",
            "full-layer",
            "--pass",
            "fwd_bwd",
            "--warmup",
            "2",
            "--iterations",
            "5",
            "--interleave-training",
        ]
    elif kind == "moe-forward-heavy":
        command = [
            "benchmarks/moe.py",
            "--models",
            "gemma4_26b_a4b,deepseek_v3_2,kimi_k2_7",
            "--tokens",
            "8192",
            "--routing",
            "imbalanced",
            "--backends",
            ("lightmoe,transformer_engine_nvfp4,deepgemm_bf16,deepgemm_fp8_fp4,pytorch_bf16"),
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
            "benchmarks/moe.py",
            "--models",
            models,
            "--tokens",
            "8192",
            "--routing",
            "imbalanced",
            "--backends",
            ("lightmoe,transformer_engine_nvfp4,deepgemm_bf16,deepgemm_fp8_fp4,pytorch_bf16"),
            "--scope",
            "full-layer",
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


@app.function(gpu="B200", image=dense_benchmark_img, timeout=14400, volumes={"/vol": vol})
def benchmark_dense(preset="smoke"):
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = "/root/proj"
    if preset == "test":
        command = [sys.executable, "-m", "pytest", "-q"]
    elif preset == "smoke":
        models = "qwen3_30b_a3b"
        rows = "128,512"
        projections = "down"
        iterations = "5"
    elif preset == "full":
        models = "qwen3_30b_a3b,qwen3_235b_a22b,deepseek_v3_2,llama4_scout"
        rows = "128,512,2048,8192"
        projections = "gate_up,down"
        iterations = "20"
        mode = "prepacked"
        backends = "lightmoe,cublaslt"
    elif preset == "dynamic":
        models = "qwen3_235b_a22b,deepseek_v3_2,llama4_scout"
        rows = "512,2048,8192"
        projections = "gate_up,down"
        iterations = "10"
        mode = "dynamic"
        backends = "lightmoe,cublaslt"
    elif preset == "focused":
        models = "qwen3_30b_a3b,deepseek_v3_2"
        rows = "8192"
        projections = "down"
        iterations = "20"
        mode = "prepacked"
        backends = "lightmoe,lightmoe_grouped,cublaslt"
    elif preset == "core-compare":
        models = "deepseek_v3_2"
        rows = "8192"
        projections = "gate_up"
        iterations = "30"
        mode = "prepacked"
        backends = "lightmoe,lightmoe_grouped,cublaslt"
    elif preset == "recheck":
        models = "qwen3_30b_a3b"
        rows = "2048"
        projections = "down"
        iterations = "20"
        mode = "prepacked"
        backends = "lightmoe,cublaslt"
    else:
        raise ValueError(
            "dense preset must be test, smoke, focused, core-compare, recheck, full, or dynamic"
        )
    if preset != "test":
        if preset == "smoke":
            mode = "prepacked"
            backends = "lightmoe,cublaslt"
        command = [
            sys.executable,
            "/root/proj/benchmarks/nvfp4_gemm.py",
            "--workload",
            "dense",
            "--suite",
            "full",
            "--models",
            models,
            "--tokens",
            rows,
            "--projections",
            projections,
            "--backends",
            backends,
            "--mode",
            mode,
            "--warmup",
            "3",
            "--iterations",
            iterations,
        ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=13800,
        env=env,
        cwd="/root/proj",
        check=False,
    )
    print(result.stdout[-200_000:])
    rows = []
    repeats = {}
    for line in result.stdout.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == "repeat":
            repeats[(record["case"]["label"], record["mode"])] = record
        elif record.get("event") == "result" and record.get("status") == "ok":
            rows.append(record)
    summary = []
    for record in rows:
        case = record["case"]
        item = {
            "label": case["label"],
            "backend": record["backend"],
            "mode": record["mode"],
            "p50_us": 1000 * record["timing"]["event_ms_p50"],
            "iqr_us": 1000 * record["timing"]["event_ms_iqr"],
            "logical_tflops": record["timing"].get("logical_tflops"),
            "dense_fp4_spec_peak_pct": record["timing"].get("dense_fp4_spec_peak_pct"),
            "health_valid": record["timing"]["health_valid"],
        }
        if "out_timing" in record:
            item["out_p50_us"] = 1000 * record["out_timing"]["event_ms_p50"]
        if "config" in record:
            item["config"] = record["config"]
        repeat = repeats.get((case["label"], record["mode"]))
        if record["backend"] == "lightmoe" and repeat is not None:
            item["initial_to_repeat_median_deviation"] = repeat[
                "initial_to_repeat_median_deviation"
            ]
            item["repeat_deviation_valid"] = repeat["repeat_deviation_valid"]
        summary.append(item)
    print(json.dumps({"event": "dense_summary", "rows": summary}), flush=True)
    if result.stderr:
        print(result.stderr[-20_000:])
    if result.returncode != 0:
        raise RuntimeError(f"dense benchmark exited with {result.returncode}")
    vol.commit()


@app.function(gpu="B200", image=grouped_benchmark_img, timeout=14400, volumes={"/vol": vol})
def benchmark_grouped(preset="smoke"):
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = "/root/proj"
    if preset == "dual-quant":
        command = [sys.executable, "-m", "pytest", "-q", "tests/test_dual_quantize.py"]
    elif preset == "api":
        command = [
            sys.executable,
            "/root/proj/benchmarks/api_overhead.py",
            "--warmup",
            "5",
            "--iterations",
            "30",
            "--stabilize-ms",
            "1000",
            "--repeats",
            "8",
        ]
    elif preset == "layer-test":
        command = [sys.executable, "/root/proj/tests/test_training_gpu.py"]
    elif preset == "training-tiles":
        command = [
            sys.executable,
            "/root/proj/benchmarks/grouped_gemm_smoke.py",
            "--training-tile-matrix",
        ]
    elif preset in {
        "training-smoke",
        "training-qwen",
        "training-focused",
        "training-profile",
    }:
        if preset == "training-smoke":
            models = "qwen3_30b_a3b"
            tokens = "128"
            routing = "imbalanced"
            warmup = "1"
            iterations = "5"
            backends = "lightmoe,transformer_engine_nvfp4_fused,torchao_mxfp8,pytorch_bf16"
        elif preset == "training-qwen":
            models = "qwen3_30b_a3b"
            tokens = "8192"
            routing = "balanced,imbalanced"
            warmup = "3"
            iterations = "10"
            backends = "lightmoe,transformer_engine_nvfp4_fused,pytorch_bf16"
        elif preset == "training-focused":
            models = "qwen3_30b_a3b,deepseek_v3_2,kimi_k2_7,minimax_m2"
            tokens = "8192"
            routing = "balanced,imbalanced"
            warmup = "3"
            iterations = "10"
            backends = "lightmoe,transformer_engine_nvfp4_fused,pytorch_bf16"
        else:
            models = "deepseek_v3_2"
            tokens = "8192"
            routing = "balanced"
            warmup = "5"
            iterations = "20"
            backends = "lightmoe,transformer_engine_nvfp4_fused"
        command = [
            sys.executable,
            "/root/proj/benchmarks/moe.py",
            "--models",
            models,
            "--tokens",
            tokens,
            "--routing",
            routing,
            "--backends",
            backends,
            "--scope",
            "full-layer",
            "--pass",
            "fwd_bwd",
            "--warmup",
            warmup,
            "--iterations",
            iterations,
            "--interleave-training",
        ]
        if preset == "training-profile":
            command.append("--profile-kernels")
    elif preset == "smoke":
        command = [
            sys.executable,
            "/root/proj/benchmarks/nvfp4_gemm.py",
            "--models",
            "qwen3_30b_a3b",
            "--tokens",
            "128",
            "--routing",
            "balanced,imbalanced",
            "--projections",
            "down",
            "--backends",
            "lightmoe,flashinfer_cutedsl,torch_scaled_grouped_mm",
            "--warmup",
            "2",
            "--iterations",
            "5",
        ]
    elif preset in {"training-dgrad", "training-wgrad"}:
        command = [
            sys.executable,
            "/root/proj/benchmarks/nvfp4_gemm.py",
            "--models",
            "qwen3_30b_a3b,deepseek_v3_2,kimi_k2_7,minimax_m2",
            "--tokens",
            "8192",
            "--routing",
            "balanced,imbalanced",
            "--projections",
            "gate_up,down",
            "--backends",
            "lightmoe",
            "--mode",
            "both",
            "--direction",
            "dgrad" if preset == "training-dgrad" else "wgrad",
            "--warmup",
            "3",
            "--iterations",
            "10",
        ]
    elif preset in {"focused", "full", "long-hidden"}:
        models = (
            "deepseek_v3_2,kimi_k2_7"
            if preset == "long-hidden"
            else (
                "qwen3_30b_a3b,deepseek_v3_2,kimi_k2_7,minimax_m2"
                if preset == "focused"
                else (
                    "qwen3_30b_a3b,qwen3_235b_a22b,gemma4_26b_a4b,deepseek_v3_2,"
                    "kimi_k2_7,minimax_m2,llama4_scout"
                )
            )
        )
        command = [
            sys.executable,
            "/root/proj/benchmarks/nvfp4_gemm.py",
            "--models",
            models,
            "--tokens",
            "8192",
            "--routing",
            "balanced,imbalanced",
            "--projections",
            "gate_up,down",
            "--backends",
            "lightmoe,flashinfer_cutedsl,torch_scaled_grouped_mm",
            "--warmup",
            "3",
            "--iterations",
            "20",
            "--stabilize-ms",
            "1000",
        ]
    elif preset == "recheck":
        command = [
            sys.executable,
            "/root/proj/benchmarks/nvfp4_gemm.py",
            "--models",
            "qwen3_235b_a22b,deepseek_v3_2,llama4_scout",
            "--tokens",
            "8192",
            "--routing",
            "balanced,imbalanced",
            "--projections",
            "gate_up",
            "--backends",
            "lightmoe,flashinfer_cutedsl,torch_scaled_grouped_mm",
            "--warmup",
            "5",
            "--iterations",
            "30",
        ]
    else:
        raise ValueError(
            "grouped preset must be training-smoke, training-qwen, training-focused, smoke, "
            "training-tiles, "
            "training-dgrad, training-wgrad, recheck, "
            "focused, full, long-hidden, dual-quant, "
            "api, or layer-test"
        )
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=13800,
        env=env,
        cwd="/root/proj",
        check=False,
    )
    print(result.stdout[-200_000:])
    rows = []
    repeats = []
    for line in result.stdout.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        if record.get("event") == "repeat":
            repeats.append(record)
        elif record.get("event") == "result" and record.get("status") == "ok":
            rows.append(
                {
                    "label": record["case"]["label"],
                    "backend": record["backend"],
                    "p50_us": 1000 * record["timing"]["event_ms_p50"],
                    "iqr_us": 1000 * record["timing"].get("event_ms_iqr", 0.0),
                    "logical_tflops": record["timing"].get("logical_tflops"),
                    "dense_fp4_spec_peak_pct": record["timing"].get("dense_fp4_spec_peak_pct"),
                    "health_valid": record["timing"].get("health_valid"),
                    "routing_statistics": record["case"].get("routing_statistics"),
                    "cosine": record.get("sample_cosine"),
                    "config": record.get("config"),
                }
            )
    if rows:
        by_case = {}
        for row in rows:
            by_case.setdefault(row["label"], {})[row["backend"]] = row
        repeat_by_case = {repeat["case"]["label"]: repeat for repeat in repeats}
        release_rows = []
        rejected = []
        for label, backends_by_name in by_case.items():
            lightmoe = backends_by_name.get("lightmoe")
            if lightmoe is None:
                continue
            repeat = repeat_by_case.get(label)
            retained = all(row["health_valid"] for row in backends_by_name.values()) and (
                repeat is None or repeat["repeat_deviation_valid"]
            )
            if not retained:
                rejected.append(
                    {
                        "label": label,
                        "initial_to_repeat_median_deviation": (
                            None if repeat is None else repeat["initial_to_repeat_median_deviation"]
                        ),
                        "reason": "timing-health or 5% repeat-deviation check failed",
                    }
                )
                continue
            release_rows.append(
                {
                    "label": label,
                    "lightmoe_us": lightmoe["p50_us"],
                    "lightmoe_iqr_us": lightmoe["iqr_us"],
                    "lightmoe_logical_tflops": lightmoe["logical_tflops"],
                    "lightmoe_dense_fp4_spec_peak_pct": lightmoe["dense_fp4_spec_peak_pct"],
                    "flashinfer_us": backends_by_name.get("flashinfer_cutedsl", {}).get("p50_us"),
                    "pytorch_us": backends_by_name.get("torch_scaled_grouped_mm", {}).get("p50_us"),
                    "initial_to_repeat_median_deviation": (
                        None if repeat is None else repeat["initial_to_repeat_median_deviation"]
                    ),
                    "routing_statistics": lightmoe["routing_statistics"],
                    "config": lightmoe["config"],
                }
            )
        print(
            json.dumps(
                {
                    "event": "grouped_release_summary",
                    "quality": {
                        "result_rows": len(rows),
                        "health_failures": sum(not row["health_valid"] for row in rows),
                        "min_sample_cosine": min(row["cosine"] for row in rows),
                        "repeat_measurements": len(repeats),
                        "repeat_deviation_failures": sum(
                            not repeat["repeat_deviation_valid"] for repeat in repeats
                        ),
                        "max_abs_initial_to_repeat_median_deviation": max(
                            (
                                abs(repeat["initial_to_repeat_median_deviation"])
                                for repeat in repeats
                            ),
                            default=None,
                        ),
                    },
                    "rows": release_rows,
                    "rejected": rejected,
                }
            ),
            flush=True,
        )
    if result.stderr:
        print(result.stderr[-20_000:])
    if result.returncode != 0:
        raise RuntimeError(f"grouped benchmark exited with {result.returncode}")


@app.local_entrypoint()
def main(
    smoke_only: bool = False,
    lightmoe_fp32: bool = False,
    dgrad_matrix: bool = False,
    frontier_matrix: bool = False,
    benchmark_smoke: bool = False,
    matrix: str = "",
    dense: str = "",
    grouped: str = "",
    profile_case: str = "",
):
    if grouped:
        benchmark_grouped.remote(grouped)
    elif dense:
        benchmark_dense.remote(dense)
    elif matrix:
        benchmark_matrix.remote(matrix)
    elif profile_case:
        profile.remote(profile_case)
    else:
        run.remote(smoke_only, lightmoe_fp32, dgrad_matrix, frontier_matrix, benchmark_smoke)
