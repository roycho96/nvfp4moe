"""Run the 8192-token MoE expert and training benchmark on Modal B200.

The primary comparison covers the complete expert layer. SDPA appears only
in the end-to-end block context.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal


app = modal.App("moe-nvfp4-q32-real-stack")
NGC = "nvcr.io/nvidia/pytorch:26.07-py3"
ROOT = Path(__file__).resolve().parent.parent
DEEPGEMM_SHA = "559d79fb6994a58b8a15b4b93bf13ccc16edf247"
volume = modal.Volume.from_name("moe-real-stack-cache", create_if_missing=True)

base_image = (
    modal.Image.from_registry(NGC, add_python=None)
    .apt_install("git", "ninja-build")
    .pip_install(
        "transformers>=5.5,<5.8",
        "datasets>=4.0",
        "huggingface-hub>=0.34",
        "safetensors>=0.5",
        "apache-tvm-ffi>=0.1.12,<0.2",
        "torch-c-dlpack-ext",
        "einops>=0.8",
        "nvidia-cutlass-dsl[cu13]==4.6.0",
    )
)

image = (
    base_image.run_commands(
        "git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git /opt/DeepGEMM",
        f"cd /opt/DeepGEMM && git checkout {DEEPGEMM_SHA} "
        "&& git submodule update --init --recursive",
        "cd /opt/DeepGEMM && DG_JIT_CACHE_DIR=/cache/deepgemm ./install.sh",
    )
    .add_local_dir(str(ROOT / "third_party" / "quack" / "quack"), "/root/fork/quack")
    .add_local_dir(str(ROOT / "nvfp4moe"), "/root/proj/nvfp4moe")
    .add_local_dir(str(ROOT / "kernels"), "/root/proj/kernels")
    .add_local_dir(str(ROOT / "benchmarks"), "/root/proj/benchmarks")
)

capture_image = base_image.add_local_dir(
    str(ROOT / "benchmarks"), "/root/proj/benchmarks"
)


def _setup(vendored_quack=True):
    paths = ["/root/proj", "/root/proj/benchmarks"]
    if vendored_quack:
        paths.append("/root/fork")
        os.environ["NVFP4MOE_QUACK_PATH"] = "/root/fork"
    os.environ["PYTHONPATH"] = ":".join(paths)
    os.environ["QUACK_CACHE_DIR"] = "/cache/quack"
    os.environ["DG_JIT_CACHE_DIR"] = "/cache/deepgemm"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)


def _trace_path(key):
    canonical = "gemma4_26b_a4b" if key.startswith("gemma4_26b_a4b") else key
    return f"/cache/q32/traces/{canonical}_s8192.pt"


def _result_slug(keys, exhaustive=False):
    suffix = "_exhaustive" if exhaustive else ""
    return "__".join(keys) + suffix


@app.function(
    gpu="B200",
    image=capture_image,
    timeout=3600 * 4,
    memory=65536,
    volumes={"/cache": volume},
    single_use_containers=True,
)
def capture(spec_key: str):
    _setup(vendored_quack=False)
    from q32_capture import capture_trace
    from q32_specs import get_spec

    spec = get_spec(spec_key)
    output = _trace_path(spec_key)
    metadata = capture_trace(spec, 8192, "/cache/huggingface", output)
    volume.commit()
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


@app.function(
    gpu="B200",
    image=image,
    timeout=1800,
    memory=16384,
    volumes={"/cache": volume},
    single_use_containers=True,
)
def probe():
    _setup()
    import cutlass
    import deep_gemm
    import torch
    import transformer_engine
    from q32_backends import make_backend
    from q32_specs import get_spec

    trace_file = Path(_trace_path("qwen3_30b_a3b"))
    if not trace_file.exists():
        raise FileNotFoundError(trace_file)
    trace = torch.load(trace_file, map_location="cpu", weights_only=False)
    fixed = {
        key: trace[key].cuda()
        for key in (
            "expert_input",
            "topk_index",
            "topk_weight",
            "gate_up_weight",
            "down_weight",
        )
    }
    backend = make_backend("nvfp4moe", get_spec("qwen3_30b_a3b"), fixed)
    with torch.no_grad():
        y = backend(
            fixed["expert_input"], fixed["topk_index"], fixed["topk_weight"]
        )
    torch.cuda.synchronize()
    result = {
        "torch": torch.__version__,
        "transformer_engine": transformer_engine.__version__,
        "deep_gemm": deep_gemm.__version__,
        "cutlass": cutlass.__version__,
        "device": torch.cuda.get_device_name(0),
        "nvfp4moe_qwen_forward": {
            "finite": bool(torch.isfinite(y).all()),
            "shape": list(y.shape),
        },
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.function(
    gpu="B200",
    image=image,
    timeout=3600 * 8,
    memory=65536,
    volumes={"/cache": volume},
    single_use_containers=True,
)
def benchmark(spec_keys: list[str], exhaustive: bool = False):
    _setup()
    from q32_runner import run_all, save_result

    trace_paths = {key: _trace_path(key) for key in spec_keys}
    missing = [path for path in trace_paths.values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"capture traces first: {missing}")
    result = run_all(
        trace_paths,
        exhaustive=exhaustive,
        expert_backends=(
            "te_bf16",
            "te_nvfp4",
            "te_deepgemm_bf16",
            "te_deepgemm_fp8_fp4",
            "nvfp4moe",
        ),
        stack_backends=("te_bf16", "te_deepgemm_bf16", "nvfp4moe"),
        attention_backends=("sdpa",),
    )
    result["attention_scope"] = "torch_sdpa_context_only"
    slug = _result_slug(spec_keys, exhaustive)
    remote_out = f"/cache/q32/results/{slug}.json"
    Path(remote_out).parent.mkdir(parents=True, exist_ok=True)
    save_result(result, remote_out)
    volume.commit()
    return result


@app.local_entrypoint()
def main(
    models: str = "qwen3_30b_a3b,gemma4_26b_a4b_local",
    capture_only: bool = False,
    benchmark_only: bool = False,
    exhaustive: bool = False,
    probe_only: bool = False,
):
    if probe_only:
        print(json.dumps(probe.remote(), indent=2))
        return
    if capture_only and benchmark_only:
        raise ValueError("--capture-only and --benchmark-only are mutually exclusive")
    keys = [item.strip() for item in models.split(",") if item.strip()]
    if not benchmark_only:
        captured = set()
        for key in keys:  # Keep B200 jobs serial to preserve session isolation.
            canonical = "gemma4_26b_a4b" if key.startswith("gemma4_26b_a4b") else key
            if canonical in captured:
                continue
            capture.remote(key)
            captured.add(canonical)
    if capture_only:
        return
    result = benchmark.remote(keys, exhaustive)
    output = ROOT / "benchmarks" / f"q32_real_stack_{_result_slug(keys, exhaustive)}.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {output}")
    if not result["health"]["healthy"]:
        raise RuntimeError(f"B200 session health gate failed: {result['health']}")
