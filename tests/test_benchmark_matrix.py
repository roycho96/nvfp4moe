import json

import pytest

from benchmarks import nvfp4_gemm, nvfp4_moe
from benchmarks.model_shapes import (
    FULL_ROUTINGS,
    FULL_TOKENS,
    MODEL_SHAPES,
    QUICK_ROUTINGS,
    QUICK_TOKENS,
    generate_gemm_cases,
    generate_moe_cases,
    routing_counts,
)


def test_model_registry_contains_release_matrix():
    assert tuple(MODEL_SHAPES) == (
        "qwen3_30b_a3b",
        "qwen3_235b_a22b",
        "gemma4_26b_a4b",
        "deepseek_v3_2",
        "kimi_k2_7",
        "minimax_m2",
        "llama4_scout",
    )
    expected = {
        "qwen3_30b_a3b": (2048, 768, 128, 8),
        "qwen3_235b_a22b": (4096, 1536, 128, 8),
        "gemma4_26b_a4b": (2816, 704, 128, 8),
        "deepseek_v3_2": (7168, 2048, 256, 8),
        "kimi_k2_7": (7168, 2048, 384, 8),
        "minimax_m2": (3072, 1536, 256, 8),
        "llama4_scout": (5120, 8192, 16, 1),
    }
    assert {
        key: (spec.hidden, spec.intermediate, spec.experts, spec.topk)
        for key, spec in MODEL_SHAPES.items()
    } == expected


@pytest.mark.parametrize("key", MODEL_SHAPES)
def test_projection_shapes_and_ep_math(key):
    spec = MODEL_SHAPES[key]
    assert spec.gemm_shape("fc1") == (2 * spec.padded_intermediate, spec.hidden)
    assert spec.gemm_shape("fc2") == (spec.hidden, spec.padded_intermediate)
    for ep in spec.ep_sizes:
        assert spec.local_experts(ep) * ep == spec.experts


def test_quick_and_full_gemm_case_grids():
    quick = generate_gemm_cases(("qwen3_30b_a3b",), QUICK_TOKENS, "quick", QUICK_ROUTINGS)
    assert len(quick) == 1 * 2 * 2 * 2
    assert {case.local_experts for case in quick} == {8}
    assert {case.routed_rows for case in quick} == {1024, 65536}
    assert {case.projection for case in quick} == {"fc1", "fc2"}

    full = generate_gemm_cases(("qwen3_30b_a3b",), FULL_TOKENS, "full", FULL_ROUTINGS)
    assert len(full) == 4 * 6 * 2 * 4
    assert {case.ep_size for case in full} == {1, 8, 16, 32}
    assert all(case.flops == 2 * case.routed_rows * case.n * case.k for case in full)


@pytest.mark.parametrize("routing", ("balanced", "jagged", "hotspot", "tail"))
@pytest.mark.parametrize("rows,experts", ((0, 8), (7, 8), (1024, 8), (65536, 128)))
def test_routing_counts_preserve_rows(rows, experts, routing):
    counts = routing_counts(rows, experts, routing)
    assert len(counts) == experts
    assert sum(counts) == rows
    assert min(counts) >= 0
    assert counts == routing_counts(rows, experts, routing)


def test_moe_matrix_distinguishes_synthetic_and_trace_sources():
    synthetic = generate_moe_cases(("deepseek_v3_2",), (8192,), "quick", ("balanced",), "synthetic")
    assert len(synthetic) == 1
    assert synthetic[0].ep_size == 8
    assert synthetic[0].local_experts == 32
    assert synthetic[0].routed_rows == 65536

    replay = generate_moe_cases(("deepseek_v3_2",), (8192,), "quick", ("balanced",), "trace")
    assert len(replay) == 1
    assert replay[0].source == "trace"
    assert replay[0].routing == "captured"
    assert replay[0].ep_size == 32
    assert replay[0].local_experts == 8


@pytest.mark.parametrize("key", MODEL_SHAPES)
def test_quick_synthetic_moe_has_more_experts_than_topk(key):
    case = generate_moe_cases((key,), (8192,), "quick", ("balanced",))[0]
    assert case.local_experts > case.topk


def test_gemm_list_is_cpu_safe(monkeypatch, capsys):
    monkeypatch.setattr(nvfp4_gemm, "detect_backends", dict)
    result = nvfp4_gemm.main(
        [
            "--list",
            "--models",
            "qwen3_30b_a3b",
            "--tokens",
            "8192",
            "--routing",
            "balanced",
            "--projections",
            "fc2",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["benchmark"] == "standalone_nvfp4_grouped_gemm"
    assert payload["case_count"] == 1
    assert payload["cases"][0]["routed_rows"] == 65536
    assert payload["definitions"]["prepacked"].startswith("resident NVFP4")


def test_dense_gemm_list_is_cpu_safe(capsys):
    result = nvfp4_gemm.main(
        [
            "--list",
            "--workload",
            "dense",
            "--models",
            "qwen3_30b_a3b",
            "--tokens",
            "128,8192",
            "--projections",
            "fc2",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["benchmark"] == "standalone_nvfp4_dense_gemm"
    assert payload["case_count"] == 2
    assert payload["cases"][0]["m"] == 128
    assert payload["cases"][1]["m"] == 8192
    assert payload["definitions"]["operation"] == "C = A @ B.T"


def test_moe_list_is_cpu_safe(monkeypatch, capsys):
    monkeypatch.setattr(nvfp4_moe, "detect_backends", dict)
    result = nvfp4_moe.main(
        [
            "--list",
            "--models",
            "minimax_m2",
            "--tokens",
            "8192",
            "--routing",
            "jagged",
            "--source",
            "trace",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["benchmark"] == "nvfp4_moe_layer"
    assert payload["case_count"] == 1
    assert payload["cases"][0]["source"] == "trace"
    assert payload["boundaries"]["excluded"] == "router logits and top-k selection"


def test_trace_listing_defaults_to_primary_8192_shape(monkeypatch):
    monkeypatch.setattr(nvfp4_moe, "detect_backends", dict)
    args = nvfp4_moe.build_parser().parse_args(
        ["--list", "--models", "kimi_k2_7", "--source", "trace"]
    )
    payload = nvfp4_moe.listing_payload(args)
    assert payload["case_count"] == 1
    assert payload["cases"][0]["tokens"] == 8192
    assert payload["cases"][0]["routing"] == "captured"
