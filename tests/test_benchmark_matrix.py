import json

import pytest

from benchmarks import moe, nvfp4_gemm
from benchmarks.model_shapes import (
    FULL_ROUTINGS,
    FULL_TOKENS,
    MODEL_SHAPES,
    QUICK_ROUTINGS,
    QUICK_TOKENS,
    generate_gemm_cases,
    generate_moe_cases,
    parse_models,
    routing_counts,
)


def test_model_registry_contains_release_matrix():
    assert tuple(MODEL_SHAPES) == (
        "qwen3_5_35b_a3b",
        "qwen3_5_397b_a17b",
        "deepseek_v4_flash",
        "kimi_k3",
        "glm_5_2",
        "minimax_m3",
        "nemotron_3_5_30b_a3b",
    )
    expected = {
        "qwen3_5_35b_a3b": (2048, 512, 256, 8),
        "qwen3_5_397b_a17b": (4096, 1024, 512, 10),
        "deepseek_v4_flash": (4096, 2048, 256, 6),
        "kimi_k3": (3584, 3072, 896, 16),
        "glm_5_2": (6144, 2048, 256, 8),
        "minimax_m3": (6144, 3072, 128, 4),
        "nemotron_3_5_30b_a3b": (2688, 1856, 128, 6),
    }
    assert {
        key: (spec.hidden, spec.intermediate, spec.experts, spec.topk)
        for key, spec in MODEL_SHAPES.items()
    } == expected


@pytest.mark.parametrize("key", MODEL_SHAPES)
def test_projection_shapes_and_ep_math(key):
    spec = MODEL_SHAPES[key]
    assert spec.gemm_shape("gate_up") == (2 * spec.padded_intermediate, spec.hidden)
    assert spec.gemm_shape("down") == (spec.hidden, spec.padded_intermediate)
    for ep in spec.ep_sizes:
        assert spec.local_experts(ep) * ep == spec.experts


def test_quick_and_full_gemm_case_grids():
    quick = generate_gemm_cases(("qwen3_5_35b_a3b",), QUICK_TOKENS, "quick", QUICK_ROUTINGS)
    assert len(quick) == 1 * 2 * 2 * 2
    assert {case.local_experts for case in quick} == {16}
    assert {case.token_expert_assignments for case in quick} == {1024, 65536}
    assert {case.projection for case in quick} == {"gate_up", "down"}

    full = generate_gemm_cases(("qwen3_5_35b_a3b",), FULL_TOKENS, "full", FULL_ROUTINGS)
    assert len(full) == 5 * 6 * 2 * 4
    assert {case.ep_size for case in full} == {1, 8, 16, 32, 64}
    assert all(case.flops == 2 * case.token_expert_assignments * case.n * case.k for case in full)


@pytest.mark.parametrize(
    "routing",
    ("balanced", "imbalanced", "single_expert_skew", "alignment_stress"),
)
@pytest.mark.parametrize("rows,experts", ((0, 8), (7, 8), (1024, 8), (65536, 128)))
def test_routing_counts_preserve_rows(rows, experts, routing):
    counts = routing_counts(rows, experts, routing)
    assert len(counts) == experts
    assert sum(counts) == rows
    assert min(counts) >= 0
    assert counts == routing_counts(rows, experts, routing)


def test_moe_matrix_distinguishes_synthetic_and_trace_sources():
    synthetic = generate_moe_cases(
        ("qwen3_5_397b_a17b",), (8192,), "quick", ("balanced",), "synthetic"
    )
    assert len(synthetic) == 1
    assert synthetic[0].ep_size == 32
    assert synthetic[0].local_experts == 16
    assert synthetic[0].token_expert_assignments == 81920

    replay = generate_moe_cases(("qwen3_5_397b_a17b",), (8192,), "quick", ("balanced",), "trace")
    assert len(replay) == 1
    assert replay[0].source == "trace"
    assert replay[0].routing == "captured"
    assert replay[0].ep_size == 32
    assert replay[0].local_experts == 16


@pytest.mark.parametrize("key", (key for key, spec in MODEL_SHAPES.items() if spec.layer_supported))
def test_quick_synthetic_moe_has_more_experts_than_topk(key):
    case = generate_moe_cases((key,), (8192,), "quick", ("balanced",))[0]
    assert case.local_experts > case.topk


def test_layer_registry_explains_model_specific_exclusion():
    with pytest.raises(ValueError, match="bounded SwiGLU backward"):
        parse_models("deepseek_v4_flash", layer_only=True)


def test_gemm_list_is_cpu_safe(monkeypatch, capsys):
    monkeypatch.setattr(nvfp4_gemm, "detect_backends", dict)
    result = nvfp4_gemm.main(
        [
            "--list",
            "--models",
            "qwen3_5_35b_a3b",
            "--tokens",
            "8192",
            "--routing",
            "balanced",
            "--projections",
            "down",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["benchmark"] == "standalone_nvfp4_grouped_gemm"
    assert payload["case_count"] == 1
    assert payload["cases"][0]["token_expert_assignments"] == 65536
    assert payload["cases"][0]["logical_flops"] == 2 * 65536 * 2048 * 512
    assert payload["cases"][0]["expert_assignment_counts"] == [4096] * 16
    assert payload["cases"][0]["routing_statistics"]["coefficient_of_variation"] == 0.0
    assert payload["definitions"]["prepacked"].startswith("resident NVFP4")
    assert payload["definitions"]["logical_flops"].startswith("2*token_expert_assignments")


def test_quick_gemm_accepts_explicit_stress_routing(monkeypatch):
    monkeypatch.setattr(nvfp4_gemm, "detect_backends", dict)
    args = nvfp4_gemm.build_parser().parse_args(
        [
            "--list",
            "--models",
            "qwen3_5_35b_a3b",
            "--tokens",
            "8192",
            "--routing",
            "single_expert_skew,alignment_stress",
            "--projections",
            "down",
        ]
    )
    payload = nvfp4_gemm.listing_payload(args)
    assert [case["routing"] for case in payload["cases"]] == [
        "single_expert_skew",
        "alignment_stress",
    ]
    assert payload["cases"][-1]["routing_statistics"]["zero_assignment_experts"] == 1


def test_dense_gemm_list_is_cpu_safe(capsys):
    result = nvfp4_gemm.main(
        [
            "--list",
            "--workload",
            "dense",
            "--models",
            "qwen3_5_35b_a3b",
            "--tokens",
            "128,8192",
            "--projections",
            "down",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["benchmark"] == "standalone_nvfp4_dense_gemm"
    assert payload["case_count"] == 2
    assert payload["cases"][0]["m"] == 128
    assert payload["cases"][1]["m"] == 8192
    assert payload["definitions"]["operation"] == "C = A @ B.T"
    assert payload["cases"][1]["logical_flops"] == 2 * 8192 * 2048 * 512


def test_imbalanced_definition_and_statistics_are_reproducible(monkeypatch):
    monkeypatch.setattr(nvfp4_gemm, "detect_backends", dict)
    args = nvfp4_gemm.build_parser().parse_args(
        [
            "--list",
            "--models",
            "deepseek_v4_flash",
            "--tokens",
            "8192",
            "--routing",
            "imbalanced",
            "--projections",
            "gate_up",
        ]
    )
    payload = nvfp4_gemm.listing_payload(args)
    case = payload["cases"][0]
    expected_counts = routing_counts(
        case["token_expert_assignments"], case["local_experts"], "imbalanced"
    )
    assert case["expert_assignment_counts"] == expected_counts
    assert case["routing_statistics"]["min_assignments"] == min(expected_counts)
    assert case["routing_statistics"]["max_assignments"] == max(expected_counts)
    assert case["routing_statistics"]["zero_assignment_experts"] == 0


def test_throughput_uses_dense_fp4_peak_only_for_gemm_scope():
    gemm = {"event_ms_p50": 1.0}
    nvfp4_gemm._annotate_throughput(
        gemm,
        9_000_000_000_000,
        9_000.0,
        gemm_only=True,
    )
    assert gemm["logical_tflops"] == 9000.0
    assert gemm["dense_fp4_spec_peak_pct"] == 100.0

    dynamic = {"event_ms_p50": 1.0}
    nvfp4_gemm._annotate_throughput(
        dynamic,
        9_000_000_000_000,
        9_000.0,
        gemm_only=False,
    )
    assert dynamic["equivalent_logical_tflops"] == 9000.0
    assert "dense_fp4_spec_peak_pct" not in dynamic


def test_tile_rounded_flops_exposes_padding_work():
    actual = nvfp4_gemm._tile_rounded_flops(
        (1, 129),
        n=130,
        k=257,
        tile_m=128,
        tile_n=128,
    )
    assert actual == 2 * (128 + 256) * 256 * 512


def test_moe_list_is_cpu_safe(monkeypatch, capsys):
    monkeypatch.setattr(moe, "detect_backends", dict)
    result = moe.main(
        [
            "--list",
            "--models",
            "glm_5_2",
            "--tokens",
            "8192",
            "--routing",
            "imbalanced",
            "--source",
            "trace",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["benchmark"] == "routed_expert_layer"
    assert payload["case_count"] == 1
    assert payload["cases"][0]["source"] == "trace"
    assert "shared experts" in payload["boundaries"]["excluded"]


def test_trace_listing_defaults_to_primary_8192_shape(monkeypatch):
    monkeypatch.setattr(moe, "detect_backends", dict)
    args = moe.build_parser().parse_args(
        ["--list", "--models", "qwen3_5_397b_a17b", "--source", "trace"]
    )
    payload = moe.listing_payload(args)
    assert payload["case_count"] == 1
    assert payload["cases"][0]["tokens"] == 8192
    assert payload["cases"][0]["routing"] == "captured"
