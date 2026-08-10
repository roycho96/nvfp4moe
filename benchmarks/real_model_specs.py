"""Model geometries used by the real-stack benchmark.

Capture validates these values against each checkpoint before loading weights.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    hidden: int
    intermediate: int
    experts: int
    topk: int
    activation: str
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rope_theta: float
    attention_kind: str = "full"
    sliding_window: int | None = None

    @property
    def padded_intermediate(self) -> int:
        """Common I accepted by NVFP4 and DeepGEMM SM100 kernels."""
        return ((self.intermediate + 127) // 128) * 128


SPECS = {
    "qwen3_30b_a3b": ModelSpec(
        key="qwen3_30b_a3b",
        model_id="Qwen/Qwen3-30B-A3B",
        hidden=2048,
        intermediate=768,
        experts=128,
        topk=8,
        activation="swiglu",
        num_heads=32,
        num_kv_heads=4,
        head_dim=128,
        rope_theta=1_000_000.0,
    ),
    # Gemma 4 alternates five local layers and one global layer.  Expert-only
    # measurements use the common MoE geometry below; stack attention is
    # reported separately for the local and global shapes.
    "gemma4_26b_a4b_local": ModelSpec(
        key="gemma4_26b_a4b_local",
        model_id="google/gemma-4-26B-A4B",
        hidden=2816,
        intermediate=704,
        experts=128,
        topk=8,
        activation="geglu",
        num_heads=16,
        num_kv_heads=8,
        head_dim=256,
        rope_theta=10_000.0,
        attention_kind="sliding",
        sliding_window=1024,
    ),
    "gemma4_26b_a4b_global": ModelSpec(
        key="gemma4_26b_a4b_global",
        model_id="google/gemma-4-26B-A4B",
        hidden=2816,
        intermediate=704,
        experts=128,
        topk=8,
        activation="geglu",
        num_heads=16,
        num_kv_heads=2,
        head_dim=512,
        rope_theta=1_000_000.0,
        attention_kind="full",
    ),
}


def get_spec(key: str) -> ModelSpec:
    try:
        return SPECS[key]
    except KeyError as exc:
        raise ValueError(f"unknown model spec {key!r}; choose from {sorted(SPECS)}") from exc


def canonical_trace_key(key: str) -> str:
    """Local/global Gemma attention cases share one expert/router trace."""
    return "gemma4_26b_a4b" if key.startswith("gemma4_26b_a4b") else key


__all__ = ["SPECS", "ModelSpec", "canonical_trace_key", "get_spec"]
