"""Model geometries and case generation for the two public benchmarks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

QUICK_TOKENS = (128, 8192)
FULL_TOKENS = (1, 128, 512, 2048, 8192, 16384)
QUICK_ROUTINGS = ("balanced", "jagged")
FULL_ROUTINGS = ("balanced", "jagged", "hotspot", "tail")


@dataclass(frozen=True)
class ModelShape:
    key: str
    name: str
    model_id: str
    hidden: int
    intermediate: int
    experts: int
    topk: int
    ep_sizes: tuple[int, ...]
    quick_ep: int
    activation: str = "swiglu"

    def __post_init__(self):
        if self.experts % self.quick_ep:
            raise ValueError(f"{self.key}: quick_ep must divide the expert count")
        if any(self.experts % ep for ep in self.ep_sizes):
            raise ValueError(f"{self.key}: every EP size must divide the expert count")

    @property
    def padded_intermediate(self) -> int:
        return ((self.intermediate + 127) // 128) * 128

    def local_experts(self, ep_size: int) -> int:
        if ep_size != 1 and ep_size not in self.ep_sizes:
            raise ValueError(f"{self.key}: unsupported EP={ep_size}; choose from {self.ep_sizes}")
        return self.experts // ep_size

    def gemm_shape(self, projection: str) -> tuple[int, int]:
        """Return (N, K) for B[E, N, K] in the grouped GEMM convention."""
        if projection == "fc1":
            return 2 * self.padded_intermediate, self.hidden
        if projection == "fc2":
            return self.hidden, self.padded_intermediate
        raise ValueError(f"unknown projection {projection!r}")


MODEL_SHAPES = {
    spec.key: spec
    for spec in (
        ModelShape(
            "qwen3_30b_a3b",
            "Qwen3-30B-A3B",
            "Qwen/Qwen3-30B-A3B",
            2048,
            768,
            128,
            8,
            (1, 8, 16, 32),
            16,
        ),
        ModelShape(
            "qwen3_235b_a22b",
            "Qwen3-235B-A22B",
            "Qwen/Qwen3-235B-A22B",
            4096,
            1536,
            128,
            8,
            (1, 8, 16, 32),
            16,
        ),
        ModelShape(
            "gemma4_26b_a4b",
            "Gemma 4 26B A4B",
            "google/gemma-4-26B-A4B",
            2816,
            704,
            128,
            8,
            (1, 8, 16, 32),
            16,
            "geglu",
        ),
        ModelShape(
            "deepseek_v3_2",
            "DeepSeek-V3.2",
            "deepseek-ai/DeepSeek-V3.2",
            7168,
            2048,
            256,
            8,
            (4, 8, 32, 64),
            32,
        ),
        ModelShape(
            "kimi_k2_7",
            "Kimi-K2.7",
            "moonshotai/Kimi-K2.7-Code",
            7168,
            2048,
            384,
            8,
            (3, 24, 48, 96),
            48,
        ),
        ModelShape(
            "minimax_m2",
            "MiniMax-M2",
            "MiniMaxAI/MiniMax-M2",
            3072,
            1536,
            256,
            8,
            (4, 8, 32, 64),
            32,
        ),
        ModelShape(
            "llama4_scout",
            "Llama 4 Scout",
            "meta-llama/Llama-4-Scout-17B-16E",
            5120,
            8192,
            16,
            1,
            (1, 2, 4, 8),
            2,
        ),
    )
}


@dataclass(frozen=True)
class GemmCase:
    model: str
    projection: str
    tokens: int
    routed_rows: int
    global_experts: int
    local_experts: int
    ep_size: int
    topk: int
    n: int
    k: int
    routing: str

    @property
    def flops(self) -> int:
        return 2 * self.routed_rows * self.n * self.k

    @property
    def label(self) -> str:
        return (
            f"{self.model}:{self.projection}:t{self.tokens}:ep{self.ep_size}:"
            f"m{self.routed_rows}:e{self.local_experts}:{self.routing}"
        )


@dataclass(frozen=True)
class MoeCase:
    model: str
    tokens: int
    routed_rows: int
    global_experts: int
    local_experts: int
    ep_size: int
    topk: int
    hidden: int
    intermediate: int
    activation: str
    routing: str
    source: str

    @property
    def label(self) -> str:
        return (
            f"{self.model}:t{self.tokens}:ep{self.ep_size}:m{self.routed_rows}:"
            f"e{self.local_experts}:{self.source}:{self.routing}"
        )


def parse_models(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        requested = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        requested = tuple(value)
    if not requested or requested == ("all",):
        return tuple(MODEL_SHAPES)
    unknown = sorted(set(requested) - MODEL_SHAPES.keys())
    if unknown:
        raise ValueError(f"unknown models {unknown}; choose from {sorted(MODEL_SHAPES)}")
    return requested


def parse_ints(value: str | Iterable[int] | None, suite: str) -> tuple[int, ...]:
    if value is None or value == "":
        return QUICK_TOKENS if suite == "quick" else FULL_TOKENS
    if isinstance(value, str):
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        parsed = tuple(int(item) for item in value)
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError("tokens must be a comma-separated list of positive integers")
    return parsed


def parse_names(value: str | Iterable[str], allowed: Iterable[str], kind: str) -> tuple[str, ...]:
    if isinstance(value, str):
        parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        parsed = tuple(value)
    allowed_set = set(allowed)
    if parsed == ("all",):
        return tuple(allowed)
    unknown = sorted(set(parsed) - allowed_set)
    if not parsed or unknown:
        raise ValueError(f"invalid {kind} {unknown or parsed}; choose from {sorted(allowed_set)}")
    return parsed


def _ep_grid(spec: ModelShape, suite: str) -> tuple[int, ...]:
    return (spec.quick_ep,) if suite == "quick" else spec.ep_sizes


def generate_gemm_cases(
    models: Iterable[str],
    tokens: Iterable[int],
    suite: str,
    routings: Iterable[str] | None = None,
    projections: Iterable[str] = ("fc1", "fc2"),
) -> list[GemmCase]:
    routing_grid = tuple(routings or (QUICK_ROUTINGS if suite == "quick" else FULL_ROUTINGS))
    rows = []
    for key in models:
        spec = MODEL_SHAPES[key]
        for ep_size in _ep_grid(spec, suite):
            local_experts = spec.local_experts(ep_size)
            for token_count in tokens:
                routed_rows = token_count * spec.topk
                for projection in projections:
                    n, k = spec.gemm_shape(projection)
                    for routing in routing_grid:
                        rows.append(
                            GemmCase(
                                key,
                                projection,
                                token_count,
                                routed_rows,
                                spec.experts,
                                local_experts,
                                ep_size,
                                spec.topk,
                                n,
                                k,
                                routing,
                            )
                        )
    return rows


def generate_moe_cases(
    models: Iterable[str],
    tokens: Iterable[int],
    suite: str,
    routings: Iterable[str] | None = None,
    source: str = "synthetic",
) -> list[MoeCase]:
    routing_grid = (
        ("captured",)
        if source == "trace"
        else tuple(routings or (QUICK_ROUTINGS if suite == "quick" else FULL_ROUTINGS))
    )
    rows = []
    for key in models:
        spec = MODEL_SHAPES[key]
        # A replay file represents one local EP shard.  Using the quick EP
        # keeps every registered model within the native 256-expert limit and
        # avoids pretending that one global checkpoint tensor is a local run.
        ep_grid = (spec.quick_ep,) if source == "trace" else _ep_grid(spec, suite)
        for ep_size in ep_grid:
            for token_count in tokens:
                for routing in routing_grid:
                    rows.append(
                        MoeCase(
                            key,
                            token_count,
                            token_count * spec.topk,
                            spec.experts,
                            spec.local_experts(ep_size),
                            ep_size,
                            spec.topk,
                            spec.hidden,
                            spec.padded_intermediate,
                            spec.activation,
                            routing,
                            source,
                        )
                    )
    return rows


def routing_counts(rows: int, experts: int, routing: str) -> tuple[int, ...]:
    """Build deterministic expert counts while preserving exactly ``rows``."""
    if rows < 0 or experts <= 0:
        raise ValueError("rows must be non-negative and experts must be positive")
    if routing == "balanced":
        weights = [1] * experts
    elif routing == "jagged":
        weights = [1 + ((expert * 17 + 11) % 31) for expert in range(experts)]
    elif routing == "hotspot":
        weights = [max(1, experts // 2)] + [1] * (experts - 1)
    elif routing == "tail":
        edge = (1, 15, 16, 127, 128, 129, 255, 256, 257)
        weights = [edge[expert % len(edge)] for expert in range(experts)]
        if experts > 1:
            weights[-1] = 0
    else:
        raise ValueError(f"unknown routing {routing!r}")
    total_weight = sum(weights)
    counts = [rows * weight // total_weight for weight in weights]
    remainder = rows - sum(counts)
    order = sorted(
        range(experts),
        key=lambda expert: (rows * weights[expert]) % total_weight,
        reverse=True,
    )
    for expert in order[:remainder]:
        counts[expert] += 1
    return tuple(counts)


__all__ = [
    "FULL_ROUTINGS",
    "FULL_TOKENS",
    "MODEL_SHAPES",
    "QUICK_ROUTINGS",
    "QUICK_TOKENS",
    "GemmCase",
    "ModelShape",
    "MoeCase",
    "generate_gemm_cases",
    "generate_moe_cases",
    "parse_ints",
    "parse_models",
    "parse_names",
    "routing_counts",
]
