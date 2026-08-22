"""Model geometries and case generation for the two public benchmarks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

QUICK_TOKENS = (128, 8192)
FULL_TOKENS = (1, 128, 512, 2048, 8192, 16384)
QUICK_ROUTINGS = ("balanced", "imbalanced")
FULL_ROUTINGS = ("balanced", "imbalanced", "single_expert_skew", "alignment_stress")


@dataclass(frozen=True)
class ModelShape:
    key: str
    name: str
    model_id: str
    revision: str
    hidden: int
    intermediate: int
    experts: int
    topk: int
    ep_sizes: tuple[int, ...]
    quick_ep: int
    activation: str = "swiglu"
    layer_supported: bool = True
    layer_exclusion: str = ""

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
        if projection == "gate_up":
            return 2 * self.padded_intermediate, self.hidden
        if projection == "down":
            return self.hidden, self.padded_intermediate
        raise ValueError(f"unknown projection {projection!r}")


MODEL_SHAPES = {
    spec.key: spec
    for spec in (
        ModelShape(
            "qwen3_5_35b_a3b",
            "Qwen3.5-35B-A3B",
            "Qwen/Qwen3.5-35B-A3B-FP8",
            "9d1823d2dee688a6b25e77009dc727688c44936e",
            2048,
            512,
            256,
            8,
            (1, 8, 16, 32, 64),
            16,
        ),
        ModelShape(
            "qwen3_5_397b_a17b",
            "Qwen3.5-397B-A17B",
            "Qwen/Qwen3.5-397B-A17B-FP8",
            "ea5b4f81096f3901c91dea97f81324302495781d",
            4096,
            1024,
            512,
            10,
            (1, 8, 16, 32, 64),
            32,
        ),
        ModelShape(
            "deepseek_v4_flash",
            "DeepSeek-V4-Flash",
            "nvidia/DeepSeek-V4-Flash-NVFP4",
            "e3cd60e7de98e9867116860d522499a728de1cf9",
            4096,
            2048,
            256,
            6,
            (4, 8, 32, 64),
            32,
            "swiglu",
            False,
            "bounded SwiGLU backward is not implemented",
        ),
        ModelShape(
            "kimi_k3",
            "Kimi-K3",
            "moonshotai/Kimi-K3",
            "a590ce090cb049c93a33dfe8c208ec652aa20503",
            3584,
            3072,
            896,
            16,
            (1, 8, 16, 32, 64, 128),
            32,
            "situ_glu",
            False,
            "latent projections and SiTU-GLU are outside the grouped expert core",
        ),
        ModelShape(
            "glm_5_2",
            "GLM-5.2",
            "zai-org/GLM-5.2",
            "b4734de4facf877f85769a911abafc5283eab3d9",
            6144,
            2048,
            256,
            8,
            (1, 8, 16, 32, 64),
            16,
        ),
        ModelShape(
            "minimax_m3",
            "MiniMax-M3",
            "MiniMaxAI/MiniMax-M3",
            "f0e1c1e04d40177e4673a22097036854f536e9c0",
            6144,
            3072,
            128,
            4,
            (1, 8, 16, 32),
            16,
            "swiglu_oai",
            False,
            "SwiGLU-OAI is not implemented",
        ),
        ModelShape(
            "nemotron_3_5_30b_a3b",
            "Nemotron-3.5-Lightning-30B-A3B",
            "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
            "e8f3c7c4de75ad84fe1bcef95d38eca76214480b",
            2688,
            1856,
            128,
            6,
            (1, 8, 16, 32),
            16,
            "relu2",
            False,
            "ReLU-squared is not implemented",
        ),
    )
}


@dataclass(frozen=True)
class GemmCase:
    model: str
    projection: str
    tokens: int
    token_expert_assignments: int
    global_experts: int
    local_experts: int
    ep_size: int
    topk: int
    n: int
    k: int
    routing: str

    @property
    def flops(self) -> int:
        return 2 * self.token_expert_assignments * self.n * self.k

    @property
    def label(self) -> str:
        return (
            f"{self.model}:projection={self.projection}:input_tokens={self.tokens}:"
            f"ep_size={self.ep_size}:assignments={self.token_expert_assignments}:"
            f"local_experts={self.local_experts}:routing={self.routing}"
        )


@dataclass(frozen=True)
class MoeCase:
    model: str
    tokens: int
    token_expert_assignments: int
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
            f"{self.model}:input_tokens={self.tokens}:ep_size={self.ep_size}:"
            f"assignments={self.token_expert_assignments}:local_experts={self.local_experts}:"
            f"source={self.source}:routing={self.routing}"
        )


def parse_models(value: str | Iterable[str], layer_only: bool = False) -> tuple[str, ...]:
    allowed = {key for key, spec in MODEL_SHAPES.items() if not layer_only or spec.layer_supported}
    if isinstance(value, str):
        requested = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        requested = tuple(value)
    if not requested or requested == ("all",):
        return tuple(key for key in MODEL_SHAPES if key in allowed)
    excluded = [
        key
        for key in requested
        if key in MODEL_SHAPES and layer_only and not MODEL_SHAPES[key].layer_supported
    ]
    if excluded:
        details = "; ".join(f"{key}: {MODEL_SHAPES[key].layer_exclusion}" for key in excluded)
        raise ValueError(f"unsupported routed-expert layer model: {details}")
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise ValueError(f"unsupported models {unknown}; choose from {sorted(allowed)}")
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


def _quick_moe_ep(spec: ModelShape) -> int:
    candidates = [ep for ep in spec.ep_sizes if spec.local_experts(ep) > spec.topk]
    return max(candidates, default=spec.quick_ep)


def generate_gemm_cases(
    models: Iterable[str],
    tokens: Iterable[int],
    suite: str,
    routings: Iterable[str] | None = None,
    projections: Iterable[str] = ("gate_up", "down"),
) -> list[GemmCase]:
    routing_grid = tuple(routings or (QUICK_ROUTINGS if suite == "quick" else FULL_ROUTINGS))
    rows = []
    for key in models:
        spec = MODEL_SHAPES[key]
        for ep_size in _ep_grid(spec, suite):
            local_experts = spec.local_experts(ep_size)
            for token_count in tokens:
                token_expert_assignments = token_count * spec.topk
                for projection in projections:
                    n, k = spec.gemm_shape(projection)
                    for routing in routing_grid:
                        rows.append(
                            GemmCase(
                                key,
                                projection,
                                token_count,
                                token_expert_assignments,
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
        if not spec.layer_supported:
            raise ValueError(f"{key}: {spec.layer_exclusion}")
        # A trace represents one local EP shard.
        if source == "trace":
            ep_grid = (spec.quick_ep,)
        elif suite == "quick":
            ep_grid = (_quick_moe_ep(spec),)
        else:
            ep_grid = spec.ep_sizes
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


def routing_counts(assignments: int, experts: int, routing: str) -> tuple[int, ...]:
    """Build deterministic expert loads for a fixed assignment count."""
    if assignments < 0 or experts <= 0:
        raise ValueError("assignments must be non-negative and experts must be positive")
    if routing == "balanced":
        weights = [1] * experts
    elif routing == "imbalanced":
        weights = [1 + ((expert * 17 + 11) % 31) for expert in range(experts)]
    elif routing == "single_expert_skew":
        weights = [max(1, experts // 2)] + [1] * (experts - 1)
    elif routing == "alignment_stress":
        edge = (1, 15, 16, 127, 128, 129, 255, 256, 257)
        weights = [edge[expert % len(edge)] for expert in range(experts)]
        if experts > 1:
            weights[-1] = 0
    else:
        raise ValueError(f"unknown routing {routing!r}")
    total_weight = sum(weights)
    counts = [assignments * weight // total_weight for weight in weights]
    remainder = assignments - sum(counts)
    order = sorted(
        range(experts),
        key=lambda expert: (assignments * weights[expert]) % total_weight,
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
