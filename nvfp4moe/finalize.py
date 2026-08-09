"""Fixed-order MoE combine and its per-slot backward gather."""

from . import _vendor  # noqa: F401
from quack.moe_finalize import moe_finalize, moe_finalize_bwd  # noqa: E402

__all__ = ["moe_finalize", "moe_finalize_bwd"]
