"""Load the vendored QuACK sources and register them with the JIT cache."""

import os
import sys
from pathlib import Path

_SOURCE_TREE = Path(__file__).resolve().parent.parent / "third_party" / "quack"


def _contains_quack(path: Path) -> bool:
    try:
        return (path / "quack").is_dir()
    except OSError:
        return False


def quack_path() -> Path | None:
    override = os.environ.get("NVFP4MOE_QUACK_PATH")
    if override:
        path = Path(override)
        if not _contains_quack(path):
            raise ImportError(f"NVFP4MOE_QUACK_PATH has no quack package: {path}")
        return path
    for path in (_SOURCE_TREE, Path("/root/fork")):
        if _contains_quack(path):
            return path
    return None


def ensure_quack():
    path = quack_path()
    if path is not None and str(path) not in sys.path:
        sys.path.insert(0, str(path))
    import quack.cache as qcache

    src = Path(qcache.__file__).resolve().parent
    if src not in qcache.EXTRA_SOURCE_DIRS:
        qcache.EXTRA_SOURCE_DIRS.append(src)
    # Local grouped-wgrad kernels also participate in cache invalidation.
    pkg = Path(__file__).resolve().parent
    if pkg not in qcache.EXTRA_SOURCE_DIRS:
        qcache.EXTRA_SOURCE_DIRS.append(pkg)


ensure_quack()
