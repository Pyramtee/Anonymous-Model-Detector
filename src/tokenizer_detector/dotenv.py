"""Minimal .env loader (no extra dependency)."""

from __future__ import annotations

import os
from pathlib import Path

from .registry import PROJECT_ROOT


def load_dotenv(path: Path | None = None, override: bool = False) -> Path | None:
    """Load KEY=VALUE pairs from .env into os.environ. Returns path if loaded."""
    path = path or (PROJECT_ROOT / ".env")
    if not path.exists():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value
    return path
