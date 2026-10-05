"""Strategy registry. Every strategy version stays in this package (v1, v2, ...) so results stay reproducible."""
from __future__ import annotations

import importlib
from typing import Any, Dict

from .base import Decision, Snapshot, Strategy

_CLASSES = {
    "v1_momentum_rotation": ("strategies.v1_momentum_rotation", "MomentumRotation"),
}


def available() -> list:
    return sorted(_CLASSES)


def make(name: str, params: Dict[str, Any] | None = None) -> Strategy:
    if name not in _CLASSES:
        raise KeyError(f"unknown strategy {name!r}; available: {available()}")
    module, cls = _CLASSES[name]
    return getattr(importlib.import_module(module), cls)(**(params or {}))


__all__ = ["Decision", "Snapshot", "Strategy", "make", "available"]
