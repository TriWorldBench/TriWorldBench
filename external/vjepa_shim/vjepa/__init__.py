"""Namespace shim for the Meta JEPA source layout used by videojedi."""

from __future__ import annotations

from pathlib import Path

JEPA_ROOT = Path(__file__).resolve().parents[2] / "jepa" / "src"
