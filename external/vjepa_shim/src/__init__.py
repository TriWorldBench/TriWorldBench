from __future__ import annotations

from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "jepa" / "src"
__path__ = [str(_SRC)]
