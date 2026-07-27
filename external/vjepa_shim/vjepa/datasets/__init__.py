from __future__ import annotations

from pathlib import Path

_SRC = Path(__file__).resolve().parents[3] / "jepa" / "src" / "datasets"
__path__ = [str(_SRC)]
