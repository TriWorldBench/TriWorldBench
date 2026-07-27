from __future__ import annotations

from pathlib import Path

_SRC = Path(__file__).resolve().parents[3] / "jepa" / "src" / "masks"
__path__ = [str(_SRC)]
