from __future__ import annotations

from pathlib import Path

_SRC = Path(__file__).resolve().parents[3] / "jepa" / "src" / "models"
__path__ = [str(_SRC)]

from models.vision_transformer import (  # noqa: E402,F401
    vit_tiny,
    vit_small,
    vit_base,
    vit_large,
    vit_huge,
    vit_giant,
    vit_gigantic,
)
