"""Episode/view constants and numeric sorting for identifiers."""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

VIEW_ORDER = ("head", "left", "right")
EP_RE = re.compile(r"^episode(\d+)$", re.IGNORECASE)


def natural_key(text: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(text))]


def episode_sort_key(name: str) -> tuple[Any, ...]:
    match = EP_RE.match(str(name))
    if match:
        return (0, int(match.group(1)))
    return (1, *natural_key(name))


def sort_episode_names(names: Iterable[str]) -> list[str]:
    return sorted(names, key=episode_sort_key)


def sort_episode_items(items: Sequence[Any], *, key: str = "episode") -> list[Any]:
    def lookup(item: Any) -> str:
        if isinstance(item, dict):
            return str(item.get(key) or "")
        return str(getattr(item, key))

    return sorted(items, key=lambda item: episode_sort_key(lookup(item)))


def sample_sort_key(sample: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(sample.get("task") or ""),
        episode_sort_key(str(sample.get("episode") or "")),
        str(sample.get("phase") or ""),
        str(sample.get("arm") or ""),
        int(sample.get("sampled_frame") or 0),
    )
