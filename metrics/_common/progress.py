"""Lightweight progress sidecar for Triworld eval jobs.

Each metric process writes frame counts to ``TRIWORLD_PROGRESS_FILE``.
When unset, helpers degrade to plain ``tqdm``.
"""

from __future__ import annotations

import json
import os
import time

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def count_frames(path) -> int:
    """Frames an item contributes: image files in folder, else 1."""
    try:
        if os.path.isdir(path):
            n = sum(1 for f in os.listdir(path) if f.lower().endswith(_IMAGE_EXTS))
            return n if n > 0 else 1
    except OSError:
        pass
    return 1


def _progress_file() -> str | None:
    return os.environ.get("TRIWORLD_PROGRESS_FILE") or os.environ.get("WA_PROGRESS_FILE")


def _rank() -> int:
    for key in ("RANK", "LOCAL_RANK"):
        val = os.environ.get(key)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                return 0
    return 0


def _enabled() -> bool:
    return bool(_progress_file()) and _rank() == 0


def write_progress(
    processed_frames: int,
    total_frames: int,
    processed_items: int,
    total_items: int,
    *,
    phase: str | None = None,
    active_items: int | None = None,
) -> None:
    path = _progress_file()
    if not path or _rank() != 0:
        return
    payload = {
        "processed_frames": int(processed_frames),
        "total_frames": int(total_frames),
        "processed_items": int(processed_items),
        "total_items": int(total_items),
        "updated_at": time.time(),
    }
    if phase:
        payload["phase"] = phase
    if active_items is not None:
        payload["active_items"] = int(active_items)
    try:
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except OSError:
        pass


def progress_iter(items, disable=False, count_fn=count_frames, desc=None):
    items = list(items)
    emit = _enabled()
    total_frames = sum(count_fn(it) for it in items) if emit else 0
    processed_frames = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(items, disable=disable or emit, desc=desc)
    except Exception:
        iterator = items

    if emit:
        write_progress(0, total_frames, 0, len(items))

    for idx, item in enumerate(iterator):
        yield item
        if emit:
            processed_frames += count_fn(item)
            write_progress(processed_frames, total_frames, idx + 1, len(items))
