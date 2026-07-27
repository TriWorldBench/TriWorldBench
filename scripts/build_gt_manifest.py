#!/usr/bin/env python3
"""Build GT manifest CSV from test_data/gt_dataset (episode, frames, resolution).

If manifest already exists and --force is not set, skip regeneration.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR / "lib"))
sys.path.insert(0, str(PROJECT_ROOT))

from eval_data_common import (  # noqa: E402
    VIEW_NAMES,
    count_frames,
    gt_episode_ready,
    probe_image_size,
    resolve_frame_dir,
)
from metrics._common.frames import FRAME_EXT  # noqa: E402
from metrics._common.episode_sort import sort_episode_names  # noqa: E402

EP_RE = re.compile(r"^episode(\d+)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=PROJECT_ROOT / "test_data" / "gt_dataset",
        help="Flat GT episode root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "test_data" / "gt_manifest.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate even if output exists.")
    return parser.parse_args()


def discover_episodes(gt_root: Path) -> list[str]:
    return sort_episode_names(
        p.name for p in gt_root.iterdir() if p.is_dir() and EP_RE.match(p.name)
    )


def episode_row(gt_root: Path, episode: str, ffmpeg: str = "ffmpeg") -> dict[str, str | int]:
    ep_dir = gt_root / episode
    frame_counts = []
    width = height = 0
    for view in VIEW_NAMES:
        frame_dir = resolve_frame_dir(ep_dir, view)
        if frame_dir is None:
            raise FileNotFoundError(f"missing view {view} for {episode}")
        frame_counts.append(count_frames(frame_dir))
        if width == 0:
            sample = sorted(frame_dir.glob(f"frame_*{FRAME_EXT}"))[0]
            width, height = probe_image_size(sample, ffmpeg)
    return {
        "episode": episode,
        "frame_count": min(frame_counts),
        "width": width,
        "height": height,
    }


def main() -> int:
    args = parse_args()
    gt_root = args.gt_root.resolve()
    output = args.output.resolve()

    if output.is_file() and not args.force:
        print(f"[skip] manifest exists: {output}")
        return 0

    if not gt_root.is_dir():
        print(f"[error] GT root not found: {gt_root}", file=sys.stderr)
        return 1

    episodes = discover_episodes(gt_root)
    rows = []
    skipped = []
    for ep in episodes:
        if not gt_episode_ready(gt_root, ep):
            skipped.append(ep)
            continue
        rows.append(episode_row(gt_root, ep))
    if skipped:
        print(f"[warn] skipped {len(skipped)} incomplete episodes (missing frames)", file=sys.stderr)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "frame_count", "width", "height"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[ok] {len(rows)} episodes -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
