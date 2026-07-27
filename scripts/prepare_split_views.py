#!/usr/bin/env python3
"""Build Triworld split_views from flat episode trees (GT-compatible layout).

Flat layout:
  <gen-root>/episodeN/{head,left,right}/frames/frame_*.jpg
  <gt-root>/episodeN/...

Output:
  <out-root>/generated_dataset/episodeN/{head,left,right}/video/frame_*.jpg
  <out-root>/gt_dataset/episodeN/{head,left,right}/video/frame_*.jpg
  <out-root>/gt_dataset/episodeN/{view}/prompt/prompt.txt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR / "lib"))
sys.path.insert(0, str(PROJECT_ROOT))

from eval_data_common import PROMPT_KEYS, VIEW_NAMES, count_frames, read_json, resolve_frame_dir  # noqa: E402
from metrics._common.episode_sort import sort_episode_names  # noqa: E402
from metrics._common.frames import FRAME_EXT  # noqa: E402
EP_RE = re.compile(r"^episode(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class EpisodeRecord:
    episode: str
    gen_dir: Path
    gt_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--gt-frame-mode",
        choices=("matched", "full"),
        default="matched",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def relative_symlink(src: Path, dst: Path) -> None:
    src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    target = os.path.relpath(src, start=dst.parent.resolve())
    os.symlink(target, dst, target_is_directory=False)


def read_instruction(episode_dir: Path, episode: str) -> str:
    for name in (f"{episode}.json", "episode.json"):
        path = episode_dir / name
        if not path.is_file():
            continue
        try:
            data = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            for key in PROMPT_KEYS:
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return ""


def discover_episodes(gen_root: Path, gt_root: Path, limit: int) -> list[EpisodeRecord]:
    """Input-driven discovery: iterate generated episodes, sample matching GT."""
    episodes = sort_episode_names(p.name for p in gen_root.iterdir() if p.is_dir() and EP_RE.match(p.name))
    records: list[EpisodeRecord] = []
    missing_gt: list[str] = []
    for episode in episodes:
        gen_dir = gen_root / episode
        gt_dir = gt_root / episode
        if not gt_dir.is_dir():
            missing_gt.append(episode)
            continue
        if not all(resolve_frame_dir(d, v) for d, v in ((gen_dir, view) for view in VIEW_NAMES)):
            print(f"[skip] {episode} missing generated view frames")
            continue
        if not all(resolve_frame_dir(d, v) for d, v in ((gt_dir, view) for view in VIEW_NAMES)):
            print(f"[skip] {episode} missing GT view frames")
            continue
        records.append(EpisodeRecord(episode=episode, gen_dir=gen_dir, gt_dir=gt_dir))
    if missing_gt:
        print(f"[warn] {len(missing_gt)} input episode(s) missing GT: {missing_gt[:5]}{'...' if len(missing_gt) > 5 else ''}")
    if limit > 0:
        records = records[:limit]
    return records


def symlink_frames(src_dir: Path, dst_dir: Path, max_frames: int | None) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(p for p in src_dir.iterdir() if p.suffix.lower() == FRAME_EXT)
    if max_frames is not None:
        frames = frames[:max_frames]
    for frame in frames:
        relative_symlink(frame, dst_dir / frame.name)
    return len(frames)


def build_split_views(records: list[EpisodeRecord], out_root: Path, gt_frame_mode: str) -> dict:
    gen_out = out_root / "generated_dataset"
    gt_out = out_root / "gt_dataset"
    summary: list[dict] = []

    for rec in records:
        gen_counts = {v: count_frames(resolve_frame_dir(rec.gen_dir, v)) for v in VIEW_NAMES}
        gt_counts = {v: count_frames(resolve_frame_dir(rec.gt_dir, v)) for v in VIEW_NAMES}
        matched = min(min(gen_counts.values()), min(gt_counts.values()))

        for view in VIEW_NAMES:
            gen_src = resolve_frame_dir(rec.gen_dir, view)
            gt_src = resolve_frame_dir(rec.gt_dir, view)
            gen_dst = gen_out / rec.episode / view / "video"
            gt_dst = gt_out / rec.episode / view / "video"
            symlink_frames(gen_src, gen_dst, matched)
            gt_max = matched if gt_frame_mode == "matched" else None
            symlink_frames(gt_src, gt_dst, gt_max)

            prompt_dir = gt_out / rec.episode / view / "prompt"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = prompt_dir / "prompt.txt"
            if not prompt_path.is_file():
                instruction = read_instruction(rec.gt_dir, rec.episode)
                prompt_path.write_text(instruction, encoding="utf-8")

        summary.append({
            "episode": rec.episode,
            "matched_frames": matched,
            "gen_frames": gen_counts,
            "gt_frames": gt_counts,
        })

    return {"episodes": len(summary), "records": summary}


def main() -> int:
    args = parse_args()
    gen_root = args.gen_root.resolve()
    gt_root = args.gt_root.resolve()
    out_root = args.out_root.resolve()

    records = discover_episodes(gen_root, gt_root, args.limit)
    if not records:
        print("[error] no episodes found", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"[dry-run] {len(records)} episodes -> {out_root}")
        return 0

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = build_split_views(records, out_root, args.gt_frame_mode)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "gen_root": str(gen_root),
        "gt_root": str(gt_root),
        "out_root": str(out_root),
        **summary,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] {summary['episodes']} episodes -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
