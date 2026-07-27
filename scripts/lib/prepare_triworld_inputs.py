#!/usr/bin/env python3
"""Prepare paired robot-video data for the Triworld evaluator.

The adapter consumes a SINGLE split-view dataset (per-view PNG frame folders) and
produces the run-local input tree:
  - inputs/data              : per-view frame folders (symlinks) for the frame metrics + trajectory
  - inputs/flat/{generated,gt}: 3-view panel MP4s reconstructed from the frames, for JEPA
  - inputs/vlm_summary.json   : per-episode prompt + generated frame-folder refs, for the
                                VLM judge (which reads frames directly from split_views)

No separate all_infer dataset is required: the panel MP4s and prompts are derived from
the split-view frames (prompt lives at gt_dataset/<episode>/<view>/prompt/prompt.txt).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from eval_data_common import count_frames, write_lossless_mp4_from_frame_dir
from metrics._common.episode_sort import episode_sort_key, sort_episode_names
from metrics._common.frames import FRAME_EXT, frame_filename
VIEW_ORDER = ("head", "left", "right")
EP_RE = re.compile(r"^episode(\d+)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build run-local Triworld input trees from split-view data.")
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--max-split-records",
        type=int,
        default=0,
        help="Limit to the first N episodes for smoke tests.",
    )
    parser.add_argument("--flat-fps", type=int, default=10, help="FPS for reconstructed panel MP4s.")
    parser.add_argument("--flat-workers", type=int, default=min(16, os.cpu_count() or 4))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def ensure_clean(path: Path, clean: bool) -> None:
    if path.exists() and clean:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def replace_symlink(src: Path, dst: Path) -> None:
    src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    target = os.path.relpath(src, start=dst.parent.resolve())
    os.symlink(target, dst, target_is_directory=src.is_dir())


def read_prompt_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def discover_split_views(split_root: Path) -> list[dict[str, Any]]:
    gen_root = split_root / "generated_dataset"
    gt_root = split_root / "gt_dataset"
    if not gen_root.is_dir():
        raise FileNotFoundError(f"generated split dataset not found: {gen_root}")
    if not gt_root.is_dir():
        raise FileNotFoundError(f"GT split dataset not found: {gt_root}")

    records: list[dict[str, Any]] = []
    for episode_dir in sorted(
        (p for p in gen_root.iterdir() if p.is_dir() and EP_RE.match(p.name)),
        key=lambda p: episode_sort_key(p.name),
    ):
        episode = episode_dir.name
        for view in VIEW_ORDER:
            gen_video = episode_dir / view / "video"
            gt_video = gt_root / episode / view / "video"
            if not gen_video.is_dir() or not gt_video.is_dir():
                continue
            records.append(
                {
                    "episode": episode,
                    "view": view,
                    "gen_video": gen_video,
                    "gt_video": gt_video,
                    "gen_frames": count_frames(gen_video),
                    "gt_frames": count_frames(gt_video),
                }
            )
    if not records:
        raise RuntimeError(f"no split-view records found under {split_root}")
    return records


def build_standard_tree(records: list[dict[str, Any]], out_root: Path) -> dict[str, Path]:
    gen_root = out_root / "data" / "generated_dataset"
    gt_root = out_root / "data" / "gt_dataset"
    for rec in records:
        gen_dst = gen_root / rec["view"] / rec["episode"] / "1" / "video"
        gt_dst = gt_root / rec["view"] / rec["episode"] / "video"
        replace_symlink(Path(rec["gen_video"]), gen_dst)
        replace_symlink(Path(rec["gt_video"]), gt_dst)
    return {"generated": gen_root, "gt": gt_root}


# ---------------------------------------------------------------------------
# Panel MP4 reconstruction (module-level helpers so they are picklable for Pool)
# ---------------------------------------------------------------------------

def _frames_in(folder: str) -> list[str]:
    try:
        return sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() == FRAME_EXT
        )
    except OSError:
        return []


def _write_panel_mp4(view_dirs: list[str], dst: str, fps: int, ffmpeg: str) -> tuple[bool, str | None]:
    """hconcat head|left|right per frame into a lossless H.264 MP4."""
    frame_lists = [_frames_in(d) for d in view_dirs]
    if not all(frame_lists):
        return False, f"missing frames for {dst}"
    n = min(len(fl) for fl in frame_lists)
    target_size = None
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i in range(n):
            crops = []
            for fl in frame_lists:
                img = cv2.imread(fl[i], cv2.IMREAD_COLOR)
                if img is None:
                    return False, f"unreadable frame: {fl[i]}"
                if target_size is None:
                    h, w = img.shape[:2]
                    target_size = (w, h)
                elif img.shape[1] != target_size[0] or img.shape[0] != target_size[1]:
                    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                crops.append(img)
            panel = cv2.hconcat(crops)
            if not cv2.imwrite(
                str(tmp_dir / frame_filename(i)),
                panel,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            ):
                return False, f"failed to write panel jpeg for {dst}"
        try:
            write_lossless_mp4_from_frame_dir(tmp_dir, Path(dst), fps, n, ffmpeg=ffmpeg)
        except RuntimeError as exc:
            return False, str(exc)
    return True, None


def _reconstruct_episode(task: tuple) -> tuple[bool, str | None]:
    gen_views, gt_views, gen_dst, gt_dst, fps, ffmpeg = task
    ok, err = _write_panel_mp4(gen_views, gen_dst, fps, ffmpeg)
    if not ok:
        return False, err
    return _write_panel_mp4(gt_views, gt_dst, fps, ffmpeg)


def build_flat_from_views(
    split_records: list[dict[str, Any]],
    out_root: Path,
    fps: int,
    workers: int,
    ffmpeg: str,
) -> dict[str, Any]:
    gen_root = out_root / "flat" / "generated"
    gt_root = out_root / "flat" / "gt"
    gen_root.mkdir(parents=True, exist_ok=True)
    gt_root.mkdir(parents=True, exist_ok=True)
    data_gen_root = out_root / "data" / "generated_dataset"

    groups: dict[str, dict[str, Any]] = {}
    for rec in split_records:
        groups.setdefault(rec["episode"], {})[rec["view"]] = rec

    jobs: list[tuple] = []
    summary: list[dict[str, Any]] = []
    skipped: list[str] = []
    for episode, views in sorted(groups.items(), key=lambda item: episode_sort_key(item[0])):
        if not all(v in views for v in VIEW_ORDER):
            skipped.append(episode)
            continue
        gen_views = [str(views[v]["gen_video"]) for v in VIEW_ORDER]
        gt_views = [str(views[v]["gt_video"]) for v in VIEW_ORDER]
        gen_dst = gen_root / f"{episode}.mp4"
        gt_dst = gt_root / f"{episode}.mp4"
        jobs.append((gen_views, gt_views, str(gen_dst), str(gt_dst), fps, ffmpeg))
        prompt = read_prompt_txt(Path(views["head"]["gt_video"]).parent / "prompt" / "prompt.txt")
        summary.append(
            {
                "video": str(gen_dst),
                "gt_path": str(gt_dst),
                "prompt": prompt,
                "episode": episode,
                "gen_views": {v: str(data_gen_root / v / episode / "1" / "video") for v in VIEW_ORDER},
            }
        )

    failures: list[str] = []
    if jobs:
        with Pool(processes=max(1, workers)) as pool:
            for ok, err in pool.imap_unordered(_reconstruct_episode, jobs, chunksize=1):
                if not ok and err:
                    failures.append(err)
    if failures:
        for err in failures[:10]:
            print(f"[flat]   FAIL {err}")
        raise RuntimeError(f"panel reconstruction: {len(failures)} episode(s) failed")

    summary_path = out_root / "vlm_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[flat] reconstructed {len(jobs)} episode panel mp4 pairs (gen+gt); skipped {len(skipped)} incomplete")
    return {"generated": gen_root, "gt": gt_root, "summary": summary_path, "episodes": len(jobs)}


def main() -> None:
    args = parse_args()
    ensure_clean(args.output_root, args.clean)

    split_records = discover_split_views(args.split_root)
    if args.max_split_records > 0:
        ordered: list[str] = []
        for rec in split_records:
            if rec["episode"] not in ordered:
                ordered.append(rec["episode"])
        keep = set(ordered[: args.max_split_records])
        split_records = [r for r in split_records if r["episode"] in keep]

    standard = build_standard_tree(split_records, args.output_root)
    flat = build_flat_from_views(
        split_records, args.output_root, args.flat_fps, args.flat_workers, args.ffmpeg
    )

    episodes = sort_episode_names({rec["episode"] for rec in split_records})
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": args.dataset_name,
        "split_root": str(args.split_root),
        "output_root": str(args.output_root),
        "standard_data": {key: str(value) for key, value in standard.items()},
        "flat_data": {key: str(value) for key, value in flat.items() if key != "episodes"},
        "counts": {
            "episodes": len(episodes),
            "split_view_records": len(split_records),
            "panel_mp4_episodes": flat["episodes"],
            "views": list(VIEW_ORDER),
        },
        "sample_records": split_records[:5],
    }

    manifest_path = args.output_root / "workspace_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
