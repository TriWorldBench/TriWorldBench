#!/usr/bin/env python3
"""Preprocess input/<method> multi-view MP4s into flat GT-compatible episode trees.

Input layout (per method):
  input/<method>/episodeN/
    head.mp4, left.mp4, right.mp4

GT reference:
  test_data/gt_dataset/episodeN/
    episodeN.json
    {head,left,right}/frames/frame_*.jpg

Output:
  generate/<method>_<input_ts>_<run_ts>/
    input_ts = batch id (legacy: ennerverse_generated_<input_ts>);
    first run without --input-ts uses run_ts as batch id -> <method>_<run_ts>/

Frame alignment uses index-uniform resampling to match GT frame count (fps-agnostic).
Resolution is resized to target (default 320x240).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR / "lib"))
sys.path.insert(0, str(PROJECT_ROOT))

from eval_data_common import (  # noqa: E402
    FRAME_EXT,
    VIEW_NAMES,
    frame_filename,
    probe_video_frame_count,
    read_json,
    resolve_frame_dir,
    run_cmd,
    uniform_frame_indices,
)
from output_naming import (  # noqa: E402
    format_output_dir,
    is_run_timestamp,
    now_run_timestamp,
    write_latest_pointer,
)
from pipeline_ui import register_terminal_reset, reset_terminal  # noqa: E402
from metrics._common.episode_sort import episode_sort_key, sort_episode_items  # noqa: E402
register_terminal_reset()

EP_RE = re.compile(r"^episode(\d+)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", help="Method folder name under input/, e.g. ennerverse")
    parser.add_argument(
        "--input-ts",
        default=None,
        help=(
            "Batch timestamp for this inference input (YYYYMMDD_HHMMSS). "
            "Omit on first preprocess (defaults to run time). "
            "Reuse when re-preprocessing the same input batch."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config.yaml; uses gt_dataset path when --gt-root is unset.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "input",
        help="Root containing method folders.",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=None,
        help="Flat GT reference root (default: config gt_dataset or test_data/gt_dataset).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "test_data" / "gt_manifest.csv",
        help="GT manifest CSV (generated if missing).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "generate",
        help="Parent directory for generated output.",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--target-width", type=int, default=320)
    parser.add_argument("--target-height", type=int, default=240)
    parser.add_argument(
        "--align-mode",
        choices=("match_gt", "min", "none"),
        default="match_gt",
        help="Frame alignment: match_gt=resample to GT count, min=min(infer,gt), none=keep infer.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_gt_root(args: argparse.Namespace) -> Path:
    if args.gt_root is not None:
        return args.gt_root.resolve()
    if args.config and args.config.is_file():
        with args.config.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for key in ("gt_dataset", "gt_root"):
            if cfg.get(key):
                path = Path(cfg[key])
                if not path.is_absolute():
                    path = (PROJECT_ROOT / path).resolve()
                return path
    return (PROJECT_ROOT / "test_data" / "gt_dataset").resolve()


def load_gt_manifest(path: Path, gt_root: Path) -> dict[str, dict]:
    if not path.is_file():
        import subprocess

        build_script = SCRIPT_DIR / "build_gt_manifest.py"
        subprocess.run([sys.executable, str(build_script), "--gt-root", str(gt_root)], check=True)

    manifest: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            manifest[row["episode"]] = {
                "frame_count": int(row["frame_count"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
            }
    return manifest


def discover_episodes(input_dir: Path, gt_root: Path, manifest: dict[str, dict]) -> list[str]:
    episodes = []
    for ep_dir in sorted(input_dir.iterdir(), key=lambda p: episode_sort_key(p.name)):
        if not ep_dir.is_dir() or not EP_RE.match(ep_dir.name):
            continue
        episode = ep_dir.name
        if episode not in manifest:
            print(f"[skip] {episode} not in GT manifest")
            continue
        if not all((ep_dir / f"{view}.mp4").is_file() for view in VIEW_NAMES):
            print(f"[skip] {episode} missing view mp4s")
            continue
        gt_dir = gt_root / episode
        if not gt_dir.is_dir():
            print(f"[skip] {episode} missing GT dir")
            continue
        episodes.append(episode)
    return episodes


def ensure_episode_json(output_dir: Path, gt_dir: Path, episode: str, overwrite: bool) -> None:
    dst = output_dir / f"{episode}.json"
    if dst.is_file() and not overwrite:
        return
    src = gt_dir / f"{episode}.json"
    if not src.is_file():
        raise FileNotFoundError(f"GT json missing: {src}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def compute_target_frames(
    infer_count: int,
    gt_count: int,
    align_mode: str,
) -> int:
    if align_mode == "match_gt":
        return gt_count
    if align_mode == "min":
        return min(infer_count, gt_count)
    return infer_count


def extract_view_frames(
    video_path: Path,
    output_dir: Path,
    ffmpeg: str,
    overwrite: bool,
    target_frames: int,
    target_size: tuple[int, int],
) -> int:
    existing = sorted(output_dir.glob(f"frame_*{FRAME_EXT}"))
    if existing and not overwrite:
        return len(existing)

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    width, height = target_size
    temp_dir = output_dir / ".extract_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    try:
        # Decode every frame first so count/resample is fps-agnostic.
        temp_pattern = str(temp_dir / f"frame_%05d{FRAME_EXT}")
        run_cmd([
            ffmpeg, "-y", "-i", str(video_path),
            "-vf", f"scale={width}:{height}:flags=lanczos",
            "-q:v", "2",
            "-vsync", "0",
            "-start_number", "0",
            temp_pattern,
        ])

        source_frames = sorted(temp_dir.glob(f"frame_*{FRAME_EXT}"))
        if not source_frames:
            raise RuntimeError(f"No frames extracted: {video_path}")

        source_count = len(source_frames)
        indices = uniform_frame_indices(source_count, target_frames)
        for out_i, src_i in enumerate(indices):
            shutil.copy2(source_frames[src_i], output_dir / frame_filename(out_i))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    frames = sorted(output_dir.glob(f"frame_*{FRAME_EXT}"))
    if len(frames) != target_frames:
        raise RuntimeError(
            f"Frame count mismatch for {video_path}: expected {target_frames}, got {len(frames)}"
        )
    return len(frames)


def process_episode(payload: dict) -> dict:
    episode = payload["episode"]
    input_dir = Path(payload["input_dir"])
    output_dir = Path(payload["output_dir"])
    gt_dir = Path(payload["gt_dir"])
    gt_info = payload["gt_info"]
    ffmpeg = payload["ffmpeg"]
    overwrite = payload["overwrite"]
    align_mode = payload["align_mode"]
    target_size = (payload["target_width"], payload["target_height"])

    ensure_episode_json(output_dir, gt_dir, episode, overwrite)

    infer_counts = {
        view: probe_video_frame_count(input_dir / f"{view}.mp4", ffmpeg)
        for view in VIEW_NAMES
    }
    gt_count = gt_info["frame_count"]
    target_frames = compute_target_frames(min(infer_counts.values()), gt_count, align_mode)

    frame_counts: dict[str, int] = {}
    for view in VIEW_NAMES:
        out_dir = output_dir / view / "frames"
        frame_counts[view] = extract_view_frames(
            input_dir / f"{view}.mp4",
            out_dir,
            ffmpeg,
            overwrite,
            target_frames,
            target_size,
        )

    return {
        "episode": episode,
        "output_dir": str(output_dir),
        "target_frames": target_frames,
        "target_size": {"width": target_size[0], "height": target_size[1]},
        "infer_counts": infer_counts,
        "frames": frame_counts,
    }


def main() -> int:
    args = parse_args()
    exit_code = 1
    try:
        method = args.method
        input_dir = (args.input_root / method).resolve()
        gt_root = resolve_gt_root(args)
        manifest = load_gt_manifest(args.manifest.resolve(), gt_root)

        if not input_dir.is_dir():
            print(f"[error] input not found: {input_dir}", file=sys.stderr)
            return 1

        episodes = discover_episodes(input_dir, gt_root, manifest)
        if args.limit > 0:
            episodes = episodes[: args.limit]
        if not episodes:
            print("[error] no episodes to process", file=sys.stderr)
            return 1

        if args.input_ts and not is_run_timestamp(args.input_ts):
            print(f"[error] invalid --input-ts: {args.input_ts}", file=sys.stderr)
            return 1

        run_ts = now_run_timestamp()
        batch_ts = args.input_ts or run_ts
        output_root = (args.output_root / format_output_dir(method, batch_ts, run_ts)).resolve()

        if args.dry_run:
            for ep in episodes:
                print(f"[dry-run] {ep} -> {output_root / ep}")
            print(f"total: {len(episodes)}")
            exit_code = 0
            return exit_code

        if args.overwrite and output_root.exists():
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        latest_pointer = write_latest_pointer(
            args.output_root.resolve(), method, batch_ts, output_root.name
        )

        common = {
            "ffmpeg": args.ffmpeg,
            "overwrite": args.overwrite,
            "align_mode": args.align_mode,
            "target_width": args.target_width,
            "target_height": args.target_height,
        }

        manifest_entries: list[dict] = []
        failed: list[dict] = []
        workers = min(max(args.workers, 1), len(episodes))
        print(
            f"[info] method={method} batch_ts={batch_ts} "
            f"episodes={len(episodes)} workers={workers} -> {output_root}"
        )

        jobs = [
            {
                "episode": ep,
                "input_dir": str(input_dir / ep),
                "output_dir": str(output_root / ep),
                "gt_dir": str(gt_root / ep),
                "gt_info": manifest[ep],
                **common,
            }
            for ep in episodes
        ]

        if workers == 1:
            for job in jobs:
                try:
                    entry = process_episode(job)
                    manifest_entries.append(entry)
                    print(f"[ok] {job['episode']} frames={entry['frames']['head']}")
                except Exception as exc:
                    failed.append({"episode": job["episode"], "error": str(exc)})
                    print(f"[fail] {job['episode']}: {exc}")
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(process_episode, job): job for job in jobs}
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        entry = future.result()
                        manifest_entries.append(entry)
                        print(f"[ok] {job['episode']} frames={entry['frames']['head']}")
                    except Exception as exc:
                        failed.append({"episode": job["episode"], "error": str(exc)})
                        print(f"[fail] {job['episode']}: {exc}")

        manifest_entries = sort_episode_items(manifest_entries)
        manifest_path = output_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "method": method,
                    "input_ts": batch_ts,
                    "run_ts": run_ts,
                    "input_dir": str(input_dir),
                    "output_root": str(output_root),
                    "gt_root": str(gt_root),
                    "align_mode": args.align_mode,
                    "target_size": {"width": args.target_width, "height": args.target_height},
                    "episodes": len(manifest_entries),
                    "failed": len(failed),
                    "entries": manifest_entries,
                    "failures": failed,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"manifest: {manifest_path}")
        print(f"latest:   {latest_pointer} -> {output_root.name}")
        exit_code = 1 if failed else 0
        return exit_code
    finally:
        reset_terminal()


if __name__ == "__main__":
    raise SystemExit(main())
